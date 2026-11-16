// Toolbelt web app: videos + processing queue, rally review, stat sheet.
// Plain JS, no build step. Pages are routed by the URL hash:
//   #/                    videos
//   #/queue               processing jobs
//   #/stats?v=a,b&src=pipeline
//   #/review/<set_id>[/<rally_id>][?all=1]
"use strict";

const view = document.getElementById("view");
let META = null;
let timer = null;                       // the page's refresh timer, cleared on navigation

// ---- helpers ------------------------------------------------------------------------------
// Lists that rarely change are kept for a minute, so switching pages over the VS Code tunnel
// (every request a slow round trip through a relay) doesn't refetch them. Any change made in
// the app clears them; { fresh: true } always asks the server (and refills the cache).
const CACHED = new Set(["/videos", "/folders", "/rosters"]);
const CACHE_MS = 60000;
const cache = new Map();                // path -> { at, promise }

function api(path, opts = {}) {
  const { fresh, ...rest } = opts;
  const write = rest.body !== undefined || (rest.method && rest.method !== "GET");
  if (write) cache.clear();
  if (write || !CACHED.has(path)) return request(path, rest);
  const hit = cache.get(path);
  if (!fresh && hit && Date.now() - hit.at < CACHE_MS) return hit.promise;
  const promise = request(path, rest);
  cache.set(path, { at: Date.now(), promise });
  promise.catch(() => { if (cache.get(path)?.promise === promise) cache.delete(path); });
  return promise;
}

async function request(path, opts) {
  const init = { ...opts };
  if (opts.body !== undefined) {
    init.method = init.method || "POST";
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(opts.body);
  }
  const res = await fetch("/api" + path, init);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (_) { /* not JSON */ }
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  const type = res.headers.get("content-type") || "";
  // The app never answers /api with a web page; the devtunnels relay does (its sign-in or
  // "connecting to a developer tunnel" page) when its session runs out.
  if (type.includes("html")) {
    throw new Error("the tunnel sent a web page instead of data (its sign-in or warning " +
      "page). Reload this page, sign in or press Continue, then try again.");
  }
  return type.includes("json") ? res.json() : res.text();
}

function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "class") el.className = v;
    else if (k === "value") el.value = v;
    else if (v === true) el.setAttribute(k, "");
    else el.setAttribute(k, v);
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

function toast(msg, ms = 2500) {
  const t = document.getElementById("toast");
  t.textContent = msg; t.hidden = false;
  clearTimeout(toast.t); toast.t = setTimeout(() => { t.hidden = true; }, ms);
}

function clock(t) {
  if (t === null || t === undefined) return "-";
  const m = Math.floor(t / 60), s = t - 60 * m;
  return m + ":" + s.toFixed(3).padStart(6, "0");
}

function ago(ts) {
  if (!ts) return "";
  const s = Math.round(Date.now() / 1000 - ts);
  if (s < 90) return s + " s";
  if (s < 5400) return Math.round(s / 60) + " min";
  return (s / 3600).toFixed(1) + " h";
}

function label(s) { return (s || "-").replaceAll("_", " "); }

function seg(options, current, onPick) {
  const wrap = h("div", { class: "seg" });
  for (const [value, text] of options) {
    const b = h("button", { class: value === current ? "on" : "", type: "button",
      onclick: () => {
        wrap.querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b));
        onPick(value);
      } }, text);
    wrap.append(b);
  }
  return wrap;
}

function select(options, current, onChange, blank) {
  const el = h("select", { onchange: () => onChange(el.value || null) });
  if (blank !== undefined) el.append(h("option", { value: "" }, blank));
  for (const o of options) {
    el.append(h("option", { value: o, selected: o === current }, label(o)));
  }
  if (current === null || current === undefined) el.value = "";
  return el;
}

function parseHash() {
  const raw = location.hash.replace(/^#\/?/, "");
  const [path, query] = raw.split("?");
  return { parts: path.split("/").filter(Boolean), params: new URLSearchParams(query || "") };
}

// ---- router -------------------------------------------------------------------------------
async function route() {
  clearInterval(timer); timer = null;
  document.onkeydown = null;
  if (!META) {
    try { META = await api("/meta"); } catch (e) { view.replaceChildren(errorBox(e)); return; }
  }
  const { parts, params } = parseHash();
  const page = parts[0] || "videos";
  document.querySelectorAll("nav a").forEach(a =>
    a.classList.toggle("on", a.dataset.nav ===
      (page === "review" || page === "calibrate" ? "videos" : page)));
  // Say so when a page takes a moment (over the tunnel), instead of showing the old one.
  const old = view.firstChild;
  const slow = setTimeout(() => { if (view.firstChild === old)
    view.replaceChildren(h("p", { class: "muted" }, "Loading…")); }, 250);
  try {
    if (page === "queue") await queuePage();
    else if (page === "stats") await statsPage(params);
    else if (page === "search") await searchPage(params);
    else if (page === "review") await reviewPage(+parts[1], parts[2] ? +parts[2] : null, params);
    else if (page === "calibrate") await calibratePage(decodeURIComponent(parts[1]), params);
    else if (page === "rosters") await rostersPage(parts[1]);
    else await videosPage();
  } catch (e) {
    view.replaceChildren(errorBox(e));
  } finally {
    clearTimeout(slow);
  }
}

function errorBox(e) {
  return h("div", { class: "card" }, h("strong", {}, "Something went wrong: "), e.message);
}

window.addEventListener("hashchange", route);
route();

// ---- videos -------------------------------------------------------------------------------
async function videosPage() {
  const draw = async (fresh = false) => {
    const vids = await api("/videos", { fresh });
    const list = vids.map(videoCard);
    view.replaceChildren(
      h("h1", {}, META.team + ": videos"),
      h("p", { class: "muted small" }, "Process a video on this PC's GPU, then review its " +
        "rallies and read the stats. Least confident rallies come first in review."),
      ...list);
  };
  await draw();
  timer = setInterval(() => draw(true).catch(() => {}), 5000);
}

function videoCard(v) {
  const cal = v.calibration;
  const calibrated = cal && cal.usable;
  const calChip = !cal ? h("span", { class: "chip bad" }, "not calibrated")
    : !calibrated ? h("span", { class: "chip bad" }, "auto fit only: tap the court")
    : h("span", { class: "chip " + (cal.needs_review ? "warn" : "good") },
        `calibrated` + (cal.rms_px ? ` (${cal.rms_px} px)` : "") +
        (cal.needs_review ? ", check it" : ""));
  const steps = Object.entries(v.cache).map(([k, ok]) =>
    h("span", { class: "chip " + (ok ? "good" : "") }, (ok ? "✓ " : "· ") + k));
  const job = v.job;
  const active = job && (job.status === "queued" || job.status === "running");
  const jobLine = job ? h("div", { class: "small muted" },
    `job ${job.id}: ${label(job.status)}` + (job.step_name ? `, ${job.step_name}` : "") +
    (job.status === "running" && job.progress ? ` · ${job.progress}` : "")) : null;
  const buttons = h("div", { class: "row" },
    v.has_video ? h("button", { class: calibrated ? "" : "primary", onclick: () => {
      location.hash = `#/calibrate/${v.key}`; } }, calibrated ? "Recalibrate" : "Calibrate")
      : null,
    v.set_id ? h("button", { class: "primary", onclick: () => {
      location.hash = `#/review/${v.set_id}`; } },
      `Review (${v.reviewed}/${v.rallies})`) : null,
    v.rallies ? h("button", { onclick: () => { location.hash = `#/stats?v=${v.key}`; } },
      "Stats") : null,
    v.has_video && !active ? h("button", { onclick: async () => {
      try { await api("/jobs", { body: { video: v.key } }); toast("Queued " + v.key); route(); }
      catch (e) { toast(e.message, 5000); } } }, "Process") : null);
  return h("div", { class: "card" },
    h("div", { class: "row spread" },
      h("div", {}, h("strong", {}, v.key),
        h("div", { class: "small muted" },
          [v.opponent && "vs " + v.opponent, v.venue, v.played_on].filter(Boolean).join(" · "))),
      buttons),
    h("div", { class: "row", style: "margin-top:8px" },
      v.has_video ? null : h("span", { class: "chip" }, "video not on this machine"),
      calChip,
      h("a", { class: "chip " + (v.roster ? "good" : ""), href: "#/rosters" +
        (v.roster ? `/${v.roster.slug}` : "") }, v.roster ? `roster: ${v.roster.name}` : "no roster"),
      checksChip(v),
      ...steps),
    jobLine);
}

// The set's score and sanity checks (rally/sanity.py): legal end, flagged rallies.
const CHECK_STATUS = { ok: ["good", "score ok"], past_end: ["bad", "score past the set end"],
  short: ["warn", "set ends short"], last_point_unknown: ["good", "last point to set"] };
function checksChip(v) {
  const c = v.checks;
  if (!c) return null;
  const [cls, text] = CHECK_STATUS[c.status] || ["", c.status];
  return h("a", { class: "chip " + cls, title: c.message,
    href: v.set_id ? `#/review/${v.set_id}?mode=flagged` : "#/" },
    `set ${c.set_number}: ${c.score[0]}-${c.score[1]}, ${text}` +
    (c.flagged ? ` · ${c.flagged} flagged` : ""));
}

// ---- queue --------------------------------------------------------------------------------
function bar(share, text) {
  const pct = Math.round(100 * Math.max(0, Math.min(1, share || 0)));
  return h("div", { class: "bar", title: `${pct}%` },
    h("div", { class: "fill", style: `width:${pct}%` }),
    h("span", {}, `${text} ${pct}%`));
}

async function queuePage() {
  let openLog = null;
  // One log box, kept across refreshes so it doesn't flicker or lose its scroll position.
  const logPre = h("pre", { class: "log" }, "loading…");
  const refreshLog = async () => {
    if (openLog === null) return;
    const atEnd = logPre.scrollHeight - logPre.scrollTop - logPre.clientHeight < 40;
    try { logPre.textContent = (await api(`/jobs/${openLog}/log`)) || "(empty)"; }
    catch (e) { logPre.textContent = e.message; }
    if (atEnd) logPre.scrollTop = logPre.scrollHeight;
  };
  const draw = async () => {
    const jobs = await api("/jobs");
    const cards = jobs.map(j => {
      const n = j.steps.length;
      const done = j.status === "done" ? n : j.step;
      const chipClass = { done: "good", running: "warn", failed: "bad",
        needs_calibration: "bad", interrupted: "bad" }[j.status] || "";
      const actions = h("div", { class: "row" },
        ["queued", "running"].includes(j.status) ? h("button", { onclick: async () => {
          await api(`/jobs/${j.id}/cancel`, { body: {} }); draw(); } }, "Cancel") : null,
        ["failed", "cancelled", "needs_calibration", "interrupted"].includes(j.status)
          ? h("button", { onclick: async () => {
            try { await api(`/jobs/${j.id}/resume`, { body: {} }); draw(); }
            catch (e) { toast(e.message, 5000); } } }, "Resume") : null,
        h("button", { onclick: () => {
          openLog = openLog === j.id ? null : j.id;
          logPre.textContent = "loading…"; draw().then(refreshLog); } },
          openLog === j.id ? "Hide log" : "Log"));
      const live = j.status === "running";
      const bars = (live || j.status === "queued") ? h("div", { style: "margin:6px 0" },
        bar(j.share, "all steps"),
        live ? bar(j.step_share, `${j.step_name}` +
          (j.step_share === null ? " (no progress count yet)" : "") +
          (j.step_eta_min !== null ? `, ~${Math.round(j.step_eta_min)} min left` : "")) : null)
        : null;
      const card = h("div", { class: "card" },
        h("div", { class: "row spread" },
          h("div", {}, h("strong", {}, `#${j.id} ${j.video}`), " ",
            h("span", { class: "chip " + chipClass }, label(j.status))),
          actions),
        h("div", { class: "small" }, `step ${Math.min(done + 1, n)}/${n}: ` +
          (j.step_name || "finished") +
          (j.status === "running" ? ` (${ago(j.step_started)})` : "")),
        bars,
        j.progress ? h("div", { class: "small mono muted" }, j.progress) : null,
        j.error ? h("div", { class: "small", style: "color:var(--bad)" }, j.error) : null,
        h("div", { class: "small muted" }, "queued " + ago(j.created) + " ago" +
          (j.started ? ", started " + ago(j.started) + " ago" : "")));
      if (openLog === j.id) card.append(logPre);
      return card;
    });
    view.replaceChildren(h("h1", {}, "Processing queue"),
      h("p", { class: "muted small" }, "One video at a time on this PC's GPU. Steps: " +
        META.steps.join(" → ") + ". Detection resumes where it stopped."),
      ...(cards.length ? cards : [h("p", { class: "muted" }, "Nothing queued yet.")]));
  };
  await draw();
  timer = setInterval(() => { draw().then(refreshLog).catch(() => {}); }, 3000);
}

// ---- stats --------------------------------------------------------------------------------
async function statsPage(params) {
  const [allVids, fd] = await Promise.all([api("/videos"), api("/folders")]);
  const vids = allVids.filter(v => v.rallies);
  let chosen = (params.get("v") || "").split(",").filter(Boolean);
  if (!chosen.length && vids.length) chosen = [vids[0].key];
  const src = params.get("src") || "pipeline";
  // Ask for the summary now, alongside the stat sheet, not after it.
  const summaryP = src === "pipeline" ? summaryFetch(chosen) : null;
  const situP = src === "pipeline" ? situationalFetch(chosen) : null;
  let scale = params.get("pass");
  if (scale !== "3" && scale !== "4") {
    try { scale = localStorage.getItem("passScale"); } catch (_) { scale = null; }
    if (scale !== "3" && scale !== "4") scale = "4";
  }
  const go = (keys, s, p = scale) => {
    try { localStorage.setItem("passScale", p); } catch (_) { /* private mode */ }
    location.hash = `#/stats?v=${keys.join(",")}&src=${s}&pass=${p}`; };

  // Sets in folders: tournament -> game -> sets. "Edit folders" moves sets between them.
  const folderTree = folderPicker(fd, vids, chosen, keys => go(keys, src));
  const tournaments = [...new Set(Object.values(fd.folders).map(f => f.tournament)
    .filter(t => t !== fd.unfiled))];
  const games = [...new Set(Object.values(fd.folders).map(f => f.game))];
  const editor = h("div", { class: "card", style: "display:none" },
    h("div", { class: "small muted" }, "Type a tournament and a game for each set (pick from " +
      "the list or type a new one). Leave the tournament empty for none."),
    h("datalist", { id: "dl-tournaments" }, ...tournaments.map(t => h("option", { value: t }))),
    h("datalist", { id: "dl-games" }, ...games.map(g => h("option", { value: g }))),
    h("div", { class: "small muted", style: "margin-top:8px" }, "Match format per game (sets " +
      "to 25 or 15, best of 3 or 5): the score checks use it."),
    ...[...new Set(vids.map(v => fd.folders[v.key].game))].sort().map(g => {
      const cur = fd.formats[g] || fd.presets[fd.default_format];
      const pick = h("select", { onchange: async e => {
        try {
          await api(`/games/${encodeURIComponent(g)}/format`, { method: "PUT",
            body: { preset: e.target.value } });
          toast(`${g}: ${fd.presets[e.target.value].label}`);
        } catch (err) { toast(err.message, 5000); } } },
      ...Object.entries(fd.presets).map(([k, p]) => h("option", { value: k,
        selected: k === (cur.preset || "") }, p.label)),
      cur.preset ? null : h("option", { value: "", selected: true }, "custom: " + cur.label));
      return h("div", { class: "row folder-edit" }, h("span", { class: "mono small" }, g), pick);
    }),
    h("div", { class: "small muted", style: "margin-top:8px" }, "Which set of its match each " +
      "video is (from the name's _setN unless changed here):"),
    ...vids.map(v => h("div", { class: "row folder-edit" },
      h("span", { class: "mono small" }, v.key),
      h("input", { type: "number", min: 1, max: 9, value: fd.set_numbers[v.key],
        onchange: async e => {
          try {
            const res = await api(`/videos/${v.key}/set_number`, { method: "PUT",
              body: { set_number: +e.target.value || null } });
            toast(`${v.key}: set ${res.set_number}`);
          } catch (err) { toast(err.message, 5000); } } }))),
    h("div", { class: "small muted", style: "margin-top:8px" }, "Folders:"),
    ...vids.map(v => {
      const f = fd.folders[v.key];
      const tIn = h("input", { list: "dl-tournaments", placeholder: "tournament",
        value: f.tournament === fd.unfiled ? "" : f.tournament });
      const gIn = h("input", { list: "dl-games", placeholder: "game", value: f.game });
      const save = async () => {
        try {
          await api(`/videos/${v.key}/folder`, { method: "PUT",
            body: { tournament: tIn.value, game: gIn.value } });
          toast(`${v.key}: saved`);
        } catch (e) { toast(e.message, 5000); }
      };
      tIn.onchange = save; gIn.onchange = save;
      return h("div", { class: "row folder-edit" }, h("span", { class: "mono small" }, v.key),
        tIn, gIn);
    }),
    h("button", { onclick: () => route() }, "Done"));

  const picker = h("div", { class: "card" },
    h("div", { class: "row spread" }, h("strong", {}, "Sets"),
      h("div", { class: "row" },
        h("button", { onclick: () => go(vids.map(v => v.key), src) }, "All"),
        h("button", { onclick: () => { editor.style.display =
          editor.style.display === "none" ? "" : "none"; } }, "Edit folders"),
        seg([["pipeline", "Pipeline + review"], ["labels", "Hand labels"]], src,
          s => go(chosen, s)))),
    folderTree,
    h("div", { class: "row", style: "margin-top:8px" },
      h("span", { class: "small" }, "Passing average on a"),
      seg([["4", "4-point scale"], ["3", "3-point scale (4s count as 3)"]], scale,
        p => go(chosen, src, p))));

  let body;
  try {
    const st = await api(`/stats?videos=${encodeURIComponent(chosen.join(","))}&source=${src}&pass_scale=${scale}`);
    body = statsBody(st, src);
    if (src === "pipeline") {
      const names = new Map(st.players.filter(p => p.player !== "?")
        .map(p => [+p.player, p.name]));
      body.push(summarySection(summaryP, names), situationalSection(situP, names));
    }
  } catch (e) {
    body = [h("p", { class: "muted" }, e.message)];
  }
  view.replaceChildren(h("h1", {}, "Stat sheet"), picker, editor, ...body);
}

// Processed sets as a tournament -> game -> set tree of checkboxes (data/folders.yaml).
// Ticking a tournament or a game picks all of its sets; onPick gets the new list of keys.
function folderPicker(fd, vids, chosen, onPick) {
  const have = new Set(vids.map(v => v.key));
  const tree = fd.tree.map(t => ({ ...t, games: t.games.map(g =>
    ({ ...g, videos: g.videos.filter(k => have.has(k)) })).filter(g => g.videos.length) }))
    .filter(t => t.games.length);
  const pickAll = (keys, on) => onPick(on ? [...new Set([...chosen, ...keys])]
    : chosen.filter(k => !keys.includes(k)));
  const box = (keys, text, cls) => h("label", { class: "check " + cls },
    h("input", { type: "checkbox", checked: keys.every(k => chosen.includes(k)),
      onchange: e => pickAll(keys, e.target.checked) }), text);
  return h("div", { class: "folders" }, ...tree.map(t => {
    const tKeys = t.games.flatMap(g => g.videos);
    return h("div", { class: "folder" },
      box(tKeys, t.tournament, "tournament"),
      ...t.games.map(g => h("div", { class: "game" },
        box(g.videos, g.game, "gamebox"),
        h("span", { class: "row" }, ...g.videos.map(k => box([k], k.replace(g.game + "_", ""),
          "chip"))))));
  }));
}

function statsBody(st, src) {
  const t = st.team;
  const pct = (a, b) => b ? Math.round(100 * a / b) + "%" : "-";
  const tiles = h("div", { class: "tiles" },
    tile(`${t.points_us}-${t.points_them}`, "points us - them"),
    tile(pct(t.side_out_won, t.side_out_chances),
      `side-out (${t.side_out_won}/${t.side_out_chances})`),
    tile(pct(t.serve_points_won, t.serve_points_played),
      `points on our serve (${t.serve_points_won}/${t.serve_points_played})`),
    tile(fmt(t.hitting_eff), "hitting efficiency"),
    tile(fmt(t.pass_avg), `pass average, ${st.pass_scale}-point scale (${t.pass_n || 0} ` +
      "scored passes)"));
  const cols = st.columns;
  const head = h("tr", {}, h("th", {}, "Player"), ...cols.map(c => h("th", { title: label(c) },
    st.short[c])));
  const rows = [{ player: "TEAM", ...t, team: true }, ...st.players].map(r =>
    h("tr", { class: r.team ? "team" : "" },
      h("td", {}, r.team ? "TEAM" : (r.player === "?" ? "#?" : "#" + r.player) +
        (r.name && r.player !== "?" ? " " + r.name : "") +
        (r.height_cm ? ` (${Math.round(r.height_cm)} cm)` : "")),
      ...cols.map(c => h("td", {}, fmt(r[c])))));
  const note = src === "pipeline"
    ? `${st.counted} rallies counted; ${st.reviewed} of ${st.rallies} reviewed by you. ` +
      "Unreviewed rallies use the pipeline's guesses (winner from who serves next)."
    : `${st.counted} hand-labelled rallies.`;
  const bad = (st.checks || []).filter(c => c.status !== "ok" &&
    c.status !== "last_point_unknown");
  const checks = bad.length ? h("div", { class: "flags" },
    h("strong", {}, "Score checks: these stats may be off"),
    h("ul", {}, ...bad.map(c => h("li", {}, `${c.video}: ${c.message}` +
      (c.flagged ? ` ${c.flagged} rallies flagged for review.` : ""))))) : null;
  return [tiles, checks, h("p", { class: "small muted" }, note),
    h("div", { class: "scroll" }, h("table", {}, h("thead", {}, head), h("tbody", {}, rows))),
    h("p", { class: "small muted" }, cols.map(c => `${st.short[c]} = ${label(c)}`).join(" · "))];
}

function tile(v, k) { return h("div", { class: "tile" }, h("div", { class: "v" }, v),
  h("div", { class: "k" }, k)); }

function fmt(v) {
  if (v === null || v === undefined) return "-";
  return typeof v === "number" && !Number.isInteger(v) ? v.toFixed(3) : String(v);
}

// ---- review -------------------------------------------------------------------------------
async function reviewPage(setId, rallyId, params) {
  // Modes: the review queue (unreviewed, most doubtful first), the rallies a sanity check
  // flags (reviewed ones too), or every rally in order.
  const mode = params.get("all") === "1" ? "all" : (params.get("mode") || "queue");
  const showAll = mode === "all";
  const list = await api(`/sets/${setId}/rallies`);
  const byId = Object.fromEntries(list.rallies.map(r => [r.id, r]));
  const order = mode === "all" ? list.rallies.map(r => r.id)
    : mode === "flagged" ? list.flagged : list.queue;
  const modeSuffix = mode === "all" ? "?all=1" : mode === "flagged" ? "?mode=flagged" : "";
  if (!rallyId) {
    if (!order.length) {
      view.replaceChildren(h("h1", {}, "Review"),
        h("p", {}, mode === "flagged" ? "No rally of this set is flagged by the checks. " :
          "Every rally of this set is reviewed. "),
        h("p", { class: "small muted" }, list.checks.message),
        h("button", { onclick: () => { location.hash = `#/review/${setId}?all=1`; } },
          "Show all rallies"));
      return;
    }
    location.replace(`#/review/${setId}/${order[0]}` + modeSuffix);
    return;
  }
  let d;
  try {
    d = await api(`/rallies/${rallyId}`);
  } catch (e) {
    // A pipeline re-run replaces unreviewed rallies with new ones (new ids): an old link
    // points at nothing. Reviewed rallies keep their ids.
    if (!byId[rallyId]) {
      toast("That rally was rebuilt by a pipeline re-run (it has a new number now): " +
        "showing this set's review queue.", 6000);
      location.replace(`#/review/${setId}` + modeSuffix);
      return;
    }
    throw e;
  }
  const r = d.rally, fps = r.fps;
  const pad = META.clip_pad_s;
  const S = {                                 // this page's editable state
    before: pad, after: pad,
    start: r.start_t, end: r.end_t,
    touches: d.touches.map(x => ({ ...x, delete: false })),
    winner: r.point_winner || (["replay", "not_a_rally"].includes(r.outcome) ? r.outcome : null),
    outcome: r.reviewed ? r.outcome : null,
    serving: r.serving_team, checked: true, suggested: null, endDetail: r.end_detail || null,
  };
  const t0 = () => Math.max(0, r.start_t - S.before);
  const suffix = modeSuffix;
  const pos = order.indexOf(rallyId);
  const nextId = pos >= 0 && pos + 1 < order.length ? order[pos + 1] : null;

  // header: picker and progress
  const picker = h("select", { class: "picker", onchange: e => {
    location.hash = `#/review/${setId}/${e.target.value}${suffix}`; } },
  ...(order.includes(rallyId) ? order : [rallyId, ...order]).map(id => {
    const x = byId[id];
    return h("option", { value: id, selected: id === rallyId },
      `${x.reviewed ? "✓ " : ""}${x.flags.length ? "⚠ " : ""}#${x.rally_number}  ` +
      `${clock(x.start_t)}  ${x.score[0]}-${x.score[1]}  conf ${(x.priority || 0).toFixed(2)}`);
  }));
  const reviewed = list.rallies.filter(x => x.reviewed).length;
  const top = h("div", { class: "card" },
    h("div", { class: "row spread" },
      h("strong", {}, `${r.video_key}: rally ${r.rally_number}`),
      seg([["queue", "Review queue"], ["flagged", `Flagged (${list.flagged.length})`],
        ["all", "All in order"]], mode, v => {
        location.hash = `#/review/${setId}` + (v === "all" ? "?all=1" :
          v === "flagged" ? "?mode=flagged" : ""); })),
    h("div", { class: "small muted" }, `${reviewed}/${list.rallies.length} reviewed · ` +
      `score from reviewed rallies ${list.score_reviewed[0]}-${list.score_reviewed[1]} · ` +
      `${clock(r.start_t)}-${clock(r.end_t)} (${(r.end_t - r.start_t).toFixed(1)} s)`),
    h("div", { class: "small " + (list.checks.status === "ok" ||
      list.checks.status === "last_point_unknown" ? "muted" : "warntext") },
      `Set ${list.checks.set_number} (${list.checks.format.label}): ${list.checks.message}`),
    d.flags.length ? h("div", { class: "flags" }, h("strong", {}, "Checks flag this rally"),
      h("ul", {}, ...d.flags.map(f => h("li", {}, f)))) : null,
    h("div", { style: "margin-top:8px" }, picker));

  // player
  const video = h("video", { playsinline: true, preload: "auto", controls: false });
  const readout = h("div", { class: "readout" });
  const bar = h("input", { type: "range", min: 0, max: 1000, value: 0 });
  const now = () => t0() + video.currentTime;
  const show = () => {
    readout.textContent = `clip ${video.currentTime.toFixed(3)} s | video ${clock(now())}` +
      ` | frame ${Math.round(now() * fps)}`;
    if (video.duration) bar.value = 1000 * video.currentTime / video.duration;
  };
  const seek = t => { video.pause(); video.currentTime = Math.max(0, t - t0()); };
  const frame = n => { video.pause(); video.currentTime = Math.max(0, video.currentTime + n / fps); };
  const loadClip = () => {
    if (!d.has_video) return;
    video.src = `/api/rallies/${rallyId}/clip?t0=${t0().toFixed(2)}&t1=` +
      `${(r.end_t + S.after).toFixed(2)}`;
  };
  ["timeupdate", "seeked", "loadedmetadata"].forEach(e => video.addEventListener(e, show));
  video.addEventListener("error", () => {
    readout.textContent = "Could not load the clip (is the video on this machine?)"; });
  bar.oninput = () => { if (video.duration) video.currentTime = video.duration * bar.value / 1000; };
  video.onclick = () => (video.paused ? video.play() : video.pause());
  const slowBtn = h("button", { onclick: () => {
    video.playbackRate = video.playbackRate === 1 ? 0.25 : 1;
    slowBtn.classList.toggle("on", video.playbackRate !== 1); } }, "×0.25");
  const widen = (key, label) => h("label", { class: "row small" }, label,
    h("input", { type: "number", min: 0, max: 60, step: 1, value: S[key], onchange: e => {
      S[key] = Math.max(0, +e.target.value || 0); loadClip(); drawTouches(); drawSpan(); } }));
  const player = h("div", {},
    d.has_video ? video : h("div", { class: "card muted" },
      "The video isn't on this machine: review the touches without the clip, or open " +
      "the app on the desktop."),
    readout, bar,
    h("div", { class: "row" },
      h("button", { onclick: () => seek(now() - 0.1) }, "−0.1 s"),
      h("button", { onclick: () => frame(-1) }, "◀ frame"),
      h("button", { class: "primary", onclick: () => (video.paused ? video.play() : video.pause()) },
        "Play / pause"),
      h("button", { onclick: () => frame(1) }, "frame ▶"),
      h("button", { onclick: () => seek(now() + 0.1) }, "+0.1 s"), slowBtn),
    h("div", { class: "row", style: "margin-top:8px" },
      widen("before", "s before"), widen("after", "s after")),
    h("p", { class: "small muted" }, "Keys: , . step a frame · space plays · ↑ ↓ (or k j) " +
      "pick a touch and jump to it · on that touch: s serve, r receive, e set, a attack, " +
      "b block, d dig, p dump, f free ball, o overpass, c coverage, x switches team, " +
      "0-4 pass score, t moves it to now, Delete removes it · n adds a touch now · " +
      "w we won, l they won, R replay, N not a rally · Enter saves and goes to the next, " +
      "Shift+Enter saves. Times below are seconds into the clip (the \"clip\" time above)."),
    mergeButtons());

  function mergeButtons() {
    const b = (x, what) => x && h("button", { onclick: async () => {
      if (!confirm(`Merge this rally with the ${what} one (#${x.rally_number})?`)) return;
      await api(`/rallies/${rallyId}/merge`, { body: { other_id: x.id } });
      toast("Merged"); route(); } },
    `Merge with ${what} #${x.rally_number} (${x.gap_s} s apart)`);
    // Two points counted as one: split where the second point's serve is.
    // Two points counted as one: split where the second point's serve is. The start / end
    // as edited here (e.g. widened back) are applied first.
    const split = h("button", { onclick: async () => {
      const t = now();
      if (!(t > S.start + 0.1 && t < S.end - 0.1)) {
        toast(t <= S.start ? "That is before this rally starts. For a point the pipeline " +
          "missed, use \"Missed point\" below (or move the rally's start back first)."
          : "Go to the second point's serve inside this rally first.", 7000);
        return;
      }
      if (!confirm(`Split this rally at ${clock(t)}? Touches from there on become a new ` +
        "rally (the second point). Unsaved touch edits here are lost.")) return;
      try {
        await api(`/rallies/${rallyId}/split`, { body: { t, start_t: S.start, end_t: S.end } });
        toast("Split into two rallies; both are in the review queue"); route();
      } catch (e) { toast(e.message, 6000); }
    } }, "Split here: a second point starts at this frame");
    return h("div", {}, h("div", { class: "row" }, b(d.prev, "previous"), b(d.next, "next"),
      split), missedBox());
  }

  // A point the pipeline missed entirely (taken for downtime): mark it in the clip, add it.
  function missedBox() {
    const M = { start: null, end: null, serving: null };
    const box = h("div", { class: "card", style: "margin-top:8px" });
    const gapNote = d.prev && d.prev.gap_s > META.missed_point_gap_s
      ? h("div", { class: "small", style: "color:var(--warn)" }, `${d.prev.gap_s} s since the ` +
        "previous rally: points rarely take that long to start, so one may be missing here.")
      : null;
    const draw = () => box.replaceChildren(
      h("strong", {}, "Missed point in this clip?"),
      gapNote,
      h("div", { class: "small muted" }, "Widen the clip (s before / after) to see it, then " +
        "mark where it starts (the serve) and ends."),
      h("div", { class: "row", style: "margin-top:6px" },
        h("button", { onclick: () => { M.start = now(); draw(); } },
          "Starts here" + (M.start !== null ? ` (${clock(M.start)})` : "")),
        h("button", { onclick: () => { M.end = now(); draw(); } },
          "Ends here" + (M.end !== null ? ` (${clock(M.end)})` : "")),
        seg([["us", "We serve"], ["them", "They serve"]], M.serving, v => { M.serving = v; draw(); }),
        h("button", { class: "primary", disabled: M.start === null || M.end === null,
          onclick: async () => {
            if (M.end <= M.start) { toast("It has to end after it starts."); return; }
            try {
              const res = await api(`/sets/${setId}/rallies`, { body: {
                start_t: M.start, end_t: M.end, serving_team: M.serving } });
              toast("Added: now set its touches, winner and how it ended, and save");
              location.hash = `#/review/${setId}/${res.id}?all=1`;
            } catch (e) { toast(e.message, 7000); }
          } }, "Add missed point")));
    draw();
    return box;
  }

  // touches
  const touchBox = h("div");
  const visible = () => S.touches.filter(x => !x.delete).sort((a, b) => a.t - b.t);
  function drawTouches() {
    const rows = visible().map(x => {
      const row = h("div", { class: "touch " + (x.team || "") + (x === S.sel ? " sel" : "") },
        h("button", { class: "iconbtn", title: "go to this touch", onclick: () => {
          S.sel = x; seek(x.t); drawTouches(); } }, "▶"),
        h("input", { type: "number", step: 0.001, value: (x.t - t0()).toFixed(3),
          onchange: e => { x.t = t0() + (+e.target.value); suggest(); } }),
        select(["us", "them"], x.team, v => { x.team = v; drawTouches(); suggest(); }),
        select(META.actions, x.action, v => { x.action = v; drawTouches(); suggest(); }, "action?"),
        h("button", { class: "iconbtn", title: "delete", onclick: () => {
          x.delete = true; drawTouches(); suggest(); } }, "✕"));
      if (x.action_confidence !== undefined && x.action_confidence !== null && x.id) {
        row.append(h("div", { class: "sure" }, `pipeline sure ${x.action_confidence.toFixed(2)}`));
      }
      if (x.team === "us") row.append(extraRow(x));
      return row;
    });
    touchBox.replaceChildren(...rows,
      h("div", { class: "row", style: "margin-top:8px" },
        h("button", { onclick: addTouch }, "+ Touch at the current frame")));
  }

  function addTouch() {
    const x = { id: null, t: now(), team: "us", action: null, delete: false };
    S.touches.push(x); S.sel = x; drawTouches(); suggest();
  }

  // Who touched it (our touches) and, on our receive, the pass score 0-4.
  // A pipeline guess shows how sure it was and a ✓ to confirm it as it is; setting a value
  // confirms it too. Only confirmed ones count as labels (export_reviewed.py).
  function extraRow(x) {
    const guess = (v, c, checked, confirm, from = "pipeline guess") =>
      v !== null && v !== undefined && !checked && c !== null && c !== undefined && c < 1
        ? h("span", { class: "row small muted" }, `${from}, sure ${c.toFixed(2)}`,
          h("button", { class: "iconbtn", title: "it's right", onclick: confirm }, "✓"))
        : null;
    const who = h("select", { onchange: e => {
      x.jersey = e.target.value === "" ? null : +e.target.value; x.jersey_checked = true; drawTouches(); } },
      h("option", { value: "" }, d.roster.length ? "player?" : "no roster for this video"),
      ...d.roster.map(p => h("option", { value: p.number, selected: p.number === x.jersey },
        `#${p.number} ${p.name}`)));
    const parts = [h("label", { class: "row small" }, "Player", who,
      guess(x.jersey, x.player_confidence, x.jersey_checked,
        () => { x.jersey_checked = true; drawTouches(); },
        x.player_from_fix ? "from your fix on this player's track" : "pipeline guess"))];
    if (x.action === "receive") {                 // an overpass on serve receive is an automatic 0
      parts.push(h("span", { class: "row small" }, "Pass",
        seg([0, 1, 2, 3, 4].map(n => [String(n), String(n)]),
          x.pass_score === null || x.pass_score === undefined ? null : String(x.pass_score),
          v => { x.pass_score = +v; x.pass_checked = true; drawTouches(); }),
        guess(x.pass_score, x.pass_confidence, x.pass_checked,
          () => { x.pass_checked = true; drawTouches(); })));
    }
    return h("div", { class: "extra" }, ...parts);
  }

  // rally span
  const spanInput = key => h("input", { type: "number", step: 0.001,
    value: (S[key] - t0()).toFixed(3), onchange: e => { S[key] = t0() + (+e.target.value); suggest(); } });
  const spanBox = h("div");
  function drawSpan() {
    spanBox.replaceChildren(h("div", { class: "row" },
      h("span", { class: "small" }, "Started (serve)"), spanInput("start"),
      h("button", { onclick: () => { S.start = now(); drawSpan(); } }, "= now")),
    h("div", { class: "row", style: "margin-top:6px" },
      h("span", { class: "small" }, "Ended (touches after it go)"), spanInput("end"),
      h("button", { onclick: () => { S.end = now(); drawSpan(); suggest(); } }, "= now")));
  }

  // point
  const pointBox = h("div");
  function drawPoint() {
    const winners = [["us", "Us"], ["them", "Them"], ["replay", "Replay"],
      ["not_a_rally", "Not a rally"]];
    const scored = S.winner === "us" || S.winner === "them";
    const outcomes = META.outcomes.filter(o => o !== "replay");
    const guess = !r.reviewed && r.point_winner ? h("div", { class: "small muted" },
      `Pipeline's guess: ${r.point_winner} (from who serves next)`) : null;
    pointBox.replaceChildren(
      h("div", { class: "small" }, "Who won the point?"), seg(winners, S.winner, v => {
        S.winner = v; drawPoint(); suggest(); }), guess,
      scored ? h("div", { style: "margin-top:8px" },
        h("div", { class: "small" }, `How (suggested from the last touch: ${label(S.suggested)})`),
        select(outcomes, S.outcome || S.suggested, v => { S.outcome = v; })) : null,
      scored ? h("div", { style: "margin-top:8px" },
        h("div", { class: "small" }, "Ended abruptly? (a whistle the video can't show)"),
        select(META.end_details, S.endDetail, v => { S.endDetail = v; },
          "no: the ball decided it")) : null,
      h("div", { class: "small", style: "margin-top:8px" }, "Serving team"),
      seg([["us", "Us"], ["them", "Them"]], S.serving, v => { S.serving = v; drawPoint(); }),
      h("label", { class: "check small", style: "margin-top:8px" },
        h("input", { type: "checkbox", checked: S.checked, disabled: S.winner === "not_a_rally",
          onchange: e => { S.checked = e.target.checked; } }),
        "All touches are right (use them as training data)"));
  }

  let suggestSeq = 0;
  async function suggest() {
    const seq = ++suggestSeq;
    try {
      const res = await api(`/rallies/${rallyId}/suggest`, { body: {
        winner: S.winner, end_t: S.end, touches: S.touches } });
      if (seq !== suggestSeq) return;
      S.suggested = res.outcome;
      drawPoint();
    } catch (_) { /* keep the old suggestion */ }
  }

  async function save(goNext) {
    if (!S.winner) { toast("Set who won the point first."); return; }
    if (S.end <= S.start) { toast("The rally has to end after it starts."); return; }
    const scored = S.winner === "us" || S.winner === "them";
    try {
      await api(`/rallies/${rallyId}/review`, { body: {
        winner: S.winner, outcome: scored ? (S.outcome || S.suggested) : null,
        serving_team: S.serving, start_t: S.start, end_t: S.end,
        touches: S.touches, touches_checked: S.checked,
        end_detail: scored ? S.endDetail : null } });
    } catch (e) { toast(e.message, 5000); return; }
    toast("Saved");
    if (goNext && nextId) location.hash = `#/review/${setId}/${nextId}${suffix}`;
    else if (goNext && !showAll) location.hash = `#/review/${setId}`;
    else route();
  }

  view.replaceChildren(top, h("div", { class: "review" },
    h("div", {}, player),
    h("div", {},
      h("div", { class: "card" }, h("strong", {}, "Touches"), touchBox),
      h("div", { class: "card" }, h("strong", {}, "Rally"), spanBox),
      h("div", { class: "card" }, h("strong", {}, "Point"), pointBox),
      h("div", { class: "row" },
        h("button", { onclick: () => save(false) }, "Save"),
        h("button", { class: "primary", onclick: () => save(true) }, "Save & next")))));
  drawTouches(); drawSpan(); drawPoint(); loadClip(); suggest();

  // Keys (roadmap E4): review a rally without the mouse. Letters act on the selected touch
  // (highlighted; ↑/↓ or j/k move it and jump the clip there).
  const KEY_ACTIONS = { s: "serve", r: "receive", e: "set", a: "attack", b: "block", d: "dig",
    p: "dump", f: "free_ball", o: "overpass", c: "coverage" };
  function pickTouch(step) {
    const v = visible();
    if (!v.length) return;
    const i = v.indexOf(S.sel);
    S.sel = v[i < 0 ? (step > 0 ? 0 : v.length - 1) : Math.max(0, Math.min(v.length - 1, i + step))];
    seek(S.sel.t); drawTouches();
  }
  function setWinner(w) { S.winner = w; drawPoint(); suggest(); }
  document.onkeydown = e => {
    if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const k = e.key, x = S.sel && !S.sel.delete ? S.sel : null;
    let done = true;
    if (k === ",") frame(-1);
    else if (k === ".") frame(1);
    else if (k === " ") video.paused ? video.play() : video.pause();
    else if (k === "Enter") save(!e.shiftKey);
    else if (k === "w") setWinner("us");
    else if (k === "l") setWinner("them");
    else if (k === "R") setWinner("replay");
    else if (k === "N") setWinner("not_a_rally");
    else if (k === "ArrowDown" || k === "j") pickTouch(1);
    else if (k === "ArrowUp" || k === "k") pickTouch(-1);
    else if (k === "n") addTouch();
    else if (x && k === "x") { x.team = x.team === "us" ? "them" : "us"; drawTouches(); suggest(); }
    else if (x && k === "t") { x.t = now(); drawTouches(); suggest(); }
    else if (x && KEY_ACTIONS[k]) { x.action = KEY_ACTIONS[k]; drawTouches(); suggest(); }
    else if (x && /^[0-4]$/.test(k) && x.team === "us" && x.action === "receive") {
      x.pass_score = +k; x.pass_checked = true; drawTouches(); }
    else if (x && (k === "Delete" || k === "Backspace")) {
      const v = visible(), i = v.indexOf(x);
      x.delete = true; S.sel = v[i + 1] || v[i - 1] || null; drawTouches(); suggest(); }
    else done = false;
    if (done) e.preventDefault();
  };
}
