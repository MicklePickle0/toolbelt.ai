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
  // DESIGN.md's two rooms: Review and Calibrate are the dark film room, the rest is paper.
  document.body.classList.toggle("film", page === "review" || page === "calibrate");
  document.querySelectorAll("nav a").forEach(a =>
    a.classList.toggle("on", a.dataset.nav ===
      (page === "review" || page === "calibrate" ? "videos" : page)));
  // Say so when a page takes a moment (over the tunnel), instead of showing the old one.
  const old = view.firstChild;
  const slow = setTimeout(() => { if (view.firstChild === old)
    view.replaceChildren(h("p", { class: "muted" }, "Loading")); }, 250);
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
  return h("p", { class: "bad" }, "Something went wrong: " + e.message);
}

window.addEventListener("hashchange", route);
route();

// ---- videos -------------------------------------------------------------------------------
async function videosPage() {
  const draw = async (fresh = false) => {
    const [vids, fd] = await Promise.all([api("/videos", { fresh }), api("/folders")]);
    const st = vids.map(videoState);
    const n = pred => st.filter(pred).length;
    const review = n(s => s.next === "review"), uncal = n(s => s.next === "calibrate"),
      busy = n(s => s.active);
    const lead = h("p", { class: "lead" }, `${vids.length} video${vids.length === 1 ? "" : "s"}.`,
      ...[[review, "to review"], [uncal, "not calibrated"], [busy, "processing"]]
        .filter(([k]) => k).flatMap(([k, what], i) =>
          [i ? ", " : " ", h("strong", {}, String(k)), " " + what]),
      review || uncal || busy ? "." : " Nothing waiting.");
    // Grouped as in the folders (tournament, then its games), anything unfiled last.
    const byKey = new Map(vids.map((v, i) => [v.key, [v, st[i]]]));
    const seen = new Set();
    const groups = fd.tree.map(t => {
      const keys = t.games.flatMap(g => g.videos).filter(k => byKey.has(k));
      keys.forEach(k => seen.add(k));
      return [t.tournament, keys];
    }).filter(([, keys]) => keys.length);
    const rest = vids.map(v => v.key).filter(k => !seen.has(k));
    if (rest.length) groups.push([fd.unfiled || "Other videos", rest]);
    view.replaceChildren(
      h("h1", {}, "Videos"), lead,
      h("p", { class: "muted small" }, "Process a video on this PC's GPU, then review its " +
        "rallies and read the stats. The least confident rallies come first in review."),
      ...groups.flatMap(([title, keys]) => [h("h2", {}, title),
        h("div", { class: "vrow vhead" }, h("span", {}, "Set"), h("span", {}, "Score"),
          h("span", {}, "Status"), h("span", {})),
        ...keys.map(k => videoRow(...byKey.get(k)))]));
  };
  await draw();
  timer = setInterval(() => draw(true).catch(() => {}), 5000);
}

// What a video needs next, and the one sentence that says so.
function videoState(v) {
  const cal = v.calibration, calibrated = cal && cal.usable;
  const job = v.job, active = job && (job.status === "queued" || job.status === "running");
  const toReview = v.set_id ? v.rallies - v.reviewed : 0;
  let next = null, status;
  if (active) {
    status = job.status === "queued" ? "Queued for processing"
      : `Processing, ${job.step_name || "starting"}` + (job.progress ? `, ${job.progress}` : "");
  } else if (v.has_video && !calibrated) {
    next = "calibrate";
    status = cal ? "Only an automatic court fit: tap the court to calibrate" : "Not calibrated";
  } else if (job && ["failed", "interrupted", "needs_calibration"].includes(job.status)) {
    next = "queue"; status = `Processing stopped: ${label(job.status)}`;
  } else if (toReview > 0) {
    next = "review";
    status = `${toReview} of ${v.rallies} rallies to review`;
  } else if (v.rallies) {
    next = "stats"; status = "Reviewed";
  } else if (v.has_video) {
    next = "process"; status = "Calibrated, not processed yet";
  } else {
    status = "The video isn't on this machine";
  }
  return { next, status, active, calibrated };
}

function videoRow(v, s) {
  const c = v.checks;
  const go = hash => () => { location.hash = hash; };
  const process = async () => {
    try { await api("/jobs", { body: { video: v.key } }); toast("Queued " + v.key); route(); }
    catch (e) { toast(e.message, 5000); } };
  const btn = (kind, text, onclick) => h("button", { class: s.next === kind ? "primary" : "",
    onclick }, text);
  const flagged = c && c.flagged ? h("a", { href: `#/review/${v.set_id}?mode=flagged` },
    `${c.flagged} flagged by the score checks`) : null;
  const scoreBad = c && (c.status === "past_end" || c.status === "short");
  // Only what tells the sets apart; the rest is in the hover text.
  const cal = v.calibration;
  const details = [v.opponent && "against " + v.opponent, v.venue, v.played_on,
    v.roster ? null : "no roster"].filter(Boolean).join(", ");
  const more = [s.calibrated ? "Calibrated" + (cal.rms_px ? ` to ${cal.rms_px} px` : "") +
      (cal.needs_review ? ", worth a check" : "") : null,
    v.roster ? "roster " + v.roster.name : null,
    v.has_video && Object.values(v.cache).some(ok => !ok)
      ? "steps not done: " + Object.entries(v.cache).filter(([, ok]) => !ok).map(([k]) => k).join(", ")
      : null].filter(Boolean).join(", ");
  return h("div", { class: "vrow", title: more },
    h("div", { class: "stack" }, h("span", { class: "name" }, v.key),
      details ? h("span", { class: "small muted" }, details) : null),
    h("span", { class: "num" + (scoreBad ? " bad" : ""), title: c ? c.message : "" },
      c ? `${c.score[0]}-${c.score[1]}` : ""),
    h("div", { class: "stack" }, h("span", { class: s.next === "queue" ? "bad" : "" }, s.status),
      flagged ? h("span", { class: "small" }, flagged) : null),
    h("div", { class: "row acts" },
      v.has_video ? btn("calibrate", s.calibrated ? "Recalibrate" : "Calibrate",
        go(`#/calibrate/${v.key}`)) : null,
      v.set_id ? btn("review", "Review", go(`#/review/${v.set_id}`)) : null,
      v.rallies ? btn("stats", "Stats", go(`#/stats?v=${v.key}`)) : null,
      s.next === "queue" ? btn("queue", "Queue", go("#/queue")) : null,
      v.has_video && !s.active ? btn("process", "Process", process) : null));
}

// ---- queue --------------------------------------------------------------------------------
function bar(share, text) {
  const pct = Math.round(100 * Math.max(0, Math.min(1, share || 0)));
  return h("div", { class: "bar" },
    h("div", { class: "small" }, `${text}, ${pct}%`),
    h("div", { class: "track" }, h("div", { class: "fill", style: `width:${pct}%` })));
}

async function queuePage() {
  let openLog = null;
  // One log box, kept across refreshes so it doesn't flicker or lose its scroll position.
  const logPre = h("pre", { class: "log" }, "loading");
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
      const stopped = ["failed", "needs_calibration", "interrupted"].includes(j.status);
      const actions = h("div", { class: "row" },
        ["queued", "running"].includes(j.status) ? h("button", { onclick: async () => {
          await api(`/jobs/${j.id}/cancel`, { body: {} }); draw(); } }, "Cancel") : null,
        ["failed", "cancelled", "needs_calibration", "interrupted"].includes(j.status)
          ? h("button", { onclick: async () => {
            try { await api(`/jobs/${j.id}/resume`, { body: {} }); draw(); }
            catch (e) { toast(e.message, 5000); } } }, "Resume") : null,
        h("button", { onclick: () => {
          openLog = openLog === j.id ? null : j.id;
          logPre.textContent = "loading"; draw().then(refreshLog); } },
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
          h("div", {}, h("strong", {}, j.video), " ",
            h("span", { class: stopped ? "bad" : "muted" }, label(j.status)),
            h("span", { class: "muted small" }, `, job ${j.id}`)),
          actions),
        h("div", { class: "small" }, `Step ${Math.min(done + 1, n)} of ${n}: ` +
          (j.step_name || "finished") +
          (j.status === "running" ? `, for ${ago(j.step_started)}` : "")),
        bars,
        j.progress ? h("div", { class: "small muted" }, j.progress) : null,
        j.error ? h("div", { class: "small bad" }, j.error) : null,
        h("div", { class: "small muted" }, "Queued " + ago(j.created) + " ago" +
          (j.started ? ", started " + ago(j.started) + " ago" : "")));
      if (openLog === j.id) card.append(logPre);
      return card;
    });
    view.replaceChildren(h("h1", {}, "Queue"),
      h("p", { class: "muted small" }, "One video at a time on this PC's GPU. Steps: " +
        META.steps.join(", ") + ". Detection resumes where it stopped."),
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
  const tab = STAT_TABS.some(([k]) => k === params.get("tab")) ? params.get("tab") : "summary";
  // The tabs past the box score need the game summary and the situational stats: ask for
  // them now, alongside the stat sheet, and only on those tabs.
  const deep = src === "pipeline" && ["passing", "setting", "hitting", "trends"].includes(tab);
  const summaryP = deep ? summaryFetch(chosen) : null;
  const situP = deep ? situationalFetch(chosen) : null;
  let scale = params.get("pass");
  if (scale !== "3" && scale !== "4") {
    try { scale = localStorage.getItem("passScale"); } catch (_) { scale = null; }
    if (scale !== "3" && scale !== "4") scale = "4";
  }
  const hashFor = (keys, s, p, t) => `#/stats?v=${keys.join(",")}&src=${s}&pass=${p}&tab=${t}`;
  const go = (keys, s, p = scale, t = tab) => {
    try { localStorage.setItem("passScale", p); } catch (_) { /* private mode */ }
    location.hash = hashFor(keys, s, p, t); };

  // Sets in folders: tournament -> game -> sets. "Edit folders" moves sets between them.
  const folderTree = scopePicker(fd, vids, chosen, keys => go(keys, src));
  const tournaments = [...new Set(Object.values(fd.folders).map(f => f.tournament)
    .filter(t => t !== fd.unfiled))];
  const games = [...new Set(Object.values(fd.folders).map(f => f.game))];
  const editor = h("div", { class: "well", style: "display:none" },
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
      return h("div", { class: "row folder-edit" }, h("span", { class: "small" }, g), pick);
    }),
    h("div", { class: "small muted", style: "margin-top:8px" }, "Which set of its match each " +
      "video is (from the name's _setN unless changed here):"),
    ...vids.map(v => h("div", { class: "row folder-edit" },
      h("span", { class: "small" }, v.key),
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
      return h("div", { class: "row folder-edit" }, h("span", { class: "small" }, v.key),
        tIn, gIn);
    }),
    h("button", { onclick: () => route() }, "Done"));

  // One row of plain dropdowns: which sets, where the touches come from, the pass scale.
  const choice = (options, current, onChange) => h("select", {
    onchange: e => onChange(e.target.value) },
  ...options.map(([v, text]) => h("option", { value: v, selected: v === current }, text)));
  const picker = h("div", { class: "filterbar" },
    ...folderTree,
    h("label", { class: "fl" }, "Touches from",
      choice([["pipeline", "Pipeline and review"], ["labels", "Hand labels"]], src,
        s => go(chosen, s))),
    h("label", { class: "fl" }, "Pass average",
      choice([["4", "4-point scale"], ["3", "3-point scale (4s count as 3)"]], scale,
        p => go(chosen, src, p))),
    h("button", { class: "quiet", onclick: () => { editor.style.display =
      editor.style.display === "none" ? "" : "none"; } }, "Edit folders"));

  // One tab at a time, each its own address (back and forward work, a tab can be linked).
  const tabs = h("nav", { class: "tabs", "aria-label": "Stats" }, ...STAT_TABS.map(([k, text]) =>
    h("a", { href: hashFor(chosen, src, scale, k), class: k === tab ? "on" : "",
      "aria-current": k === tab ? "page" : null }, text)));

  let body;
  try {
    const st = await api(`/stats?videos=${encodeURIComponent(chosen.join(","))}&source=${src}&pass_scale=${scale}`);
    const names = new Map(st.players.filter(p => p.player !== "?")
      .map(p => [+p.player, p.name]));
    if (tab === "summary") {
      body = [keyFigures(st, chosen.length), scoreChecks(st),
        rallyStrips(vids.filter(v => chosen.includes(v.key)))];
    } else if (tab === "box") {
      body = statsBody(st, src);
    } else if (!deep) {
      body = [h("p", {}, "This tab needs the pipeline's touches: set Touches from to " +
        "Pipeline and review.")];
    } else {
      const [sm, si] = await Promise.all([summaryP, situP]);
      const parts = situationalParts(si, names);
      body = tab === "passing" ? [...summaryPassing(sm, names), ...parts.passing]
        : tab === "setting" ? [...parts.setting, ...summarySets(sm, names)]
          : tab === "hitting" ? parts.hitting : parts.trends;
    }
  } catch (e) {
    body = [h("p", { class: "bad" }, e.message)];
  }
  // The title says what is shown: "Pakmen Black, set 2", "Pakmen Black", "Humber 17U".
  const title = scopeTitle(fd, vids, chosen);
  // replaceChildren prints a null as the text "null": leave the empty parts out.
  view.replaceChildren(...[h("h1", {}, title), picker, editor, tabs, ...body].filter(Boolean));
}

const STAT_TABS = [["summary", "Summary"], ["box", "Box score"], ["passing", "Passing"],
  ["setting", "Setting"], ["hitting", "Hitting"], ["trends", "Trends"]];

// Summary tab: the numbers a coach looks for first, large, each with what it is out of.
function keyFigures(st, nSets) {
  const t = st.team;
  const pct = (a, b) => b ? Math.round(100 * a / b) + "%" : "-";
  const fig = (name, value, sub) => h("div", { class: "figure" },
    h("div", { class: "name" }, name), h("div", { class: "value" }, value),
    h("div", { class: "sub" }, sub));
  const us = t.points_us, them = t.points_them;
  return h("div", { class: "figures" },
    fig(nSets === 1 ? (us > them ? "Won" : us < them ? "Lost" : "Tied") : `Points, ${nSets} sets`,
      `${us}-${them}`, nSets === 1 ? "final score" : "us-them"),
    fig("Hitting", t.attacks ? effStr(t.hitting_eff) : "-",
      `${t.attacks || 0} attacks, ${t.kills || 0} kills, ${t.attack_errors || 0} errors`),
    fig("Passing", t.pass_n ? t.pass_avg.toFixed(2) : "-",
      `${t.pass_n || 0} serves received`),
    fig("Side-out", pct(t.side_out_won, t.side_out_chances),
      `${t.side_out_won || 0} of ${t.side_out_chances || 0} their serves`),
    fig("Point on our serve", pct(t.serve_points_won, t.serve_points_played),
      `${t.serve_points_won || 0} of ${t.serve_points_played || 0}`),
    fig("Aces", String(t.aces || 0),
      `${t.service_errors || 0} service errors on ${t.serves || 0} serves`));
}

// The score checks' warning, if any set fails them.
function scoreChecks(st) {
  const bad = (st.checks || []).filter(c => c.status !== "ok" &&
    c.status !== "last_point_unknown");
  return bad.length ? h("div", { class: "flags" },
    h("strong", {}, "These stats may be off. The score checks found:"),
    h("ul", {}, ...bad.map(c => h("li", {}, `${c.video}: ${c.message}` +
      (c.flagged ? ` ${c.flagged} rallies flagged for review.` : ""))))) : null;
}

function scopeTitle(fd, vids, chosen) {
  if (!chosen.length) return "Stats";
  const f = k => fd.folders[k] || {};
  const opp = k => (vids.find(v => v.key === k) || {}).opponent || f(k).game || k;
  const games = new Set(chosen.map(k => f(k).game));
  const tours = new Set(chosen.map(k => f(k).tournament));
  if (chosen.length === 1) {
    return `${opp(chosen[0])}, ${setName(chosen[0], f(chosen[0]).game || "").toLowerCase()}`;
  }
  if (games.size === 1) return opp(chosen[0]);
  if (tours.size === 1 && !tours.has(fd.unfiled)) return [...tours][0];
  return `${chosen.length} sets`;
}

function effStr(v) { return (v < 0 ? "-" : "") + Math.abs(v).toFixed(3); }

// One strip per set: every rally in order, our points in court blue, theirs in grey. A block
// opens that rally in review. Only the first few sets, to keep the page quick.
function rallyStrips(vids) {
  const sets = vids.filter(v => v.set_id).slice(0, 6);
  const box = h("section", {});
  if (!sets.length) return box;
  Promise.all(sets.map(v => api(`/sets/${v.set_id}/rallies`).then(l => [v, l]).catch(() => null)))
    .then(res => {
      const ok = res.filter(Boolean);
      if (!ok.length) return;
      const name = v => {
        const m = v.key.match(/_set(\d+)$/);
        return (v.opponent || v.key) + (m ? `, set ${m[1]}` : "");
      };
      box.replaceChildren(h("h2", {}, ok.length > 1 ? "How the sets went" : "How the set went"),
        ...ok.flatMap(([v, l]) => {
        const won = l.rallies.filter(r => r.point_winner === "us").length;
        const lost = l.rallies.filter(r => r.point_winner === "them").length;
        let run = 0, best = 0;
        for (const r of l.rallies) {
          run = r.point_winner === "us" ? run + 1 : r.point_winner === "them" ? 0 : run;
          best = Math.max(best, run);
        }
        return [
          sets.length > 1 ? h("div", { class: "small", style: "margin-top:12px" }, name(v),
            h("span", { class: "muted" }, `, ${won}-${lost}, longest run of ours ${best}`)) : null,
          h("div", { class: "strip", role: "list", "aria-label": `${v.key}: ${won} points to ${lost}` },
            ...l.rallies.map(r => h("a", { role: "listitem",
              class: r.point_winner === "us" ? "us" : r.point_winner === "them" ? "" : "none",
              href: `#/review/${v.set_id}/${r.id}?all=1`,
              title: `Rally ${r.rally_number}, ${r.score[0]}-${r.score[1]}: ` +
                (r.point_winner === "us" ? "our point" : r.point_winner === "them" ? "their point"
                  : label(r.outcome || "no winner")) }))),
          sets.length > 1 ? null : h("p", { class: "small muted" },
            `Our points in blue. Longest run of ours: ${best}. A block opens that rally.`),
        ].filter(Boolean);
      }),
        ok.length > 1 ? h("p", { class: "small muted", style: "margin-top:8px" },
          "Our points in blue. A block opens that rally.") : "");
    });
  return box;
}

// Which sets a page shows, as three linked dropdowns over the folders (data/folders.yaml):
// tournament, then game, then set, each with an "All" choice. One set, one game, one
// tournament or everything is at most two picks. onPick gets the new list of video keys;
// a list no single choice describes (an old link) shows as "Custom" until changed.
function scopePicker(fd, vids, chosen, onPick) {
  const have = new Set(vids.map(v => v.key));
  const tree = fd.tree.map(t => ({ ...t, games: t.games.map(g =>
    ({ ...g, videos: g.videos.filter(k => have.has(k)) })).filter(g => g.videos.length) }))
    .filter(t => t.games.length);
  const opp = new Map(vids.map(v => [v.key, v.opponent]));
  const gameName = g => opp.get(g.videos[0]) || g.game.replaceAll("-", " ");
  const tKeys = ti => tree[ti].games.flatMap(g => g.videos);
  const all = tree.flatMap((_, ti) => tKeys(ti));
  const same = (a, b) => a.length === b.length && a.every(k => b.includes(k));

  // Work out which choice the current list is.
  let T = "", G = "", S = "", custom = false;
  const findGame = test => { for (const [ti, t] of tree.entries()) for (const [gi, g] of t.games.entries())
    if (test(g)) return [ti, gi]; return null; };
  if (chosen.length === 1 && findGame(g => g.videos.includes(chosen[0]))) {
    const [ti, gi] = findGame(g => g.videos.includes(chosen[0]));
    T = String(ti); G = `${ti}/${gi}`; S = chosen[0];
  } else if (findGame(g => same(g.videos, chosen))) {
    const [ti, gi] = findGame(g => same(g.videos, chosen));
    T = String(ti); G = `${ti}/${gi}`;
  } else if (tree.some((_, ti) => same(tKeys(ti), chosen))) {
    T = String(tree.findIndex((_, ti) => same(tKeys(ti), chosen)));
  } else if (!same(all, chosen)) {
    custom = true;
  }
  const game = v => { const [ti, gi] = v.split("/").map(Number); return tree[ti].games[gi]; };

  const tSel = h("select", { onchange: e => {
    const v = e.target.value;
    if (v !== "custom") onPick(v === "" ? all : tKeys(+v)); } },
    custom ? h("option", { value: "custom", selected: true }, `Custom, ${chosen.length} sets`) : null,
    h("option", { value: "", selected: !custom && T === "" }, "All tournaments"),
    ...tree.map((t, ti) => h("option", { value: ti, selected: T === String(ti) }, t.tournament)));
  const gameOpts = ti => tree[ti].games.map((g, gi) =>
    h("option", { value: `${ti}/${gi}`, selected: G === `${ti}/${gi}` }, gameName(g)));
  const gSel = h("select", { onchange: e => {
    const v = e.target.value;
    onPick(v ? game(v).videos : T !== "" ? tKeys(+T) : all); } },
    h("option", { value: "", selected: G === "" }, "All games"),
    ...(T !== "" ? gameOpts(+T)
      : tree.map((t, ti) => h("optgroup", { label: t.tournament }, ...gameOpts(ti)))));
  const sSel = h("select", { disabled: G === "", onchange: e => {
    onPick(e.target.value ? [e.target.value] : game(G).videos); } },
    h("option", { value: "", selected: S === "" }, "All sets"),
    ...(G ? game(G).videos.map(k => h("option", { value: k, selected: S === k },
      setName(k, game(G).game))) : []));
  return [
    h("label", { class: "fl" }, "Tournament", tSel),
    h("label", { class: "fl" }, "Game", gSel),
    h("label", { class: "fl" }, "Set", sSel)];
}

// "pakmen-black_set2" in the game "pakmen-black" -> "Set 2".
function setName(key, game) {
  const rest = key.startsWith(game + "_") ? key.slice(game.length + 1) : key;
  const m = rest.match(/^set(\d+)$/);
  return m ? `Set ${m[1]}` : rest.replaceAll("_", " ");
}

// Box score tab: the table, one line on what it counts, the column key.
function statsBody(st, src) {
  const t = st.team;
  const cols = st.columns;
  const head = h("tr", {}, h("th", {}, "Player"), ...cols.map(c => h("th", { title: label(c) },
    st.short[c])));
  const rows = [...st.players, { player: "TEAM", ...t, team: true }].map(r =>
    h("tr", { class: r.team ? "team" : "" },
      h("td", {}, r.team ? "Team" : (r.player === "?" ? "Player not known" : "#" + r.player) +
        (r.name && r.player !== "?" ? " " + r.name : "")),
      ...cols.map(c => h("td", {}, r[c] === null || r[c] === undefined ? "-"
        : c === "hitting_eff" ? effStr(r[c]) : c === "pass_avg" ? r[c].toFixed(2) : fmt(r[c])))));
  const note = src === "pipeline"
    ? `${st.counted} rallies, ${st.reviewed} reviewed by you. The rest use the pipeline's guesses.`
    : `${st.counted} hand-labelled rallies.`;
  return [h("p", { class: "muted" }, note),
    h("div", { class: "scroll" }, h("table", { class: "big" }, h("thead", {}, head),
      h("tbody", {}, rows))),
    h("p", { class: "small muted", style: "margin-top:12px" },
      cols.map(c => `${st.short[c]} ${label(c)}`).join(", ") + ".")];
}

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
      `Rally ${x.rally_number}, ${clock(x.start_t)}, ${x.score[0]}-${x.score[1]}, ` +
      `sure ${(x.priority || 0).toFixed(2)}` + (x.reviewed ? ", reviewed" : "") +
      (x.flags.length ? ", flagged" : ""));
  }));
  const reviewed = list.rallies.filter(x => x.reviewed).length;
  const score = (byId[rallyId] && byId[rallyId].score) || d.score || [0, 0];
  // Every rally of the set in order, as wide as it is long: this one in yellow, flagged
  // ones in red, reviewed ones lighter. A block opens that rally.
  const timeline = h("div", { class: "timeline", "aria-label": "Rallies in this set" },
    ...list.rallies.map(x => h("a", {
      class: x.id === rallyId ? "current" : x.flags.length ? "flagged" : x.reviewed ? "reviewed" : "",
      style: `flex:${Math.max(1, x.end_t - x.start_t).toFixed(1)}`,
      href: `#/review/${setId}/${x.id}${suffix}`,
      title: `Rally ${x.rally_number}, ${x.score[0]}-${x.score[1]}` +
        (x.reviewed ? ", reviewed" : "") + (x.flags.length ? ", flagged" : "") })));
  const top = h("div", {},
    h("div", { class: "review-top" },
      h("div", { class: "title" },
        h("h1", {}, `${r.video_key}, rally ${r.rally_number}`),
        h("div", { class: "small muted" }, `${reviewed} of ${list.rallies.length} reviewed, ` +
          `${clock(r.start_t)} to ${clock(r.end_t)} (${(r.end_t - r.start_t).toFixed(1)} s). ` +
          `Score from reviewed rallies ${list.score_reviewed[0]}-${list.score_reviewed[1]}.`)),
      h("div", { class: "score", "aria-label": `Score ${score[0]}-${score[1]}` },
        `${score[0]}-${score[1]}`),
      seg([["queue", "Review queue"], ["flagged", `Flagged (${list.flagged.length})`],
        ["all", "All in order"]], mode, v => {
        location.hash = `#/review/${setId}` + (v === "all" ? "?all=1" :
          v === "flagged" ? "?mode=flagged" : ""); })),
    timeline,
    h("div", { class: "small " + (list.checks.status === "ok" ||
      list.checks.status === "last_point_unknown" ? "muted" : "warntext") },
      `Set ${list.checks.set_number} (${list.checks.format.label}): ${list.checks.message}`),
    h("div", { style: "margin:8px 0 16px; max-width:640px" }, picker));
  const flagsBox = d.flags.length ? h("div", { class: "flags" },
    h("strong", {}, d.flags.length === 1 ? d.flags[0] : "The score checks flag this rally:"),
    d.flags.length > 1 ? h("ul", {}, ...d.flags.map(f => h("li", {}, f))) : null) : null;

  // player
  const video = h("video", { playsinline: true, preload: "auto", controls: false });
  const readout = h("div", { class: "readout" });
  const bar = h("input", { type: "range", min: 0, max: 1000, value: 0 });
  const now = () => t0() + video.currentTime;
  const show = () => {
    readout.textContent = `Clip ${video.currentTime.toFixed(3)} s, video ${clock(now())}, ` +
      `frame ${Math.round(now() * fps)}`;
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
    slowBtn.classList.toggle("on", video.playbackRate !== 1); } }, "Quarter speed");
  const widen = (key, label) => h("label", { class: "row small" }, label,
    h("input", { type: "number", min: 0, max: 60, step: 1, value: S[key], onchange: e => {
      S[key] = Math.max(0, +e.target.value || 0); loadClip(); drawTouches(); drawSpan(); } }));
  const player = h("div", {},
    d.has_video ? video : h("p", { class: "card muted" },
      "The video isn't on this machine: review the touches without the clip, or open " +
      "the app on the desktop."),
    readout, bar,
    h("div", { class: "row" },
      h("button", { onclick: () => (video.paused ? video.play() : video.pause()) },
        "Play or pause"),
      h("button", { onclick: () => seek(now() - 0.1) }, "Back 0.1 s"),
      h("button", { onclick: () => frame(-1) }, "Back a frame"),
      h("button", { onclick: () => frame(1) }, "Forward a frame"),
      h("button", { onclick: () => seek(now() + 0.1) }, "Forward 0.1 s"), slowBtn),
    h("div", { class: "row", style: "margin-top:8px" },
      widen("before", "Seconds before"), widen("after", "Seconds after")),
    h("p", { class: "small muted", style: "margin-top:8px" }, "Keys: comma and full stop " +
      "step a frame, space plays. Up and down (or k and j) pick a touch and jump to it. On " +
      "that touch: s serve, r receive, e set, a attack, b block, d dig, p dump, f free ball, " +
      "o overpass, c coverage, x switches team, 0 to 4 pass score, t moves it to now, " +
      "Delete removes it. n adds a touch now. w we won, l they won, R replay, N not a rally. " +
      "Enter saves and goes to the next, Shift+Enter saves. Touch times are seconds into " +
      "the clip."),
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
    const box = h("div", { class: "card", style: "margin-top:12px" });
    const gapNote = d.prev && d.prev.gap_s > META.missed_point_gap_s
      ? h("div", { class: "small bad" }, `${d.prev.gap_s} s since the ` +
        "previous rally: points rarely take that long to start, so one may be missing here.")
      : null;
    const draw = () => box.replaceChildren(
      h("strong", { class: "panel-title" }, "Missed point in this clip?"),
      gapNote || "",                      // replaceChildren would print a null as "null"
      h("div", { class: "small muted" }, "Widen the clip (s before / after) to see it, then " +
        "mark where it starts (the serve) and ends."),
      h("div", { class: "row", style: "margin-top:6px" },
        h("button", { onclick: () => { M.start = now(); draw(); } },
          "Starts here" + (M.start !== null ? ` (${clock(M.start)})` : "")),
        h("button", { onclick: () => { M.end = now(); draw(); } },
          "Ends here" + (M.end !== null ? ` (${clock(M.end)})` : "")),
        seg([["us", "We serve"], ["them", "They serve"]], M.serving, v => { M.serving = v; draw(); }),
        h("button", { disabled: M.start === null || M.end === null,
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
        h("button", { class: "iconbtn", title: "Go to this touch", onclick: () => {
          S.sel = x; seek(x.t); drawTouches(); } }, "Go"),
        h("input", { type: "number", step: 0.001, value: (x.t - t0()).toFixed(3),
          onchange: e => { x.t = t0() + (+e.target.value); suggest(); } }),
        select(["us", "them"], x.team, v => { x.team = v; drawTouches(); suggest(); }),
        select(META.actions, x.action, v => { x.action = v; drawTouches(); suggest(); }, "action?"),
        h("button", { class: "iconbtn", title: "Remove this touch", onclick: () => {
          x.delete = true; drawTouches(); suggest(); } }, "Remove"));
      if (x.action_confidence !== undefined && x.action_confidence !== null && x.id) {
        row.append(h("div", { class: "sure" }, `pipeline sure ${x.action_confidence.toFixed(2)}`));
      }
      if (x.team === "us") row.append(extraRow(x));
      return row;
    });
    touchBox.replaceChildren(...rows,
      h("div", { class: "row", style: "margin-top:8px" },
        h("button", { onclick: addTouch }, "Add a touch at this frame")));
  }

  function addTouch() {
    const x = { id: null, t: now(), team: "us", action: null, delete: false };
    S.touches.push(x); S.sel = x; drawTouches(); suggest();
  }

  // Who touched it (our touches) and, on our receive, the pass score 0-4.
  // A pipeline guess shows how sure it was and a Right button to confirm it; setting a value
  // confirms it too. Only confirmed ones count as labels (export_reviewed.py).
  function extraRow(x) {
    const guess = (v, c, checked, confirm, from = "pipeline guess") =>
      v !== null && v !== undefined && !checked && c !== null && c !== undefined && c < 1
        ? h("span", { class: "row small muted" }, `${from}, sure ${c.toFixed(2)}`,
          h("button", { class: "iconbtn", title: "The guess is right", onclick: confirm }, "Right"))
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
      h("button", { onclick: () => { S.start = now(); drawSpan(); } }, "Set to now")),
    h("div", { class: "row", style: "margin-top:6px" },
      h("span", { class: "small" }, "Ended (touches after it go)"), spanInput("end"),
      h("button", { onclick: () => { S.end = now(); drawSpan(); suggest(); } }, "Set to now")));
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
      h("div", { class: "small", style: "margin-bottom:4px" }, "Who won the point?"), seg(winners, S.winner, v => {
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
      flagsBox,
      h("div", { class: "card" }, h("span", { class: "panel-title" }, "Touches"), touchBox),
      h("div", { class: "card" }, h("span", { class: "panel-title" }, "Rally"), spanBox),
      h("div", { class: "card" }, h("span", { class: "panel-title" }, "Point"), pointBox),
      h("div", { class: "row" },
        h("button", { onclick: () => save(false) }, "Save"),
        h("button", { class: "primary", onclick: () => save(true) }, "Save and next")))));
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
