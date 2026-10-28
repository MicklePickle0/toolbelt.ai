// Setter Eyes web app: videos + processing queue, rally review, stat sheet.
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
async function api(path, opts = {}) {
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
    wrap.append(h("button", { class: value === current ? "on" : "", type: "button",
      onclick: () => onPick(value) }, text));
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
    a.classList.toggle("on", a.dataset.nav === (page === "review" ? "videos" : page)));
  try {
    if (page === "queue") await queuePage();
    else if (page === "stats") await statsPage(params);
    else if (page === "review") await reviewPage(+parts[1], parts[2] ? +parts[2] : null, params);
    else await videosPage();
  } catch (e) {
    view.replaceChildren(errorBox(e));
  }
}

function errorBox(e) {
  return h("div", { class: "card" }, h("strong", {}, "Something went wrong: "), e.message);
}

window.addEventListener("hashchange", route);
route();

// ---- videos -------------------------------------------------------------------------------
async function videosPage() {
  const draw = async () => {
    const vids = await api("/videos");
    const list = vids.map(videoCard);
    view.replaceChildren(
      h("h1", {}, META.team + ": videos"),
      h("p", { class: "muted small" }, "Process a video on this PC's GPU, then review its " +
        "rallies and read the stats. Least confident rallies come first in review."),
      ...list);
  };
  await draw();
  timer = setInterval(() => draw().catch(() => {}), 5000);
}

function videoCard(v) {
  const cal = v.calibration;
  const calChip = !cal ? h("span", { class: "chip bad" }, "not calibrated")
    : h("span", { class: "chip " + (cal.method === "manual" ? "good" : "warn") },
        `calibration: ${cal.method} ${cal.confidence.toFixed(2)}` +
        (cal.method === "manual" ? "" : " (check)"));
  const steps = Object.entries(v.cache).map(([k, ok]) =>
    h("span", { class: "chip " + (ok ? "good" : "") }, (ok ? "✓ " : "· ") + k));
  const job = v.job;
  const active = job && (job.status === "queued" || job.status === "running");
  const jobLine = job ? h("div", { class: "small muted" },
    `job ${job.id}: ${label(job.status)}` + (job.step_name ? `, ${job.step_name}` : "") +
    (job.status === "running" && job.progress ? ` · ${job.progress}` : "")) : null;
  const buttons = h("div", { class: "row" },
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
      calChip, ...steps),
    jobLine);
}

// ---- queue --------------------------------------------------------------------------------
async function queuePage() {
  let openLog = null;
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
        h("button", { onclick: () => { openLog = openLog === j.id ? null : j.id; draw(); } },
          openLog === j.id ? "Hide log" : "Log"));
      const card = h("div", { class: "card" },
        h("div", { class: "row spread" },
          h("div", {}, h("strong", {}, `#${j.id} ${j.video}`), " ",
            h("span", { class: "chip " + chipClass }, label(j.status))),
          actions),
        h("div", { class: "small" }, `step ${Math.min(done + 1, n)}/${n}: ` +
          (j.step_name || "finished") +
          (j.status === "running" ? ` (${ago(j.step_started)})` : "")),
        j.progress ? h("div", { class: "small mono muted" }, j.progress) : null,
        j.error ? h("div", { class: "small", style: "color:var(--bad)" }, j.error) : null,
        h("div", { class: "small muted" }, "queued " + ago(j.created) + " ago" +
          (j.started ? ", started " + ago(j.started) + " ago" : "")));
      if (openLog === j.id) {
        const pre = h("pre", { class: "log" }, "loading…");
        api(`/jobs/${j.id}/log`).then(t => { pre.textContent = t || "(empty)";
          pre.scrollTop = pre.scrollHeight; });
        card.append(pre);
      }
      return card;
    });
    view.replaceChildren(h("h1", {}, "Processing queue"),
      h("p", { class: "muted small" }, "One video at a time on this PC's GPU. Steps: " +
        META.steps.join(" → ") + ". Detection resumes where it stopped."),
      ...(cards.length ? cards : [h("p", { class: "muted" }, "Nothing queued yet.")]));
  };
  await draw();
  timer = setInterval(() => { if (openLog === null) draw().catch(() => {}); }, 3000);
}

// ---- stats --------------------------------------------------------------------------------
async function statsPage(params) {
  const vids = (await api("/videos")).filter(v => v.rallies);
  let chosen = (params.get("v") || "").split(",").filter(Boolean);
  if (!chosen.length && vids.length) chosen = [vids[0].key];
  const src = params.get("src") || "pipeline";
  const go = (keys, s) => { location.hash = `#/stats?v=${keys.join(",")}&src=${s}`; };

  const picker = h("div", { class: "card" },
    h("div", { class: "row spread" }, h("strong", {}, "Sets"),
      h("div", { class: "row" },
        h("button", { onclick: () => go(vids.map(v => v.key), src) }, "All"),
        seg([["pipeline", "Pipeline + review"], ["labels", "Hand labels"]], src,
          s => go(chosen, s)))),
    h("div", { class: "row", style: "margin-top:8px" }, ...vids.map(v =>
      h("label", { class: "check chip" },
        h("input", { type: "checkbox", checked: chosen.includes(v.key), onchange: e => {
          const next = e.target.checked ? [...chosen, v.key] : chosen.filter(k => k !== v.key);
          go(next, src); } }), v.key))));

  let body;
  try {
    const st = await api(`/stats?videos=${encodeURIComponent(chosen.join(","))}&source=${src}`);
    body = statsBody(st, src);
  } catch (e) {
    body = [h("p", { class: "muted" }, e.message)];
  }
  view.replaceChildren(h("h1", {}, "Stat sheet"), picker, ...body);
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
    tile(fmt(t.pass_avg), `pass average (${t.pass_n || 0} scored passes)`));
  const cols = st.columns;
  const head = h("tr", {}, h("th", {}, "Player"), ...cols.map(c => h("th", { title: label(c) },
    st.short[c])));
  const rows = [{ player: "TEAM", ...t, team: true }, ...st.players].map(r =>
    h("tr", { class: r.team ? "team" : "" },
      h("td", {}, r.team ? "TEAM" : (r.player === "?" ? "#?" : "#" + r.player) +
        (r.name && r.player !== "?" ? " " + r.name : "")),
      ...cols.map(c => h("td", {}, fmt(r[c])))));
  const note = src === "pipeline"
    ? `${st.counted} rallies counted; ${st.reviewed} of ${st.rallies} reviewed by you. ` +
      "Unreviewed rallies use the pipeline's guesses (winner from who serves next)."
    : `${st.counted} hand-labelled rallies.`;
  return [tiles, h("p", { class: "small muted" }, note),
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
  const showAll = params.get("all") === "1";
  const list = await api(`/sets/${setId}/rallies`);
  const byId = Object.fromEntries(list.rallies.map(r => [r.id, r]));
  const order = showAll ? list.rallies.map(r => r.id) : list.queue;
  if (!rallyId) {
    if (!order.length) {
      view.replaceChildren(h("h1", {}, "Review"),
        h("p", {}, "Every rally of this set is reviewed. "),
        h("button", { onclick: () => { location.hash = `#/review/${setId}?all=1`; } },
          "Show all rallies"));
      return;
    }
    location.replace(`#/review/${setId}/${order[0]}` + (showAll ? "?all=1" : ""));
    return;
  }
  const d = await api(`/rallies/${rallyId}`);
  const r = d.rally, fps = r.fps;
  const pad = META.clip_pad_s;
  const S = {                                 // this page's editable state
    before: pad, after: pad,
    start: r.start_t, end: r.end_t,
    touches: d.touches.map(x => ({ ...x, delete: false })),
    winner: r.point_winner || (["replay", "not_a_rally"].includes(r.outcome) ? r.outcome : null),
    outcome: r.reviewed ? r.outcome : null,
    serving: r.serving_team, checked: true, suggested: null,
  };
  const t0 = () => Math.max(0, r.start_t - S.before);
  const suffix = showAll ? "?all=1" : "";
  const pos = order.indexOf(rallyId);
  const nextId = pos >= 0 && pos + 1 < order.length ? order[pos + 1] : null;

  // header: picker and progress
  const picker = h("select", { class: "picker", onchange: e => {
    location.hash = `#/review/${setId}/${e.target.value}${suffix}`; } },
  ...(order.includes(rallyId) ? order : [rallyId, ...order]).map(id => {
    const x = byId[id];
    return h("option", { value: id, selected: id === rallyId },
      `${x.reviewed ? "✓ " : ""}#${x.rally_number}  ${clock(x.start_t)}  conf ` +
      `${(x.segment_confidence || 0).toFixed(2)}`);
  }));
  const reviewed = list.rallies.filter(x => x.reviewed).length;
  const top = h("div", { class: "card" },
    h("div", { class: "row spread" },
      h("strong", {}, `${r.video_key}: rally ${r.rally_number}`),
      seg([["0", "Review queue"], ["1", "All in order"]], showAll ? "1" : "0", v => {
        location.hash = `#/review/${setId}` + (v === "1" ? "?all=1" : ""); })),
    h("div", { class: "small muted" }, `${reviewed}/${list.rallies.length} reviewed · ` +
      `score from reviewed rallies ${list.score_reviewed[0]}-${list.score_reviewed[1]} · ` +
      `${clock(r.start_t)}-${clock(r.end_t)} (${(r.end_t - r.start_t).toFixed(1)} s)`),
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
    h("p", { class: "small muted" }, "Keys: , and . step a frame, space plays. Times below " +
      "are seconds into the clip (the \"clip\" time above)."),
    mergeButtons());

  function mergeButtons() {
    const b = (x, what) => x && h("button", { onclick: async () => {
      if (!confirm(`Merge this rally with the ${what} one (#${x.rally_number})?`)) return;
      await api(`/rallies/${rallyId}/merge`, { body: { other_id: x.id } });
      toast("Merged"); route(); } },
    `Merge with ${what} #${x.rally_number} (${x.gap_s} s apart)`);
    return h("div", { class: "row" }, b(d.prev, "previous"), b(d.next, "next"));
  }

  // touches
  const touchBox = h("div");
  function drawTouches() {
    const rows = S.touches.filter(x => !x.delete).sort((a, b) => a.t - b.t).map(x => {
      const row = h("div", { class: "touch " + (x.team || "") },
        h("button", { class: "iconbtn", title: "go to this touch", onclick: () => seek(x.t) }, "▶"),
        h("input", { type: "number", step: 0.001, value: (x.t - t0()).toFixed(3),
          onchange: e => { x.t = t0() + (+e.target.value); suggest(); } }),
        select(["us", "them"], x.team, v => { x.team = v; drawTouches(); suggest(); }),
        select(META.actions, x.action, v => { x.action = v; suggest(); }, "action?"),
        h("button", { class: "iconbtn", title: "delete", onclick: () => {
          x.delete = true; drawTouches(); suggest(); } }, "✕"));
      if (x.action_confidence !== undefined && x.action_confidence !== null && x.id) {
        row.append(h("div", { class: "sure" }, `pipeline sure ${x.action_confidence.toFixed(2)}`));
      }
      return row;
    });
    touchBox.replaceChildren(...rows,
      h("div", { class: "row", style: "margin-top:8px" },
        h("button", { onclick: () => {
          S.touches.push({ id: null, t: now(), team: "us", action: null, delete: false });
          drawTouches(); suggest(); } }, "+ Touch at the current frame")));
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
        touches: S.touches, touches_checked: S.checked } });
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

  document.onkeydown = e => {
    if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return;
    if (e.key === ",") frame(-1);
    else if (e.key === ".") frame(1);
    else if (e.key === " ") { e.preventDefault(); video.paused ? video.play() : video.pause(); }
  };
}
