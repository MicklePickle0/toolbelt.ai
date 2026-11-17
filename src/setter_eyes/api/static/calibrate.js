// Calibration page (#/calibrate/<video>[?f=<frame>]): tap the court points on one frame, check
// the court drawn back over it, save. Same solve as scripts/calibrate.py's click UI
// (calibration/solve.py). Uses api(), h(), toast(), clock() and `view` from app.js.
"use strict";

const POINT_HINTS = {
  near_left_corner: "near end line meets the left sideline (nearest the camera, left)",
  near_right_corner: "near end line meets the right sideline",
  near_left_attack: "our attack line meets the left sideline",
  near_right_attack: "our attack line meets the right sideline",
  centre_left: "centre line (under the net) meets the left sideline",
  centre_right: "centre line (under the net) meets the right sideline",
  far_left_attack: "their attack line meets the left sideline",
  far_right_attack: "their attack line meets the right sideline",
  far_left_corner: "far end line meets the left sideline",
  far_right_corner: "far end line meets the right sideline",
  net_tape_left: "top of the net tape at the left antenna",
  net_tape_right: "top of the net tape at the right antenna",
};
const LINE_COLOURS = ["#facc15", "#22d3ee", "#a78bfa", "#f472b6", "#4ade80", "#fb923c",
  "#60a5fa", "#f87171"];

async function calibratePage(key, params) {
  const enc = encodeURIComponent(key);
  const info = await api(`/videos/${enc}/calibration`);
  const cur = info.current;
  const firstFrame = () => {
    if (cur && cur.frame !== null && cur.frame !== undefined) return cur.frame;
    const want = Math.round(30 * info.fps);
    if (!info.cached_frames.length) return want;
    return info.cached_frames.reduce((a, b) => (Math.abs(b - want) < Math.abs(a - want) ? b : a));
  };
  const S = {
    frame: params.get("f") ? +params.get("f") : firstFrame(),
    clicks: {}, lines: {}, phase: "points", i: 0, j: 0, last: null, drag: null,
    skipped: new Set(), solved: null,
  };
  if (cur && cur.method === "manual" && cur.frame === S.frame) {   // edit the saved points
    S.clicks = structuredClone(cur.clicks || {});
    S.lines = structuredClone(cur.line_clicks || {});
  }
  const K = info.keypoints, L = info.lines;
  const pretty = n => n.replaceAll("_", " ");
  const nPoints = () => Object.keys(S.clicks).length;
  const nLine = () => Object.values(S.lines).reduce((a, v) => a + v.length, 0);
  const nextOpen = i => { while (i < K.length && (K[i] in S.clicks)) i++; return i; };

  const img = new Image();
  const canvas = h("canvas", { style: "width:100%;display:block;touch-action:none;" +
    "border-radius:6px;background:#000;cursor:crosshair" });
  const loupe = h("canvas", { width: 180, height: 180, style: "position:absolute;top:8px;" +
    "right:8px;width:180px;height:180px;outline:1px solid #E8EAED;" +
    "display:none;pointer-events:none" });
  const stage = h("div", { style: "position:relative" }, canvas, loupe);
  const panel = h("div");
  const status = h("div", { class: "small muted" });
  const scale = () => info.width / img.naturalWidth;        // shown px -> full-res px

  function toFull(e) {
    const r = canvas.getBoundingClientRect();
    return [(e.clientX - r.left) * canvas.width / r.width * scale(),
      (e.clientY - r.top) * canvas.height / r.height * scale()];
  }

  // ---- drawing ----
  function draw() {
    if (!img.complete || !img.naturalWidth) return;
    const g = canvas.getContext("2d");
    g.drawImage(img, 0, 0);
    const k = 1 / scale(), r = Math.max(4, canvas.width / 320);
    g.font = `${Math.round(r * 3)}px system-ui`;
    g.lineWidth = Math.max(2, r / 2);
    for (const [n, [u, v]] of Object.entries(S.clicks)) {
      g.fillStyle = S.phase === "points" && K[S.i] === n ? "#facc15" : "#fb923c";
      g.beginPath(); g.arc(u * k, v * k, r, 0, 7); g.fill();
      g.strokeStyle = "#000"; g.stroke();
      g.fillStyle = "#fff"; g.fillText(pretty(n), u * k + r * 1.5, v * k - r * 1.5);
    }
    L.forEach((n, idx) => {
      g.strokeStyle = LINE_COLOURS[idx % LINE_COLOURS.length];
      for (const [u, v] of S.lines[n] || []) {
        const x = u * k, y = v * k;
        g.beginPath(); g.moveTo(x - r, y - r); g.lineTo(x + r, y + r);
        g.moveTo(x + r, y - r); g.lineTo(x - r, y + r); g.stroke();
      }
    });
    if (S.drag) {
      const [u, v] = S.drag;
      g.strokeStyle = "#fff"; g.lineWidth = 1;
      g.beginPath(); g.arc(u * k, v * k, r * 2, 0, 7); g.stroke();
      drawLoupe(u * k, v * k);
    }
  }

  function drawLoupe(x, y) {
    const g = loupe.getContext("2d"), zoom = 5, half = 90 / zoom;
    g.imageSmoothingEnabled = false;
    g.fillStyle = "#000"; g.fillRect(0, 0, 180, 180);
    g.drawImage(img, x - half, y - half, 2 * half, 2 * half, 0, 0, 180, 180);
    g.strokeStyle = "#ef4444"; g.lineWidth = 1;
    g.beginPath(); g.moveTo(90, 0); g.lineTo(90, 180); g.moveTo(0, 90); g.lineTo(180, 90);
    g.stroke();
  }

  // ---- input: press, drag with the magnifier, let go to place ----
  canvas.addEventListener("pointerdown", e => {
    if (S.phase === "check") return;
    canvas.setPointerCapture(e.pointerId);
    S.drag = toFull(e); loupe.style.display = "block"; draw();
  });
  canvas.addEventListener("pointermove", e => { if (S.drag) { S.drag = toFull(e); draw(); } });
  canvas.addEventListener("pointerup", () => {
    if (!S.drag) return;
    const p = S.drag.map(x => Math.round(x * 10) / 10);
    S.drag = null; loupe.style.display = "none";
    place(p);
  });
  canvas.addEventListener("pointercancel", () => {
    S.drag = null; loupe.style.display = "none"; draw(); });

  function place(p) {
    if (S.phase === "points" && S.i < K.length) {
      const n = K[S.i];
      S.clicks[n] = p; S.skipped.delete(n); S.last = ["point", n];
      S.i = nextOpen(S.i + 1);
    } else if (S.phase === "lines" && S.j < L.length) {
      const n = L[S.j];
      (S.lines[n] = S.lines[n] || []).push(p); S.last = ["line", n];
    }
    redraw();
  }

  function nudge(dx, dy) {                    // one pixel of the frame shown here
    if (!S.last) return;
    const [kind, n] = S.last;
    const p = kind === "point" ? S.clicks[n] : (S.lines[n] || []).at(-1);
    if (!p) return;
    p[0] += dx * scale(); p[1] += dy * scale(); redraw();
  }

  function undo() {
    if (S.phase === "points") {
      if (S.last && S.last[0] === "point" && S.last[1] in S.clicks) {
        delete S.clicks[S.last[1]]; S.i = K.indexOf(S.last[1]); S.last = null;
      } else if (S.i > 0) {
        S.i -= 1; delete S.clicks[K[S.i]]; S.skipped.delete(K[S.i]);
      }
    } else if (S.phase === "lines") {
      const pts = S.lines[L[S.j]];
      if (pts && pts.length) pts.pop();
      else if (S.j > 0) S.j -= 1;
      else S.phase = "points";
    }
    redraw();
  }

  // ---- the step panel ----
  function redraw() { draw(); drawPanel(); }

  function drawPanel() {
    const nudges = h("div", { class: "row", style: "margin-top:8px" },
      h("span", { class: "small" }, "Nudge the last point (or the arrow keys):"),
      ...[["←", "left", -1, 0], ["↑", "up", 0, -1], ["↓", "down", 0, 1], ["→", "right", 1, 0]]
        .map(([t, name, dx, dy]) => h("button", { class: "iconbtn", "aria-label": "Nudge " + name,
          title: "Nudge " + name, onclick: () => nudge(dx, dy) }, t)));
    const counts = h("div", { class: "small muted", style: "margin-top:6px" },
      `${nPoints()} court points, ${nLine()} line points. Needs 4 court points and 6 in ` +
      "total; more points spread over the court give a better fit.");
    if (S.phase === "points") {
      const n = K[S.i];
      const chips = h("div", { class: "row", style: "margin-top:8px" }, ...K.map((k, idx) =>
        h("button", { class: idx === S.i ? "on" : "", style: "min-height:32px;padding:4px 10px",
          onclick: () => { S.i = idx; redraw(); } },
        pretty(k) + (k in S.clicks ? ", done" : S.skipped.has(k) ? ", skipped" : ""))));
      panel.replaceChildren(
        n ? h("div", {}, h("strong", {}, `Point ${S.i + 1}/${K.length}: ${pretty(n)}`),
          h("div", { class: "small" }, POINT_HINTS[n] || ""),
          h("div", { class: "small muted" }, "Press and drag on the frame: the magnifier " +
            "shows where the point lands; let go to place it. Left and right are as seen " +
            "from the camera."))
          : h("strong", {}, "Court points done"),
        h("div", { class: "row", style: "margin-top:8px" },
          n ? h("button", { onclick: () => { S.skipped.add(n); S.i = nextOpen(S.i + 1);
            redraw(); } }, "Skip (can't see it)") : null,
          h("button", { onclick: undo }, "Undo"),
          h("button", { class: "primary", onclick: () => { S.phase = "lines"; S.j = 0;
            redraw(); } }, "Next: line points")),
        nudges, counts, chips);
    } else if (S.phase === "lines") {
      const n = L[S.j];
      panel.replaceChildren(
        n ? h("div", {},
          h("strong", { style: `color:${LINE_COLOURS[S.j % LINE_COLOURS.length]}` },
            `Line ${S.j + 1}/${L.length}: ${pretty(n)}`),
          h("div", { class: "small" }, "Tap a few points along the " +
            (info.line_hints[n] || "middle of the painted line") +
            ` (${(S.lines[n] || []).length} so far). Helps where a corner is hidden; ` +
            "skip lines you can't see."))
          : h("strong", {}, "Lines done"),
        h("div", { class: "row", style: "margin-top:8px" },
          n ? h("button", { onclick: () => { S.j += 1; redraw(); } }, "Next line") : null,
          h("button", { onclick: undo }, "Undo"),
          h("button", { onclick: () => { S.phase = "points"; S.i = nextOpen(0); redraw(); } },
            "Back to court points"),
          h("button", { class: "primary", onclick: check }, "Check the fit")),
        nudges, counts);
    } else {
      const r = S.solved;
      panel.replaceChildren(
        h("strong", {}, r.ok ? "Check that the drawn lines sit on the painted lines:"
          : "The fit is loose: look closely, or adjust the points"),
        h("div", { class: "small" }, `court points off by ${r.rms} px, line points by ` +
          `${r.line_rms} px (good: under ${r.limit_px} px at full size)`),
        h("img", { src: r.overlay, alt: "The court drawn back over the frame",
          style: "width:100%;border-radius:6px;margin-top:8px" }),
        h("div", { class: "row", style: "margin-top:8px" },
          h("button", { onclick: () => { S.phase = "points"; S.i = nextOpen(0); redraw(); } },
            "Adjust points"),
          h("button", { class: "primary", onclick: save }, "Save calibration")));
    }
  }

  // ---- solve and save ----
  const body = () => ({ frame: S.frame, clicks: S.clicks, line_clicks: S.lines });

  async function check() {
    status.textContent = "Solving";
    try {
      S.solved = await api(`/videos/${enc}/calibration/solve`, { body: body() });
      S.phase = "check"; status.textContent = ""; redraw();
      panel.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (e) { status.textContent = ""; toast(e.message, 6000); }
  }

  async function save() {
    let res;
    try { res = await api(`/videos/${enc}/calibration`, { body: body() }); }
    catch (e) { toast(e.message, 6000); return; }
    const v = (await api("/videos")).find(x => x.key === key);
    const job = v && v.job;
    const go = async (path, payload) => {
      try { await api(path, { body: payload }); location.hash = "#/queue"; }
      catch (e) { toast(e.message, 5000); }
    };
    let next = null;
    if (job && job.status === "needs_calibration") {
      next = h("button", { class: "primary", onclick: () => go(`/jobs/${job.id}/resume`, {}) },
        `Resume processing (job ${job.id})`);
    } else if (res.rerun_from && !(job && ["queued", "running"].includes(job.status))) {
      next = h("button", { class: "primary", onclick: () =>
        go("/jobs", { video: key, from_step: res.rerun_from }) },
      "Re-run processing with the new calibration");
    } else if (!job || ["done", "cancelled", "failed", "interrupted"].includes(job.status)) {
      next = h("button", { class: "primary", onclick: () => go("/jobs", { video: key }) },
        "Process this video");
    }
    view.replaceChildren(h("h1", {}, `${key}: calibrated`),
      h("div", { class: "card" },
        h("p", {}, res.needs_review ? `Saved, but the fit is loose (${res.rms_px} px), so ` +
          "it is marked for checking." : `Saved (fit within ${res.rms_px} px).`),
        h("div", { class: "row" }, next,
          h("button", { onclick: () => { location.hash = "#/"; } }, "Back to videos"))));
  }

  // ---- frame chooser ----
  const tInput = h("input", { type: "number", min: 0, max: Math.floor(info.duration || 0),
    step: 1, value: Math.round(S.frame / info.fps) });
  const loadAt = sec => {
    const f = Math.max(0, Math.round(sec * info.fps));
    if ((nPoints() || nLine()) && !confirm("A new frame clears the points tapped so far. Go on?")) {
      return;
    }
    location.hash = `#/calibrate/${enc}?f=${f}`;
  };
  const chooser = h("div", { class: "card" },
    h("div", { class: "row spread" }, h("h1", {}, `Calibrate ${key}`),
      h("span", { class: "small muted" }, `Net ${info.net_height_m} m, ${info.width} by ` +
        `${info.height}, frame ${S.frame} (${clock(S.frame / info.fps)})`)),
    h("p", { class: "small muted" }, "Pick a frame where the court lines are clear (few " +
      "players standing on them), then tap the court points in order. Each video needs " +
      "this once; the camera is followed from this frame."),
    h("div", { class: "row" },
      h("button", { onclick: () => loadAt(S.frame / info.fps - 10) }, "Back 10 s"),
      h("label", { class: "row small" }, "Seconds", tInput),
      h("button", { onclick: () => loadAt(+tInput.value) }, "Load frame"),
      h("button", { onclick: () => loadAt(S.frame / info.fps + 10) }, "Forward 10 s")),
    cur ? h("div", { class: "small muted", style: "margin-top:6px" },
      `Current calibration: ${cur.method}` +
      (cur.usable ? "" : " (an automatic fit, not trusted: tap the court)") +
      (cur.frame !== null ? `, frame ${cur.frame}` : "") +
      (cur.method === "manual" && cur.frame !== S.frame ? " (its points are on that frame)" : ""))
      : null,
    status);

  // The frame as wide as the page (taps need the size); the steps under it.
  view.replaceChildren(chooser, stage, h("div", { class: "card", style: "margin-top:12px" }, panel));
  status.textContent = "Loading the frame";
  img.onload = () => {
    canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
    status.textContent = ""; S.i = nextOpen(0); redraw();
  };
  img.onerror = () => { status.textContent = "Could not load that frame."; };
  img.src = `/api/videos/${enc}/frame/${S.frame}`;

  document.onkeydown = e => {
    if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return;
    const arrows = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
    if (arrows[e.key]) { e.preventDefault(); nudge(...arrows[e.key]); }
    else if (e.key === "u") undo();
  };
}
