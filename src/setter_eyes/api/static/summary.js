// Game summary on the Stats page: where our sets went (by hitter, by zone) and how the
// serve receive went (0-4 counts, and where each pass was played on the court). Uses
// app.js helpers (h, api, label, clock). Colours are CSS tokens in style.css (--viz-*).

const SVGNS = "http://www.w3.org/2000/svg";
function svg(tag, attrs = {}, ...kids) {
  const el = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  for (const kid of kids) if (kid !== null && kid !== undefined) {
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

function summaryFetch(keys) {
  const p = api(`/summary?videos=${encodeURIComponent(keys.join(","))}`);
  p.catch(() => {});                    // shown by summarySection; not an unhandled rejection
  return p;
}

// The summary card, filled in when `pending` (from summaryFetch) arrives.
function summarySection(pending, names) {
  const box = h("div", { class: "card" }, h("strong", {}, "Game summary"),
    h("p", { class: "small muted" }, "loading…"));
  pending
    .then(s => box.replaceChildren(h("strong", {}, "Game summary"), ...summaryBody(s, names)))
    .catch(e => box.replaceChildren(h("strong", {}, "Game summary"),
      h("p", { class: "muted" }, e.message)));
  return box;
}

function who(j, names) { return j === null || j === undefined ? "player not known"
  : `#${j}` + (names.get(j) ? " " + names.get(j) : ""); }

// One horizontal bar per row: label, bar (length = count), the numbers in text.
function barList(rows, max, valueText) {
  return h("div", { class: "barlist" }, ...rows.map(r => h("div", { class: "barrow",
    title: `${r.label}: ${valueText(r)}` },
    h("span", { class: "barlabel" }, r.label),
    h("span", { class: "bartrack" }, h("span", { class: "barfill",
      style: `width:${max ? (100 * r.n / max).toFixed(1) : 0}%` })),
    h("span", { class: "barvalue small" }, valueText(r)))));
}

function summaryBody(s, names) {
  const sets = s.sets, passes = s.passes;
  const kv = r => `${r.n} set${r.n === 1 ? "" : "s"}` +
    (r.kills ? ` · ${r.kills} kill${r.kills === 1 ? "" : "s"}` : "") +
    (r.errors ? ` · ${r.errors} error${r.errors === 1 ? "" : "s"}` : "");
  const maxH = Math.max(0, ...sets.by_hitter.map(r => r.n));
  const maxZ = Math.max(0, ...sets.by_zone.map(r => r.n));
  const setsCol = h("div", { class: "viz-col" },
    h("div", { class: "viz-title" }, `Where the sets went (${sets.n} attacks off our sets)`),
    sets.n ? barList(sets.by_hitter.map(r => ({ ...r, label: who(r.jersey, names) })), maxH, kv)
      : h("p", { class: "muted small" }, "No attacks off our sets yet."),
    sets.n ? h("div", { class: "viz-sub" }, "By zone of the attack") : null,
    sets.n ? barList(sets.by_zone.map(r => ({ ...r, label: r.zone })), maxZ, kv) : null,
    sets.by_setter.length ? h("p", { class: "small muted" }, "Set by: " +
      sets.by_setter.map(r => `${who(r.jersey, names)} ${r.n}`).join(", ")) : null);

  // Pass scores 0-4: counts, then where each pass was played.
  const maxP = Math.max(1, ...passes.by_score.map(r => r.n));
  const scoreBars = h("div", { class: "scorebars" }, ...passes.by_score.map(r =>
    h("div", { class: "scorebar", title: `${r.n} pass${r.n === 1 ? "" : "es"} scored ${r.score}` },
      h("span", { class: "small" }, String(r.n)),
      h("span", { class: "scorefill", style: `height:${(100 * r.n / maxP).toFixed(1)}%;` +
        `background:var(--viz-pass-${r.score})` }),
      h("span", { class: "small muted" }, String(r.score)))));
  let mode = "dots";
  const mapBox = h("div", { class: "passmap" });
  const drawMap = () => mapBox.replaceChildren(passMap(passes.points, s.pass_target, mode, names));
  drawMap();
  const legend = h("div", { class: "row small" }, h("span", { class: "muted" }, "Pass score"),
    ...[0, 1, 2, 3, 4].map(n => h("span", { class: "legend-item" },
      h("span", { class: "swatch", style: `background:var(--viz-pass-${n})` }), String(n))));
  const passCol = h("div", { class: "viz-col" },
    h("div", { class: "viz-title" }, `Serve receive (${passes.n} scored` +
      (passes.avg !== null ? `, average ${passes.avg.toFixed(2)}` : "") + ")"),
    scoreBars,
    h("div", { class: "row spread", style: "margin-top:10px" },
      h("div", { class: "viz-sub" }, `Where the passes went (${passes.points.length} located)`),
      seg([["dots", "Dots"], ["heat", "Heatmap"]], mode, v => { mode = v; drawMap();
        legend.style.visibility = v === "dots" ? "" : "hidden"; })),
    mapBox, legend,
    h("p", { class: "small muted" }, "Our half of the court from behind our end line, the " +
      "net at the top; × is the setter's target. A pass is placed where the setter played " +
      "it (their feet)."));
  return [h("div", { class: "viz-grid" }, setsCol, passCol)];
}

function passMap(points, target, mode, names) {
  // Court metres: x 0 (left sideline) .. 9, y 0 (net, top) .. 9 (our end line).
  const root = svg("svg", { viewBox: "-1 -1 11 11", class: "court", role: "img",
    "aria-label": `Pass map, ${points.length} passes` });
  root.append(svg("rect", { x: 0, y: 0, width: 9, height: 9, class: "court-floor" }));
  if (mode === "heat") {
    const cell = 1.5, n = 6, grid = Array.from({ length: n }, () => Array(n).fill(0));
    for (const p of points) {
      const i = Math.min(n - 1, Math.max(0, Math.floor(p.x / cell)));
      const j = Math.min(n - 1, Math.max(0, Math.floor(p.y / cell)));
      grid[j][i] += 1;
    }
    const max = Math.max(1, ...grid.flat());
    grid.forEach((row, j) => row.forEach((c, i) => {
      if (!c) return;
      const step = Math.min(6, Math.round(6 * c / max));         // --viz-heat-0..6
      root.append(svg("rect", { x: i * cell + 0.02, y: j * cell + 0.02, width: cell - 0.04,
        height: cell - 0.04, rx: 0.08, style: `fill:var(--viz-heat-${step})` },
        svg("title", {}, `${c} pass${c === 1 ? "" : "es"} here`)));
    }));
  }
  root.append(svg("line", { x1: 0, y1: 3, x2: 9, y2: 3, class: "court-line" }));   // attack line
  root.append(svg("rect", { x: 0, y: 0, width: 9, height: 9, class: "court-outline" }));
  root.append(svg("line", { x1: -0.6, y1: 0, x2: 9.6, y2: 0, class: "court-net" }));
  root.append(svg("text", { x: 4.5, y: -0.35, class: "court-label" }, "net"));
  if (target) {
    root.append(svg("path", { d: `M${target.x - 0.25} ${target.y - 0.25}l0.5 0.5m0 -0.5l-0.5 0.5`,
      class: "court-target" }, svg("title", {}, "setter's target")));
  }
  if (mode === "dots") {
    for (const p of points) {
      root.append(svg("circle", { cx: Math.max(-0.8, Math.min(9.8, p.x)),
        cy: Math.max(-0.8, Math.min(9.8, p.y)), r: 0.22, class: "pass-dot",
        style: `fill:var(--viz-pass-${p.score})` },
        svg("title", {}, `${who(p.jersey, names)}: pass ${p.score}, ${p.video} ${clock(p.t)}`)));
    }
  }
  return root;
}
