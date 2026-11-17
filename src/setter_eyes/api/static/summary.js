// Game summary on the Stats page: where our sets went (by hitter, by zone) and how the
// serve receive went (0-4 counts, and where each pass was played on the court). Uses
// app.js helpers (h, api, label, clock) and situational.js (courtLines). Colours are the
// DESIGN.md tokens in style.css (--pass-N, maple court).

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
  p.catch(() => {});                    // the Stats tab awaits it; not an unhandled rejection
  return p;
}

// The summary card, filled in when `pending` (from summaryFetch) arrives.

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

// Setting tab, after the grids: who our sets went to and in which zone, as bars.
function summarySets(s, names) {
  const sets = s.sets;
  const kv = r => `${r.n} set${r.n === 1 ? "" : "s"}` +
    (r.kills ? `, ${r.kills} kill${r.kills === 1 ? "" : "s"}` : "") +
    (r.errors ? `, ${r.errors} error${r.errors === 1 ? "" : "s"}` : "");
  const maxH = Math.max(0, ...sets.by_hitter.map(r => r.n));
  const maxZ = Math.max(0, ...sets.by_zone.map(r => r.n));
  if (!sets.n) return [h("h2", {}, "Who the sets went to"), h("p", { class: "muted" }, "No attacks off our sets yet.")];
  return [h("div", { class: "viz-grid", style: "margin-top:48px" },
    h("div", {}, h("h2", { style: "margin-top:0" }, "Who the sets went to"),
      barList(sets.by_hitter.map(r => ({ ...r, label: who(r.jersey, names) })), maxH, kv),
      sets.by_setter.length ? h("p", { class: "small muted", style: "margin-top:8px" }, "Set by " +
        sets.by_setter.map(r => `${who(r.jersey, names)} ${r.n}`).join(", ")) : null),
    h("div", {}, h("h2", { style: "margin-top:0" }, "Zone of the attack"),
      barList(sets.by_zone.map(r => ({ ...r, label: r.zone })), maxZ, kv)))];
}

// Passing tab: pass scores 0-4 as counts, then where each pass was played.
function summaryPassing(s, names) {
  const passes = s.passes;
  const maxP = Math.max(1, ...passes.by_score.map(r => r.n));
  const scoreBars = h("div", { class: "scorebars" }, ...passes.by_score.map(r =>
    h("div", { class: "scorebar", title: `${r.n} pass${r.n === 1 ? "" : "es"} scored ${r.score}` },
      h("span", { class: "small" }, String(r.n)),
      h("span", { class: "scorefill", style: `height:${(100 * r.n / maxP).toFixed(1)}%;` +
        `background:var(--pass-${r.score})` }),
      h("span", { class: "small muted" }, String(r.score)))));
  let mode = "dots";
  const mapBox = h("div", { class: "passmap" });
  const drawMap = () => mapBox.replaceChildren(passMap(passes.points, s.pass_target, mode, names));
  drawMap();
  const legend = h("div", { class: "row small" }, h("span", { class: "muted" }, "Pass score"),
    ...[0, 1, 2, 3, 4].map(n => h("span", { class: "legend-item" },
      h("span", { class: "swatch", style: `background:var(--pass-${n})` }), String(n))));
  return [h("div", { class: "viz-grid" },
    h("div", {},
      h("h2", { style: "margin-top:0" }, "Pass scores"),
      h("p", { class: "muted" }, `${passes.n} serves received` +
        (passes.avg !== null ? `, average ${passes.avg.toFixed(2)}` : "")),
      scoreBars),
    h("div", {},
      h("div", { class: "row spread" },
        h("h2", { style: "margin:0" }, "Where the passes went"),
        seg([["dots", "Dots"], ["heat", "Heatmap"]], mode, v => { mode = v; drawMap(); })),
      mapBox, legend, missingLine(passes),
      h("p", { class: "small muted" }, "Placed where the setter played the pass. The cross " +
        "is the setter's target; the heatmap shows the average score per square.")))];
}

// Scored passes with no spot on the map, by score. Bad passes are the ones most often
// missing (no clean second ball to place them by), so the map alone looks too good.
function missingLine(passes) {
  const onMap = new Map();
  for (const p of passes.points) onMap.set(p.score, (onMap.get(p.score) || 0) + 1);
  const missing = passes.by_score.map(r => [r.score, r.n - (onMap.get(r.score) || 0)])
    .filter(([, n]) => n > 0);
  if (!missing.length) return null;
  const total = missing.reduce((a, [, n]) => a + n, 0);
  return h("p", { class: "small" }, h("strong", {}, `Not on the map: ${total} of ${passes.n}`),
    " (no spot found for them): ", missing.map(([s, n]) => `${n} scored ${s}`).join(", "), ".");
}

function passMap(points, target, mode, names) {
  // Court metres: x 0 (left sideline) .. 9, y 0 (net, top) .. 9 (our end line).
  const root = svg("svg", { viewBox: "-1 -1 11 11", class: "court", role: "img",
    "aria-label": `Pass map, ${points.length} passes` });
  root.append(svg("rect", { x: 0, y: 0, width: 9, height: 9, class: "court-floor" }));
  if (mode === "heat") {
    // Each 1.5 m square: the average score of the passes played there (colour and number),
    // with how many passes that is underneath.
    const cell = 1.5, n = 6;
    const grid = Array.from({ length: n }, () => Array.from({ length: n }, () => []));
    for (const p of points) {
      const i = Math.min(n - 1, Math.max(0, Math.floor(p.x / cell)));
      const j = Math.min(n - 1, Math.max(0, Math.floor(p.y / cell)));
      grid[j][i].push(p.score);
    }
    grid.forEach((row, j) => row.forEach((scores, i) => {
      if (!scores.length) return;
      const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
      const step = Math.round(avg), cx = i * cell + cell / 2, cy = j * cell + cell / 2;
      root.append(svg("rect", { x: i * cell + 0.02, y: j * cell + 0.02, width: cell - 0.04,
        height: cell - 0.04, style: `fill:var(--pass-${step})` },
        svg("title", {}, `${scores.length} pass${scores.length === 1 ? "" : "es"} here, ` +
          `average ${avg.toFixed(2)}`)));
      root.append(svg("text", { x: cx, y: cy - 0.12, class: "heat-num",
        style: `fill:var(--pass-ink-${step})` }, avg.toFixed(1)));
      root.append(svg("text", { x: cx, y: cy + 0.42, class: "heat-n",
        style: `fill:var(--pass-ink-${step})` }, `n=${scores.length}`));
    }));
  }
  courtLines(root, target);                              // situational.js
  if (mode === "dots") {
    for (const p of points) {
      const cx = Math.max(-0.8, Math.min(9.8, p.x)), cy = Math.max(-0.8, Math.min(9.8, p.y));
      root.append(svg("circle", { cx, cy, r: 0.28, class: "pass-dot",
        style: `fill:var(--pass-${p.score})` },
        svg("title", {}, `${who(p.jersey, names)}: pass ${p.score}, ${p.video} ${clock(p.t)}`)));
      root.append(svg("text", { x: cx, y: cy, class: "pass-num",
        style: `fill:var(--pass-ink-${p.score})` }, String(p.score)));
    }
  }
  return root;
}
