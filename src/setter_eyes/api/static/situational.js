// Setting and situational stats on the Stats page (roadmap E7/E8; stats/situational.py):
// where our sets were played from and where they went, setters in vs out of system,
// passers by receive zone, hitters by zone / system / setter, and trends across games.
// Uses app.js (h, api, seg, select, clock) and summary.js (svg, who). Every rate shows its
// count: a % of a handful of balls means little.

function situationalFetch(keys) {
  const p = api(`/situational?videos=${encodeURIComponent(keys.join(","))}`);
  p.catch(() => {});
  return p;
}

function situationalSection(pending, names) {
  const title = () => h("h2", {}, "Setting and situations");
  const box = h("section", {}, title(), h("p", { class: "small muted" }, "Loading"));
  pending
    .then(s => box.replaceChildren(title(), ...situationalBody(s, names)))
    .catch(e => box.replaceChildren(title(), h("p", { class: "muted" }, e.message)));
  return box;
}

const effTxt = d => !d || !d.att ? "-" :
  `${d.eff >= 0 ? "" : "-"}${Math.abs(d.eff).toFixed(3).replace(/^0/, "")} (${d.k}-${d.e}/${d.att})`;
const avgTxt = d => !d || !d.n ? "-" : `${d.avg.toFixed(2)} (${d.n})`;

// Setter grids: the court's six attack zones, the net above the front row. A zone's colour
// is a band of hitting efficiency, solid tokens only (DESIGN.md): too few attacks to judge,
// under LOW, LOW up to HIGH, HIGH and up. Display bands, not a coaching threshold.
const GRID_BANDS = { minAttacks: 3, low: 0.100, high: 0.250 };
const GRID_ZONES = [["4", "Left"], ["3", "Middle"], ["2", "Right"], ["5", "Back left"],
  ["6", "Pipe"], ["1", "Back right"]];

function setterGrid(points, title, sub) {
  const att = points.filter(p => p.result !== "no attack");
  const cells = GRID_ZONES.map(([z, name]) => {
    const here = att.filter(p => p.to_zone === z);
    const n = here.length, k = here.filter(p => p.result === "kill").length,
      e = here.filter(p => p.result === "error").length;
    const eff = n ? (k - e) / n : null;
    const band = n < GRID_BANDS.minAttacks ? "few" : eff >= GRID_BANDS.high ? "hi"
      : eff >= GRID_BANDS.low ? "mid" : "lo";
    return h("div", { class: "z-" + band, title: `${name} (${z}): ${n} attacks, ${k} kills, ${e} errors` },
      h("span", {}, `${name} ${n}`),
      n >= GRID_BANDS.minAttacks ? h("span", {}, `Eff ${effStr(eff)}`) : null,
      n >= GRID_BANDS.minAttacks ? h("span", {}, `Kill ${Math.round(100 * k / n)}%`) : null);
  });
  const unplaced = att.filter(p => !p.to_zone).length;
  return h("div", {},
    h("div", { class: "grid-head" }, title, " ", h("span", { class: "muted", style: "font-weight:400" },
      sub(att.length))),
    h("div", { class: "zgrid" }, ...cells),
    unplaced ? h("div", { class: "small muted", style: "margin-top:4px" },
      `${unplaced} attack${unplaced === 1 ? "" : "s"} without a zone`) : null);
}

function situationalBody(s, names) {
  const zn = z => z === "?" ? "not located" : (s.zone_names[z] || z);
  const rule = s.in_system_rule;
  const setters = [...new Set(s.sets.points.map(p => p.setter))];
  const state = { setter: "all", from: "dots", to: "result" };

  // ---- setter grids (all, in system, out of system), then the set maps -----------------
  const gridBox = h("div", { class: "grids" });
  const fromBox = h("div", { class: "passmap" }), toBox = h("div", { class: "passmap" });
  const pts = () => s.sets.points.filter(p => state.setter === "all" ||
    String(p.setter) === state.setter);
  const passes = rule.pass_scores.join(" or ");
  const draw = () => {
    const all = pts();
    gridBox.replaceChildren(
      setterGrid(all, "All sets", n => String(n)),
      setterGrid(all.filter(p => p.system === "in"), "In system", n => `${n}, pass ${passes}`),
      setterGrid(all.filter(p => p.system === "out"), "Out of system", n => String(n)));
    fromBox.replaceChildren(setFromMap(all.filter(p => p.from), s.target, state.from, names));
    toBox.replaceChildren(setToMap(all.filter(p => p.to), names));
  };
  draw();
  const setterPick = select(["all", ...setters.map(String)], state.setter, v => {
    state.setter = v; draw(); });
  [...setterPick.options].forEach(o => { if (o.value !== "all") o.textContent =
    who(+o.value, names); else o.textContent = "all setters"; });
  const maps = h("div", { class: "viz-grid", style: "margin-top:32px" },
    h("div", { class: "viz-col" },
      h("div", { class: "row spread" },
        h("div", { class: "viz-title" }, `Where sets were played from (${s.sets.located_from} ` +
          `of ${s.sets.n} located)`),
        seg([["dots", "Dots"], ["heat", "Heatmap"]], state.from, v => { state.from = v; draw(); })),
      fromBox,
      h("div", { class: "row small" },
        h("span", { class: "legend-item" }, h("span", { class: "swatch sys-in" }), "in system"),
        h("span", { class: "legend-item" }, h("span", { class: "swatch sys-out" }), "out of system"),
        h("span", { class: "legend-item" }, h("span", { class: "swatch sys-unknown" }), "not known")),
      h("p", { class: "small muted" }, "The setter's feet at the set, the cross the target. " +
        "Heatmap: how many sets were played from each square, darker for more.")),
    h("div", { class: "viz-col" },
      h("div", { class: "viz-title" }, `Where the sets went (${s.sets.located_to} attacks located)`),
      toBox,
      h("div", { class: "row small" },
        h("span", { class: "legend-item" }, h("span", { class: "swatch res-kill" }), "kill"),
        h("span", { class: "legend-item" }, h("span", { class: "swatch res-error" }), "error"),
        h("span", { class: "legend-item" }, h("span", { class: "swatch res-play" }), "in play")),
      h("p", { class: "small muted" }, "Where the attack after each set was hit: along the " +
        "net from the ball, front or back row from the hitter's feet.")));

  // ---- tables ----------------------------------------------------------------------------
  const table = (head, rows, textLast = false) => h("div", { class: "scroll", style: "margin-top:6px" },
    // The last column of the setter and hitter tables is a list of words: left-aligned.
    h("table", {}, h("thead", {}, h("tr", {}, ...head.map((x, i) => h("th", {
      class: textLast && i === head.length - 1 ? "t" : "" }, x)))),
      h("tbody", {}, ...rows.map(r => h("tr", {}, ...r.map((x, i) => h("td", {
        class: textLast && i === r.length - 1 ? "t" : "" }, x)))))));
  const setterRows = s.sets.by_setter.filter(r => r.n >= 3 || s.sets.by_setter.length <= 3)
    .map(r => [who(r.jersey, names), String(r.n), `${r.n_in}`, effTxt(r.in), `${r.n_out}`,
      effTxt(r.out), r.n_unknown ? String(r.n_unknown) : "-",
      r.to.map(t => `${t.zone === "no attack" ? "no attack" : t.zone === "?" ? "?" :
        t.zone}: ${t.n}`).join(", ")]);
  const zonesSeen = z => s.passers.some(p => p.by_zone.some(b => b.zone === z));
  const pzones = ["4", "3", "2", "5", "6", "1"].filter(zonesSeen);
  const passerRows = s.passers.map(p => [who(p.jersey, names), avgTxt(p), String(p.errors),
    ...pzones.map(z => avgTxt(p.by_zone.find(b => b.zone === z))),
    `${p.located} of ${p.n}`]);
  const hzones = ["4", "3", "2", "5", "6", "1"].filter(z =>
    s.hitters.some(x => x.by_zone.some(b => b.zone === z)));
  const sys = (x, k) => effTxt(x.by_system.find(b => b.system === k));
  const hitterRows = s.hitters.map(x => [who(x.jersey, names), effTxt(x),
    ...hzones.map(z => effTxt(x.by_zone.find(b => b.zone === z))), sys(x, "in"), sys(x, "out"),
    x.by_setter.filter(b => b.setter !== "no set").map(b =>
      `${b.setter === null ? "#?" : "#" + b.setter} ${effTxt(b)}`).join(", ") || "-"]);
  const trendRows = s.trends.map(g => [g.game + (g.played_on ? ` (${g.played_on})` : ""),
    String(g.sets), g.points, g.pass_avg === null ? "-" : `${g.pass_avg.toFixed(2)} (${g.pass_n})`,
    g.hitting_eff === null ? "-" : `${g.hitting_eff.toFixed(3)} (${g.attacks})`,
    g.side_out === null ? "-" : `${Math.round(100 * g.side_out)}% (${g.side_out_n})`]);
  return [
    h("div", { class: "row", style: "margin-bottom:12px" },
      h("span", { class: "small" }, "Setter"), setterPick),
    h("h3", {}, "Where the setter went"),
    gridBox,
    h("p", { class: "small muted", style: "margin-top:6px" }, "Net above the front row. Blue " +
      `efficiency ${effStr(GRID_BANDS.high)} and up, grey ${effStr(GRID_BANDS.low)} to ` +
      `${effStr(GRID_BANDS.high - 0.001)}, red under ${effStr(GRID_BANDS.low)}, light under ` +
      `${GRID_BANDS.minAttacks} attacks. In system: the pass before the set scored ${passes}; ` +
      `after a dig or free ball, the setter within ${rule.max_target_distance_m} m of the ` +
      "target (a draft, set in config/pass_scale.yaml)."),
    maps,
    h("h3", {}, "Setters: the attack after the set, in and out of system"),
    table(["Setter", "Sets", "In sys", "Eff in sys", "Out sys", "Eff out sys", "Unknown",
      "Where the sets went (zone: n)"], setterRows, true),
    h("h3", {}, "Serve receive by zone: average (passes)"),
    table(["Passer", "All", "Errors", ...pzones.map(zn), "Located"], passerRows),
    h("h3", {}, "Hitters: efficiency by zone, system and setter"),
    table(["Hitter", "All", ...hzones.map(zn), "In sys", "Out sys", "By setter"], hitterRows, true),
    h("p", { class: "small muted", style: "margin-top:6px" }, "Efficiency is (kills - errors) " +
      "/ attacks, shown with (kills-errors/attacks)."),
    h("h3", {}, "Trends across games"),
    table(["Game", "Sets", "Points", "Pass avg", "Hitting eff", "Side-out"], trendRows),
    h("p", { class: "small muted", style: "margin-top:6px" }, "Only touches whose player was " +
      "found have a spot or a zone (\"located\"); unreviewed rallies use the pipeline's " +
      "guesses. Set types (tempo, 4, 52, pipe) come later (roadmap G1)."),
  ];
}

function courtBase(label) {
  const root = svg("svg", { viewBox: "-1 -1 11 11", class: "court", role: "img",
    "aria-label": label });
  root.append(svg("rect", { x: 0, y: 0, width: 9, height: 9, class: "court-floor" }));
  return root;
}

// The court's painted lines (white with an ink edge, so they read on maple), the net, the
// zones named outside the court, and the setter's target as a cross.
function courtLines(root, target) {
  const lines = cls => [
    svg("rect", { x: 0, y: 0, width: 9, height: 9, class: cls }),
    svg("line", { x1: 0, y1: 3, x2: 9, y2: 3, class: cls })];
  root.append(...lines("court-edge"), ...lines("court-outline"));
  root.append(svg("line", { x1: 3, y1: 0, x2: 3, y2: 9, class: "court-zone" }));
  root.append(svg("line", { x1: 6, y1: 0, x2: 6, y2: 9, class: "court-zone" }));
  root.append(svg("line", { x1: -0.6, y1: 0, x2: 9.6, y2: 0, class: "court-net" }));
  [["4", 1.5], ["3", 4.5], ["2", 7.5]].forEach(([z, x]) =>
    root.append(svg("text", { x, y: -0.3, class: "court-label" }, z)));
  [["5", 1.5], ["6", 4.5], ["1", 7.5]].forEach(([z, x]) =>
    root.append(svg("text", { x, y: 9.6, class: "court-label" }, z)));
  if (target) root.append(svg("path", { d: `M${target.x - 0.25} ${target.y - 0.25}l0.5 0.5m0 -0.5l-0.5 0.5`,
    class: "court-target" }, svg("title", {}, "setter's target")));
}

const clampC = v => Math.max(-0.8, Math.min(9.8, v));

function setFromMap(points, target, mode, names) {
  const root = courtBase(`Where sets were played from, ${points.length} sets`);
  if (mode === "heat") {
    // Three solid steps of how many sets came from a square (no opacity, DESIGN.md).
    const cell = 1.5, n = 6;
    const grid = Array.from({ length: n }, () => Array(n).fill(0));
    for (const p of points) {
      const i = Math.min(n - 1, Math.max(0, Math.floor(p.from[0] / cell)));
      const j = Math.min(n - 1, Math.max(0, Math.floor(p.from[1] / cell)));
      grid[j][i] += 1;
    }
    const max = Math.max(1, ...grid.flat());
    grid.forEach((row, j) => row.forEach((c, i) => {
      if (!c) return;
      const step = c >= (2 / 3) * max ? 3 : c >= max / 3 ? 2 : 1;
      root.append(svg("rect", { x: i * cell + 0.02, y: j * cell + 0.02, width: cell - 0.04,
        height: cell - 0.04, class: `dens-${step}` },
        svg("title", {}, `${c} set${c === 1 ? "" : "s"} from here`)));
      root.append(svg("text", { x: i * cell + cell / 2, y: j * cell + cell / 2 + 0.17,
        class: `heat-num dens-num-${step}` }, String(c)));
    }));
  }
  courtLines(root, target);
  if (mode === "dots") {
    for (const p of points) {
      root.append(svg("circle", { cx: clampC(p.from[0]), cy: clampC(p.from[1]), r: 0.24,
        class: `set-dot sys-${p.system}` },
        svg("title", {}, `${who(p.setter, names)}: ${p.system === "unknown" ? "system not known"
          : p.system + " system"}` + (p.pass_score !== null ? ` (pass ${p.pass_score})` : "") +
          `, ${p.result}${p.to_zone ? " in zone " + p.to_zone : ""}, ${p.video} ${clock(p.t)}`)));
    }
  }
  return root;
}

function setToMap(points, names) {
  const root = courtBase(`Where the sets went, ${points.length} attacks`);
  courtLines(root, null);
  const cls = { kill: "res-kill", error: "res-error", "in play": "res-play" };
  for (const p of points) {
    root.append(svg("circle", { cx: clampC(p.to[0]), cy: clampC(p.to[1]), r: 0.24,
      class: `set-dot ${cls[p.result] || "res-play"}` },
      svg("title", {}, `${who(p.hitter, names)} off ${who(p.setter, names)}'s set: ` +
        `${p.result}, ${p.video} ${clock(p.t)}`)));
  }
  return root;
}
