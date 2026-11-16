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
  const box = h("div", { class: "card" }, h("strong", {}, "Setting and situations"),
    h("p", { class: "small muted" }, "loading…"));
  pending
    .then(s => box.replaceChildren(h("strong", {}, "Setting and situations"),
      ...situationalBody(s, names)))
    .catch(e => box.replaceChildren(h("strong", {}, "Setting and situations"),
      h("p", { class: "muted" }, e.message)));
  return box;
}

const effTxt = d => !d || !d.att ? "-" :
  `${d.eff >= 0 ? "" : "−"}${Math.abs(d.eff).toFixed(3).replace(/^0/, "")} (${d.k}-${d.e}/${d.att})`;
const avgTxt = d => !d || !d.n ? "-" : `${d.avg.toFixed(2)} (${d.n})`;

function situationalBody(s, names) {
  const zn = z => z === "?" ? "not located" : (s.zone_names[z] || z);
  const rule = s.in_system_rule;
  const setters = [...new Set(s.sets.points.map(p => p.setter))];
  const state = { setter: "all", from: "dots", to: "result" };

  // ---- set maps: from (density or dots by system) and to (by result) ----------------------
  const fromBox = h("div", { class: "passmap" }), toBox = h("div", { class: "passmap" });
  const pts = () => s.sets.points.filter(p => state.setter === "all" ||
    String(p.setter) === state.setter);
  const draw = () => {
    fromBox.replaceChildren(setFromMap(pts().filter(p => p.from), s.target, state.from, names));
    toBox.replaceChildren(setToMap(pts().filter(p => p.to), names));
  };
  draw();
  const setterPick = select(["all", ...setters.map(String)], state.setter, v => {
    state.setter = v; draw(); });
  [...setterPick.options].forEach(o => { if (o.value !== "all") o.textContent =
    who(+o.value, names); else o.textContent = "all setters"; });
  const located = pts();
  const maps = h("div", { class: "viz-grid" },
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
      h("p", { class: "small muted" }, "The setter's feet at the set; × is the target. Heatmap: " +
        "how many sets were played from each square.")),
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
  const table = (head, rows) => h("div", { class: "scroll", style: "margin-top:6px" },
    h("table", {}, h("thead", {}, h("tr", {}, ...head.map(x => h("th", {}, x)))),
      h("tbody", {}, ...rows.map(r => h("tr", {}, ...r.map(x => h("td", {}, x)))))));
  const setterRows = s.sets.by_setter.filter(r => r.n >= 3 || s.sets.by_setter.length <= 3)
    .map(r => [who(r.jersey, names), String(r.n), `${r.n_in}`, effTxt(r.in), `${r.n_out}`,
      effTxt(r.out), r.n_unknown ? String(r.n_unknown) : "-",
      r.to.map(t => `${t.zone === "no attack" ? "no attack" : t.zone === "?" ? "?" :
        t.zone}: ${t.n}`).join(" · ")]);
  const zonesSeen = z => s.passers.some(p => p.by_zone.some(b => b.zone === z));
  const pzones = ["4", "3", "2", "5", "6", "1"].filter(zonesSeen);
  const passerRows = s.passers.map(p => [who(p.jersey, names), avgTxt(p), String(p.errors),
    ...pzones.map(z => avgTxt(p.by_zone.find(b => b.zone === z))),
    `${p.located}/${p.n}`]);
  const hzones = ["4", "3", "2", "5", "6", "1"].filter(z =>
    s.hitters.some(x => x.by_zone.some(b => b.zone === z)));
  const sys = (x, k) => effTxt(x.by_system.find(b => b.system === k));
  const hitterRows = s.hitters.map(x => [who(x.jersey, names), effTxt(x),
    ...hzones.map(z => effTxt(x.by_zone.find(b => b.zone === z))), sys(x, "in"), sys(x, "out"),
    x.by_setter.filter(b => b.setter !== "no set").map(b =>
      `${b.setter === null ? "#?" : "#" + b.setter} ${effTxt(b)}`).join(" · ") || "-"]);
  const trendRows = s.trends.map(g => [g.game + (g.played_on ? ` (${g.played_on})` : ""),
    String(g.sets), g.points, g.pass_avg === null ? "-" : `${g.pass_avg.toFixed(2)} (${g.pass_n})`,
    g.hitting_eff === null ? "-" : `${g.hitting_eff.toFixed(3)} (${g.attacks})`,
    g.side_out === null ? "-" : `${Math.round(100 * g.side_out)}% (${g.side_out_n})`]);
  return [
    h("p", { class: "small muted" }, `In system: the pass before the set scored ` +
      `${rule.pass_scores.join(" or ")}; after a dig or free ball, the setter within ` +
      `${rule.max_target_distance_m} m of the target (a draft, set in config/pass_scale.yaml). ` +
      "Efficiency = (kills − errors) / attacks, with (kills-errors/attacks)."),
    h("div", { class: "row" }, h("span", { class: "small" }, "Setter"), setterPick),
    maps,
    h("div", { class: "viz-sub" }, "Setters: the attack after the set, in vs out of system"),
    table(["Setter", "Sets", "In sys", "Eff in sys", "Out sys", "Eff out sys", "Unknown",
      "Where the sets went (zone: n)"], setterRows),
    h("div", { class: "viz-sub" }, "Serve receive by zone: average (passes)"),
    table(["Passer", "All", "Errors", ...pzones.map(zn), "Located"], passerRows),
    h("div", { class: "viz-sub" }, "Hitters: efficiency by zone, system and setter"),
    table(["Hitter", "All", ...hzones.map(zn), "In sys", "Out sys", "By setter"], hitterRows),
    h("div", { class: "viz-sub" }, "Trends across games"),
    table(["Game", "Sets", "Points", "Pass avg", "Hitting eff", "Side-out"], trendRows),
    h("p", { class: "small muted" }, "Only touches whose player was found have a spot or a " +
      "zone (\"located\"); unreviewed rallies use the pipeline's guesses. Set types (tempo, 4, " +
      "52, pipe...) come later (roadmap G1)."),
  ];
}

function courtBase(label) {
  const root = svg("svg", { viewBox: "-1 -1 11 11", class: "court", role: "img",
    "aria-label": label });
  root.append(svg("rect", { x: 0, y: 0, width: 9, height: 9, class: "court-floor" }));
  return root;
}

function courtLines(root, target) {
  root.append(svg("line", { x1: 0, y1: 3, x2: 9, y2: 3, class: "court-line" }));
  root.append(svg("line", { x1: 3, y1: 0, x2: 3, y2: 9, class: "court-zone" }));
  root.append(svg("line", { x1: 6, y1: 0, x2: 6, y2: 9, class: "court-zone" }));
  root.append(svg("rect", { x: 0, y: 0, width: 9, height: 9, class: "court-outline" }));
  root.append(svg("line", { x1: -0.6, y1: 0, x2: 9.6, y2: 0, class: "court-net" }));
  root.append(svg("text", { x: 4.5, y: -0.35, class: "court-label" }, "net"));
  if (target) root.append(svg("path", { d: `M${target.x - 0.25} ${target.y - 0.25}l0.5 0.5m0 -0.5l-0.5 0.5`,
    class: "court-target" }, svg("title", {}, "setter's target")));
}

const clampC = v => Math.max(-0.8, Math.min(9.8, v));

function setFromMap(points, target, mode, names) {
  const root = courtBase(`Where sets were played from, ${points.length} sets`);
  if (mode === "heat") {
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
      root.append(svg("rect", { x: i * cell + 0.02, y: j * cell + 0.02, width: cell - 0.04,
        height: cell - 0.04, rx: 0.08, class: "density",
        style: `fill-opacity:${(0.15 + 0.85 * c / max).toFixed(2)}` },
        svg("title", {}, `${c} set${c === 1 ? "" : "s"} from here`)));
      root.append(svg("text", { x: i * cell + cell / 2, y: j * cell + cell / 2 + 0.17,
        class: "heat-num density-num" }, String(c)));
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
