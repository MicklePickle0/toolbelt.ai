// Rosters page (#/rosters, #/rosters/<slug>): our players (number, name, height, position),
// several rosters, each applied to the videos it played. Saved as data/rosters/*.yaml
// (rosters.py). Uses api(), h(), toast() and `view` from app.js.
"use strict";

const POSITIONS = ["S", "OH", "MB", "OPP", "L", "DS"];

function feetInches(cm) {
  if (!cm) return "";
  const inches = Math.round(cm / 2.54);
  return `${Math.floor(inches / 12)}'${inches % 12}"`;
}

async function rostersPage(slug) {
  if (slug) return rosterEdit(decodeURIComponent(slug));
  const list = await api("/rosters");
  const name = h("input", { type: "text", placeholder: "Roster name, e.g. Venom 16U" });
  const season = h("input", { type: "text", placeholder: "Season, e.g. 2026-27" });
  const create = async () => {
    if (!name.value.trim()) { toast("Give the roster a name."); return; }
    try {
      const r = await api("/rosters", { body: { name: name.value, season: season.value } });
      location.hash = `#/rosters/${r.slug}`;
    } catch (e) { toast(e.message, 5000); }
  };
  view.replaceChildren(h("h1", {}, "Rosters"),
    h("p", { class: "muted small" }, "One roster per team and season. Apply a roster to the " +
      "videos it played so stats show names, and so players can be queried across games."),
    ...list.map(r => h("div", { class: "card" },
      h("div", { class: "row spread" },
        h("div", {}, h("h2", { style: "margin:0" }, r.name),
          h("div", { class: "small muted" }, [r.season, `${r.players.length} players`,
            `${r.videos.length} videos`].filter(Boolean).join(", "))),
        h("button", { onclick: () => { location.hash = `#/rosters/${r.slug}`; } },
          "Edit")),
      h("div", { class: "small", style: "margin-top:6px" },
        r.players.map(p => `#${p.number} ${p.name}`).join(", ") || "No players yet."))),
    h("div", { class: "well" }, h("strong", {}, "New roster"),
      h("div", { class: "row", style: "margin-top:8px" }, name, season,
        h("button", { class: "primary", onclick: create }, "Create"))));
}

async function rosterEdit(slug) {
  const [list, vids] = await Promise.all([api("/rosters"), api("/videos")]);
  const r = list.find(x => x.slug === slug);
  if (!r) { view.replaceChildren(h("p", {}, `No roster ${slug}.`)); return; }
  const rows = r.players.map(p => ({ ...p, height: p.height_cm ? String(p.height_cm) : "" }));
  const name = h("input", { type: "text", value: r.name });
  const season = h("input", { type: "text", value: r.season || "", placeholder: "season" });
  const notes = h("input", { type: "text", value: r.notes || "", placeholder: "notes",
    style: "flex:1;min-width:200px" });
  const table = h("div");
  const positions = h("datalist", { id: "positions" }, ...POSITIONS.map(p => h("option", { value: p })));

  function drawRows() {
    table.replaceChildren(
      h("div", { class: "roster-row head small muted" }, h("span", {}, "No."), h("span", {}, "Name"),
        h("span", {}, "Height (cm or 6'2\")"), h("span", {}, "Pos."), h("span", {}, "Libero"),
        h("span", {})),
      ...rows.map((p, i) => {
        const hint = h("span", { class: "small muted" });
        const showHint = v => {
          const n = parseFloat(v);
          hint.textContent = /^\s*\d+(\.\d+)?\s*(cm)?\s*$/.test(v) && n > 100 ? feetInches(n) : "";
        };
        const height = h("input", { type: "text", value: p.height, oninput: e => {
          p.height = e.target.value; showHint(e.target.value); } });
        showHint(p.height);
        return h("div", { class: "roster-row" },
          h("input", { type: "number", min: 0, max: 99, value: p.number,
            oninput: e => { p.number = e.target.value === "" ? null : +e.target.value; } }),
          h("input", { type: "text", value: p.name, oninput: e => { p.name = e.target.value; } }),
          h("div", { class: "row", style: "gap:4px;flex-wrap:nowrap" }, height, hint),
          h("input", { type: "text", list: "positions", value: p.position || "",
            oninput: e => { p.position = e.target.value; } }),
          h("input", { type: "checkbox", checked: p.libero,
            onchange: e => { p.libero = e.target.checked; } }),
          h("button", { class: "iconbtn", title: "Remove this player", onclick: () => {
            rows.splice(i, 1); drawRows(); } }, "Remove"));
      }),
      h("button", { style: "margin-top:8px", onclick: () => {
        rows.push({ number: null, name: "", height: "", position: "", libero: false });
        drawRows(); table.querySelectorAll(".roster-row:last-of-type input")[0]?.focus(); } },
      "Add player"));
  }

  async function save() {
    const players = rows.filter(p => p.number !== null || p.name.trim());
    if (players.some(p => p.number === null || p.number === undefined)) {
      toast("Every player needs a jersey number."); return;
    }
    try {
      await api(`/rosters/${slug}`, { method: "PUT", body: {
        name: name.value, season: season.value, notes: notes.value,
        players: players.map(p => ({ number: p.number, name: p.name, height: p.height || null,
          position: p.position || null, libero: !!p.libero })) } });
      toast("Saved"); rosterEdit(slug);
    } catch (e) { toast(e.message, 6000); }
  }

  const videoBoxes = h("div", { class: "row" }, ...vids.map(v => {
    const mine = v.roster && v.roster.slug === slug;
    const other = v.roster && !mine ? ` (on ${v.roster.name})` : "";
    return h("label", { class: "check chip" },
      h("input", { type: "checkbox", checked: mine, onchange: async e => {
        if (e.target.checked && other && !confirm(`${v.key} uses ${v.roster.name}. Switch it?`)) {
          e.target.checked = false; return;
        }
        try {
          await api(`/videos/${encodeURIComponent(v.key)}/roster`, { method: "PUT",
            body: { roster: e.target.checked ? slug : null } });
          toast(e.target.checked ? `${v.key}: ${r.name}` : `${v.key}: no roster`);
        } catch (err) { toast(err.message, 5000); e.target.checked = !e.target.checked; }
      } }), v.key + other);
  }));

  drawRows();
  view.replaceChildren(
    h("div", { class: "row spread" }, h("h1", {}, r.name),
      h("a", { href: "#/rosters" }, "All rosters")),
    h("div", { class: "well" }, h("div", { class: "row" }, name, season, notes)),
    h("div", { class: "card" }, h("h2", { style: "margin-top:0" }, "Players"), positions, table,
      h("div", { class: "row", style: "margin-top:10px" },
        h("button", { class: "primary", onclick: save }, "Save roster"))),
    h("div", { class: "card" }, h("h2", {}, "Videos this roster played"),
      h("p", { class: "small muted" }, "Tick each set this team played. Stats for those " +
        "videos use these names, and touches credited to a jersey follow the roster."),
      videoBoxes),
    h("div", { class: "card" }, h("button", { onclick: async () => {
      if (!confirm(`Delete the roster ${r.name}? (It stays in git history.)`)) return;
      try { await api(`/rosters/${slug}`, { method: "DELETE" }); location.hash = "#/rosters"; }
      catch (e) { toast(e.message, 6000); } } }, "Delete roster")));
}
