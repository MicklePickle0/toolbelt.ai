// Search: every touch of ours in the picked sets that matches the filters (player, what
// it counts as, how the point ended, pass score, who set it), with its clip, and a
// highlight reel of the ones ticked, the player ringed (Hudl style). Uses app.js helpers.

const SEARCH_ACTIONS = ["serve", "receive", "overpass", "set", "attack", "dump", "free_ball",
  "dig", "free_ball_pass", "block", "coverage", "ball_handling_error"];

async function searchPage(params) {
  const [allVids, fd, rosters] = await Promise.all(
    [api("/videos"), api("/folders"), api("/rosters")]);
  const vids = allVids.filter(v => v.rallies);
  const people = new Map();
  for (const r of rosters) for (const p of r.players) if (!people.has(p.number)) people.set(p.number, p.name);
  const numbers = [...people.keys()].sort((a, b) => a - b);

  const q = {
    v: (params.get("v") || "").split(",").filter(Boolean),
    player: params.get("player") || "", action: params.get("action") || "",
    result: params.get("result") || "", set_by: params.get("set_by") || "",
    pass: (params.get("pass") || "").split(",").filter(x => x !== ""),
  };
  if (!q.v.length && vids.length) q.v = vids.map(v => v.key);
  const go = () => {
    const p = new URLSearchParams();
    p.set("v", q.v.join(","));
    for (const k of ["player", "action", "result", "set_by"]) if (q[k]) p.set(k, q[k]);
    if (q.pass.length) p.set("pass", q.pass.join(","));
    location.hash = "#/search?" + p.toString();
  };
  const playerSelect = (key, blank) => {
    const el = h("select", { onchange: () => { q[key] = el.value; go(); } },
      h("option", { value: "" }, blank),
      ...numbers.map(n => h("option", { value: n, selected: String(n) === q[key] },
        `#${n} ${people.get(n)}`)));
    return el;
  };
  const filters = h("div", { class: "well" },
    h("div", { class: "filterbar" },
      ...scopePicker(fd, vids, q.v, keys => { q.v = keys; go(); })),
    h("div", { class: "filterbar", style: "margin-top:12px" },
      h("label", { class: "fl" }, "Player", playerSelect("player", "anyone")),
      h("label", { class: "fl" }, "Play",
        select(SEARCH_ACTIONS, q.action || null, v => { q.action = v || ""; go(); }, "any")),
      h("label", { class: "fl" }, "How the point ended",
        select(["kills", "aces", "attack_errors", "service_errors", "block_points",
          "reception_errors", "ball_handling_errors"], q.result || null,
        v => { q.result = v || ""; go(); }, "any")),
      h("label", { class: "fl" }, "Set by", playerSelect("set_by", "anyone"))),
    h("div", { class: "row", style: "margin-top:12px" },
      h("span", { class: "small" }, "Pass score"),
      ...[0, 1, 2, 3, 4].map(n => h("button", {
        class: q.pass.includes(String(n)) ? "on" : "", onclick: () => {
          q.pass = q.pass.includes(String(n)) ? q.pass.filter(x => x !== String(n))
            : [...q.pass, String(n)];
          go(); } }, String(n))),
      h("span", { class: "small muted" }, "(only serve receives have one)")),
    h("p", { class: "small muted", style: "margin:8px 0 0" }, "Example: player #12, play " +
      "attack, set by #17 finds every attack #12 hit off #17's set."));

  const p = new URLSearchParams({ videos: q.v.join(",") });
  for (const k of ["player", "action", "result", "set_by"]) if (q[k]) p.set(k, q[k]);
  if (q.pass.length) p.set("passes", q.pass.join(","));
  let items = [];
  let error = null;
  try { items = q.v.length ? (await api("/search?" + p.toString())).items : []; }
  catch (e) { error = e.message; }

  // The clip of the touch picked in the list.
  const video = h("video", { playsinline: true, controls: true, preload: "auto" });
  const caption = h("div", { class: "small muted" }, "Pick a point below to watch it.");
  const play = it => {
    if (!it.rally_id) { toast("That rally isn't in the event DB."); return; }
    const t0 = Math.max(0, it.t - 4), t1 = it.t + 3;
    video.src = `/api/rallies/${it.rally_id}/clip?t0=${t0.toFixed(2)}&t1=${t1.toFixed(2)}`;
    video.play().catch(() => {});
    caption.textContent = `${it.video}, rally ${it.rally_number}, ${clock(it.t)}: ` +
      `${it.jersey ? "#" + it.jersey : "player ?"} ${label(it.action)}` +
      (it.pass_score !== null ? `, pass ${it.pass_score}` : "") +
      (it.result ? `, ${label(it.result)}` : "");
  };

  const picked = new Set(items.map((_, i) => i));
  const count = h("span", { class: "small" });
  const reelBox = h("div", { class: "row" });
  const showCount = () => { count.textContent = `${picked.size} of ${items.length} ticked`; };
  const rows = items.map((it, i) => h("tr", {},
    h("td", {}, h("input", { type: "checkbox", checked: true, onchange: e => {
      if (e.target.checked) picked.add(i); else picked.delete(i); showCount(); } })),
    h("td", {}, h("button", { class: "iconbtn", title: "Watch this point", onclick: () => play(it) },
      "Watch")),
    h("td", { class: "small t" }, it.video),
    h("td", {}, it.rally_id ? h("a", { href: `#/review/${setOf(it.video)}/${it.rally_id}?all=1` },
      `${it.rally_number}`) : `${it.rally_number}`),
    h("td", {}, clock(it.t)),
    h("td", { class: "t" }, it.jersey ? `#${it.jersey}` : "?"),
    h("td", { class: "t" }, label(it.action)),
    h("td", {}, it.pass_score === null ? "" : String(it.pass_score)),
    h("td", { class: "t" }, it.result ? label(it.result) : ""),
    h("td", { class: "t" }, it.set_by ? `#${it.set_by}` : "")));
  function setOf(key) { const v = vids.find(x => x.key === key); return v ? v.set_id : ""; }
  showCount();

  const makeReel = async () => {
    const chosen = items.filter((_, i) => picked.has(i));
    if (!chosen.length) { toast("Tick at least one point."); return; }
    const who = q.player ? `player${q.player}` : "team";
    const name = [who, q.action, q.result, q.set_by ? `from${q.set_by}` : ""]
      .filter(Boolean).join("_");
    try {
      const res = await api("/reels", { body: { name,
        items: chosen.map(it => ({ video: it.video, t: it.t, jersey: it.jersey })) } });
      pollReel(res.id, chosen.length);
    } catch (e) { toast(e.message, 6000); }
  };
  const pollReel = async (id, n) => {
    const st = await api(`/reels/${id}`);
    if (st.state === "done") {
      reelBox.replaceChildren(h("a", { class: "button primary", href: `/api/reels/${id}/file`,
        download: "" }, `Download reel (${n} clips, ${st.ringed} with the ring)`));
      return;
    }
    if (st.state === "failed") { reelBox.replaceChildren(h("span", { class: "small bad" },
      "Reel failed: " + st.error)); return; }
    reelBox.replaceChildren(bar(st.total ? st.done / st.total : 0, `making the reel, ` +
      `${st.done}/${st.total} clips`));
    setTimeout(() => pollReel(id, n).catch(e => toast(e.message)), 1500);
  };

  const results = h("section", {},
    h("div", { class: "row spread" },
      h("h2", { style: "margin:0" }, error ? "Error" : `${items.length} point${items.length === 1 ? "" : "s"}`),
      h("div", { class: "row" }, count,
        h("button", { onclick: () => { items.forEach((_, i) => picked.add(i));
          rows.forEach(r => { r.querySelector("input").checked = true; }); showCount(); } },
          "Tick all"),
        h("button", { onclick: () => { picked.clear();
          rows.forEach(r => { r.querySelector("input").checked = false; }); showCount(); } },
          "Untick all"),
        h("button", { class: "primary", onclick: makeReel },
          "Make a reel of the ticked points"))),
    reelBox,
    error ? h("p", { class: "muted" }, error) : items.length ? h("div", { class: "scroll" },
      h("table", {}, h("thead", {}, h("tr", {}, ...["", "", "Set", "Rally", "Time", "Player",
        "Play", "Pass", "Point", "Set by"].map(x => h("th", {
          class: ["Set", "Player", "Play", "Point", "Set by"].includes(x) ? "t" : "" }, x)))),
      h("tbody", {}, rows)))
      : h("p", { class: "muted" }, "Nothing matches. Players are only known where the " +
        "pipeline read a number or you set them in review."));

  view.replaceChildren(h("h1", {}, "Search"), filters,
    h("div", { class: "card", style: "max-width:880px" }, video, caption), results);
}
