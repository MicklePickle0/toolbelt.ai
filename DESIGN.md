---
version: alpha
name: Toolbelt
description: A volleyball film room and stat sheet for a club coach, used after the set on a laptop or iPad; it should feel like sitting down with the tape and the scorebook, not like a SaaS dashboard.
colors:
  paper: "#FFFFFF"        # stat-sheet ground (Stats, Videos, Search, Rosters, Queue)
  well: "#EEF0F2"         # grouped regions instead of bordered cards
  ink: "#14171C"          # text on paper, 17.9:1
  muted: "#5B6470"        # secondary text, 6.0:1 on paper, 5.3:1 on well
  court: "#1E4D8C"        # our team, links, the selected thing; 8.4:1 on paper
  ball: "#F2C12E"         # the one action colour: primary button fill, current touch, focus ring
  them: "#8B929C"         # the opponent in data only (never text)
  miss: "#B8322A"         # errors in data and form errors; 6.0:1 on paper
  maple: "#E2CDA2"        # the floor of every court map
  film: "#15181C"         # Review and Calibrate ground, around the video
  film-surface: "#23272C" # panels inside the film room
  film-ink: "#E8EAED"     # text on film, 14.8:1
  film-muted: "#A3AAB3"   # 7.6:1 on film, 6.4:1 on film-surface
  film-court: "#7FA8E0"   # our team on film, 7.3:1
  film-miss: "#F08A7E"    # errors on film, 7.3:1
  pass-4: "#1E4D8C"
  pass-3: "#5A8FD0"
  pass-2: "#8B929C"
  pass-1: "#D9772E"
  pass-0: "#B8322A"
typography:
  display:
    fontFamily: Barlow Condensed
    fontSize: 40px
    fontWeight: 600
    letterSpacing: 0
  heading:
    fontFamily: Barlow Condensed
    fontSize: 24px
    fontWeight: 600
    letterSpacing: 0
  body:
    fontFamily: Barlow
    fontSize: 15px
    fontWeight: 400
    lineHeight: 1.45
  table:
    fontFamily: Barlow
    fontSize: 14px
    fontWeight: 400
    fontFeature: tnum
  small:
    fontFamily: Barlow
    fontSize: 13px
    fontWeight: 400
rounded:
  none: 0px
  sm: 3px
  md: 6px
spacing:
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
  xl: 48px
components:
  button-primary:
    backgroundColor: "{colors.ball}"
    textColor: "{colors.ink}"
    rounded: "{rounded.sm}"
    padding: 8px 14px
  button:
    backgroundColor: "{colors.well}"
    textColor: "{colors.ink}"
    rounded: "{rounded.sm}"
    padding: 8px 14px
  button-film:
    backgroundColor: "{colors.film-surface}"
    textColor: "{colors.film-ink}"
    rounded: "{rounded.sm}"
    padding: 8px 14px
---

## Overview

Toolbelt is used by a volleyball coach after a tournament set: they watch the rallies, fix
what the pipeline got wrong, then read the stats to plan practice. Two rooms, so the app has
two environments:

- **The film room** (Review, Calibrate) is dark, because video needs a dark surround and the
  coach's eyes stay on the clip. Taken from Hudl: video first, the list of touches beside it
  and in sync with it, and tagging from the keyboard.
- **The stat sheet** (Videos, Stats, Search, Rosters, Queue) is paper white, because it is
  read like a printed box score. Taken from Cleaning the Glass: tables first, numbers that
  carry their context (every rate has its count, every split says what it was split on),
  plain prose to explain a filter, almost no chrome.

From the Loire tennis shot (now seen, see References): restraint, a large step between the
display size and the body size, and real images instead of decoration. Here the real images
are frames from the team's own footage, never stock photos or illustrations.

The volleyball itself supplies the rest: the maple floor and painted lines of the court
maps, the ball's yellow as the one action colour, our side in court blue, the other side in
grey because we only measure our side.

## Colors

Solid colours only. No opacity, no tints of a colour to make a new one, no gradients.

- **Paper (#FFFFFF) and well (#EEF0F2):** the stat sheet. A well is how regions are grouped;
  there are no bordered cards.
- **Ink (#14171C) and muted (#5B6470):** text. Muted is for counts, units and helper lines,
  never for a number the coach came to read.
- **Court (#1E4D8C):** us. Links, our team in charts, the selected row. On film it becomes
  film-court (#7FA8E0).
- **Ball (#F2C12E):** the single action colour, always with ink text. The primary button on a
  page (one per page), the current touch on the review timeline, and the keyboard focus
  ring. Nothing else is yellow.
- **Them (#8B929C):** the opponent in data. It is never text, so its 3.1:1 on paper is
  enough for marks and too low for words.
- **Miss (#B8322A):** errors (attack errors, service errors, 0 passes) and form errors.
- **Maple (#E2CDA2):** the floor of the court maps, and nowhere else. It is not a page
  ground.
- **Pass scale 4 to 0:** court blue, light blue, grey, orange, red. Marks on maple always
  get a 1px ink ring, since the middle colours are below 3:1 on the floor by themselves.
- **Film (#15181C), film-surface (#23272C), film-ink, film-muted:** only in the film room.
  The dark ground is there for the video, so the yellow accent on it stays rare: the
  current touch, the primary button and focus, nothing else.

## Typography

- **Barlow Condensed 600:** page titles (40px display on Stats and Videos, 24px elsewhere),
  and the big numbers: the score, a player's line at the top of their stats, the
  hitting efficiency. Its narrow figures recall jersey and scoreboard numbers.
- **Barlow 400 / 600:** everything else. Tables use tabular figures (`font-feature-settings:
  "tnum"`) so columns of numbers line up.
- Sentence case everywhere. No all-caps labels, no italics, no negative letter spacing, no
  periods at the end of titles or buttons. Monospace only for a file path or a key name
  when it must look typed.
- Self-host both families (OFL) under `static/fonts/`; the app runs on Tailscale and must
  not depend on a font CDN.

## Layout

- One top bar: "Toolbelt" on the left, then Videos, Stats, Search, Rosters, Queue. The
  current page is ink and 600, the others muted. No breadcrumbs: the page title says where
  you are (for example "Pakmen Black, set 2").
- Stat sheet pages: content up to 1200px wide, aligned left, 16px side gutter on a phone.
  Sections are separated by space (48px), not lines. Headings say what the section shows
  ("Passing by zone"), never a number or an eyebrow label above them.
- Film room: the video takes the left 60%, the touches list the right 40%; below 900px they
  stack, video on top and kept in view. The ground is film from edge to edge.

### How the coach moves through it

1. **Videos.** One row per set, grouped by game: a frame from the video, the name, the
   score, and one line on what it needs ("12 rallies to review", "Not calibrated"). The one
   primary button on each row is the next step. A set's status is a short phrase in muted
   text, not a coloured badge.
2. **Calibrate.** Film room. The frame fills the space, the court drawn over it in court
   lines; controls in one strip below.
3. **Review.** Film room. Video, then the touches beside it as a plain list (time, team,
   action, player, pass). The current touch has the ball-yellow bar on its left and plays
   in sync with the video. Keys are listed once in a small muted line under the list.
   Score checks appear as one sentence above the list when a rally is flagged.
4. **Stats.** Stat sheet. The box score first (players by rows, numbers right-aligned,
   team total in 600), then the court maps (passing, set from / set to), then the
   situational tables. Every rate shows its count next to it in muted ("0.059 on 34").
5. **Search, Rosters, Queue.** Same sheet, same tables.

## Elevation & depth

Flat. No shadows, except the one menu or popover that floats over content, which gets a
single 1px ink-at-full-strength outline. Depth comes from paper vs well, and from paper vs
film.

## Shapes

Buttons and inputs 3px radius, video and frames 6px, court maps and tables 0. No pills, no
squircles, no fully rounded chips.

## Components

- **Tables:** no outer border. One ink rule under the header row, nothing between body rows;
  the hovered or selected row gets the well colour. The first column (player) sticks on
  narrow screens. Numbers right-aligned in tabular figures, names left.
- **Court map:** the same half-court everywhere: maple floor, 2px white lines (net, attack
  line, sidelines, end line) with a 1px ink edge so they read on maple, and zones 1-6 named
  in muted text outside the court, not inside it. Our marks in court blue or the pass scale,
  theirs in grey. One map per question; a legend is one line of text under it.
- **Buttons:** primary (ball yellow, ink text) once per page; ordinary (well); text links in
  court blue for navigation. Labels are verbs ("Review", "Process set"), no arrows or icons
  glued on.
- **Status and flags:** a short sentence in muted text, or in miss red for a real problem.
  No tags, pills, badges or emoji.
- **Focus:** every interactive element shows a 2px ball-yellow outline with a 2px offset on
  `:focus-visible`, on paper and on film.
- **Empty and loading states:** one plain sentence saying what will be here and how to get it
  ("No reviewed rallies yet. Review a set to see its stats.").

## References

Screenshots the coach shared (2026-11-17) are in `docs/design-references/`. Each one lends
specific things; the rest of it is not a licence to copy.

- **Loire tennis coaching site** (`loire-tennis.webp`), editorial and minimal. Take: one
  large light headline over a very small body, lots of empty space, real motion-blurred
  action frames as the only imagery, hairline structure instead of boxes. Leave: its olive
  ground (our film stays neutral so the video's colours read true), the pill labels, the
  italic "Real progress", the decorative curves and the tiny grey-on-grey text. It is a
  marketing page, so most of it belongs to the later landing page, not the coach's app.
- **The Athletic, squad stats** (`the-athletic-squad-stats.webp`), the stat sheet done
  plainly. Take: separate tables per role with a small 600 heading ("Goalkeepers",
  "Outfield players"; for us "Setters", "Hitters", "Liberos"), the first column as name plus
  jersey number, short column abbreviations with a sort caret, small dense rows, numbers
  that line up. Leave: its zebra rows (we keep plain rows with the well on hover) and the
  photo per player (no headshots; a jersey number is enough).
- **adidas careers** (`adidas-careers.webp`), dark editorial. Take: a narrow text column
  next to large frames, and key numbers set inside a sentence in a stronger colour ("We hit
  0.245 on 955 attacks"), which is how a stat page can open: one sentence, the numbers in
  600 ink or court blue. Leave: the all-caps quote and the red outlined box.
- **VolleyMetrics setter tendencies** (`volleymetrics-setter-tendencies.png`), the closest
  product. Take: small multiples, the same court grid three times side by side (overall,
  first ball, transition), each cell carrying its count and its rates ("A att 419, assist
  42%, H% 0.263"), and the stat table under the maps. Leave: its shades of navy for value.
  We bin values into a few solid tokens (court for best, them for middle, miss for worst,
  paper for too few attempts) and say the bins in the legend line, since shades are on the
  don't-list.

### VolleyMetrics / Hudl report pages (from Hudl's support guides, 2026-11-17)

What the coach's current tool has, and how Toolbelt answers each. One panel holds all
reports as a tab row; match cards pick which games to combine; one Filters button narrows by
player, skill and situation; full screen adds a CSV export; table columns can be hidden and
reordered.

- **Box score:** team and player summary; any underlined stat plays the clips behind it.
  Ours: every count in a table links to Review filtered to those touches (clips are time
  ranges, so this costs nothing). Show it as a court-blue link, not an underline on every
  number.
- **Athlete stats:** tables grouped by skill (serves, receptions, attacks, sets, blocks,
  digs, free balls), each with "View all". Ours: the box score, with a heading per skill.
- **Attack tendencies:** each attack drawn as a line from contact to landing, coloured by
  result (kill, error, in play); views Lines, Distribution, Efficiency; filters for rotation,
  phase (first ball or transition) and result. Ours waits for depth (F2); until then show
  the along-net contact point only.
- **Rotations:** six cards, one per rotation, each a table of attacks by pass rating plus a
  court of attack lines. Ours after F7 (lineups).
- **Setter tendencies:** 3x3 zone grid with Efficiency / Frequency views (kill %, attack %,
  attempts per zone) and a per-setter table (set attempts, assists, setting efficiency).
  Ours: the set maps (E7) in this small-multiples form, split by in or out of system (E8).
- **Trends:** pick metrics, plot them across sets or matches. Ours: E8's trends, one line
  per metric, count shown on hover.
- **Goals:** a target per stat ("reception above 2.2"), pass or fail per match. Later.
- **3D:** attack contact heights. Ours after F2, as a side view (with the setting arcs, G3).

## Do's and don'ts

- Do read this file before changing any page, and reuse these tokens and components as they
  are. If a page needs a token that is not here, add it here first.
- Do keep copy short. One line of help, not a paragraph; no text that repeats the heading.
- Do keep charts as simple as the question: a court with dots, a table with numbers.
- Don't change: the product name Toolbelt, and the 0-4 pass scale's meaning.
- Don't use (the coach's list, 2026-11-17): tags and pills, italics, Instrument Serif,
  negative letter spacing, over-detailed UI, many dividers, colour shades and opacities,
  bordered containers, over-detailed diagrams, verbose or repeated text, breadcrumbs,
  section numbers, eyebrow labels, dithering, ASCII art, squircles, lots of monospace,
  all-caps subtitles, gradient backgrounds, periods in titles.
- Don't use the defaults this design refuses: Inter or system-ui as the face, Tailwind's
  default blue (#2563eb), near-black with one acid accent across the whole app, cream with
  terracotta, emoji as icons, a "→" on buttons, middle-dot "A · B · C" detail strings.
