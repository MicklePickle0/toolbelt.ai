"""A video player for the review UI that shows time to the millisecond and steps by frame.

The browser's own player shows whole seconds only. This one is plain HTML5 video plus a
readout (time in the clip, time in the whole video, frame number) and buttons: +-1
frame, +-0.1 s, play / pause, slow motion. Keys while it has focus: , and . step one
frame, space plays / pauses. The clip is embedded as a data URI (rally clips are a few
MB), so nothing has to be served separately.
"""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

_HTML = """
<div style="font-family: system-ui, sans-serif; color: #eee; background: #111; padding: 6px;
            border-radius: 6px;">
  <video id="v" src="data:video/mp4;base64,__DATA__" style="width: 100%; display: block;"
         preload="auto" playsinline></video>
  <div id="read" style="font: 600 18px ui-monospace, Consolas, monospace; margin: 6px 2px;">
  </div>
  <div style="display: flex; gap: 6px; flex-wrap: wrap;">
    <button onclick="step(-0.1)">-0.1 s</button>
    <button onclick="frame(-1)">&#9664; frame</button>
    <button id="pp" onclick="toggle()">Play</button>
    <button onclick="frame(1)">frame &#9654;</button>
    <button onclick="step(0.1)">+0.1 s</button>
    <button id="slow" onclick="slow()">Slow x0.25: off</button>
  </div>
  <input id="bar" type="range" min="0" max="1000" value="0" style="width: 100%;">
  <div style="font-size: 12px; color: #999;">Keys (click the video first): , and . one
    frame, space play / pause.</div>
</div>
<style>button { background: #333; color: #eee; border: 1px solid #555; border-radius: 4px;
                padding: 4px 10px; cursor: pointer; } button:hover { background: #444; }</style>
<script>
const v = document.getElementById("v"), read = document.getElementById("read"),
      bar = document.getElementById("bar"), T0 = __T0__, FPS = __FPS__;
function clock(t) { const m = Math.floor(t / 60), s = t - 60 * m;
  return m + ":" + s.toFixed(3).padStart(6, "0"); }
function show() {
  const t = v.currentTime;
  read.textContent = "clip " + t.toFixed(3) + " s   |   video " + clock(T0 + t) +
                     "   |   frame " + Math.round((T0 + t) * FPS);
  if (v.duration) bar.value = 1000 * t / v.duration;
}
function frame(n) { v.pause(); v.currentTime = Math.max(0, v.currentTime + n / FPS); }
function step(s) { v.pause(); v.currentTime = Math.max(0, v.currentTime + s); }
function toggle() { v.paused ? v.play() : v.pause(); }
function slow() { v.playbackRate = v.playbackRate === 1 ? 0.25 : 1;
  document.getElementById("slow").textContent =
    "Slow x0.25: " + (v.playbackRate === 1 ? "off" : "on"); }
bar.oninput = () => { if (v.duration) v.currentTime = v.duration * bar.value / 1000; };
v.onplay = () => document.getElementById("pp").textContent = "Pause";
v.onpause = () => document.getElementById("pp").textContent = "Play";
v.onclick = toggle;
["timeupdate", "seeked", "loadedmetadata"].forEach(e => v.addEventListener(e, show));
(function tick() { if (!v.paused) show(); requestAnimationFrame(tick); })();
document.addEventListener("keydown", e => {
  if (e.key === ",") frame(-1); else if (e.key === ".") frame(1);
  else if (e.key === " ") { e.preventDefault(); toggle(); } });
</script>
"""


def player(clip: Path, t0: float, fps: float, height: int = 640) -> None:
    """Show `clip` (starting at t0 seconds into the video) with a millisecond readout."""
    data = base64.b64encode(clip.read_bytes()).decode("ascii")
    html = _HTML.replace("__DATA__", data).replace("__T0__", f"{t0:.6f}") \
        .replace("__FPS__", f"{fps:.6f}")
    st.iframe(html, height=height)          # our own page: no outside content
