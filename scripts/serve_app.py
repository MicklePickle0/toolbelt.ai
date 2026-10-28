"""Start the web app: videos + processing queue, rally review and the stat sheet.

Open http://localhost:8000 on this machine. From the laptop or phone, start it with
--host 0.0.0.0 and open http://<this PC's Tailscale name or IP>:8000 (Tailscale keeps it
off the open internet: there is no login).

The processing queue runs pipeline steps on this machine's GPU, one video at a time
(`app.steps` in config/pipeline.yaml). Without footage (the laptop) review and stats work
on the event DB in git; clips and processing need the videos.

Usage:
  python scripts/serve_app.py [--host 0.0.0.0] [--port 8000] [--no-worker]
"""

from __future__ import annotations

import argparse

import uvicorn

from setter_eyes.api.app import create_app


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 to reach it from other devices (over Tailscale)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-worker", action="store_true",
                    help="don't run processing jobs (review and stats only)")
    args = ap.parse_args()
    uvicorn.run(create_app(start_worker=not args.no_worker), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
