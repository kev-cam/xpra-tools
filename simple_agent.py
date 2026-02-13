#!/usr/bin/env python3
"""
Minimal example agent.

Connects to the Xpra AI Control Plugin, takes a screenshot of each window,
saves them to disk, and demonstrates basic click/type actions.

Usage:
    python examples/simple_agent.py [--config config.yaml] [--autonomous]
"""

import sys
import time
import logging
import argparse
from pathlib import Path

# Add parent dir to path for local development
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xpra_ai_control.agent import AIAgent
from xpra_ai_control.protocol import FrameMessage, EventMessage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--autonomous", action="store_true",
                        help="Switch to autonomous mode and perform actions")
    parser.add_argument("--screenshot-dir", default="./screenshots")
    args = parser.parse_args()

    agent = AIAgent(args.config)
    agent.connect()

    # Wait for initial frames
    log.info("Waiting for frames...")
    time.sleep(2)

    # Query windows
    windows = agent.query_windows()
    log.info("Found %d windows", len(windows))

    # Save screenshots
    out_dir = Path(args.screenshot_dir)
    out_dir.mkdir(exist_ok=True)

    for w in windows:
        wid = w["wid"]
        frame = agent.get_frame(wid=wid)
        if frame:
            path = out_dir / f"window_{wid}_{w.get('title', 'untitled')}.png"
            frame.save(str(path))
            log.info("Saved screenshot: %s (%dx%d)", path, frame.width, frame.height)
        else:
            log.warning("No frame available for wid=%d", wid)

    # Demonstrate actions (only in autonomous mode)
    if args.autonomous:
        log.info("Switching to autonomous mode...")
        agent.set_mode("autonomous")

        focused = agent.query_focused()
        if focused and focused.get("wid"):
            wid = focused["wid"]
            meta = focused.get("meta", {})
            log.info("Focused window: wid=%d title=%s", wid, meta.get("title", "?"))

            # Example: click in the center of the focused window
            cx = meta.get("width", 800) // 2
            cy = meta.get("height", 600) // 2
            log.info("Clicking at center (%d, %d)...", cx, cy)
            agent.click(x=cx, y=cy, wid=wid)
            time.sleep(0.5)

            # Example: type some text
            log.info("Typing text...")
            agent.type_text("Hello from the AI agent!")
            agent.key_press("Return")

        # Revert to observer
        log.info("Reverting to observer mode...")
        agent.set_mode("observer")

    agent.disconnect()
    log.info("Done.")


if __name__ == "__main__":
    main()
