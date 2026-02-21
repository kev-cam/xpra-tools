"""
AI Agent Process

Standalone process that connects to the Xpra AI Control Plugin over ZMQ.
Receives frames, sends actions, and implements the observation → decision → action loop.

This module provides both:
1. AIAgent class — a Python API for building custom agents
2. A __main__ entry point with a basic vision-action loop
"""

import io
import time
import logging
import argparse
import threading
from typing import Optional, List, Dict, Any, Callable

import zmq
from PIL import Image

from .config import load_config, Config
from .protocol import (
    InputMode, ActionType, QueryType,
    FrameMessage, EventMessage, ControlRequest, ControlResponse,
    make_action, make_query, make_mode_change,
)

logger = logging.getLogger(__name__)


class AIAgent:
    """
    Client-side agent that communicates with the Xpra AI Control Plugin.

    Provides a clean API for:
    - Subscribing to frames and events
    - Querying window state
    - Performing input actions
    - Managing control modes

    Usage:
        agent = AIAgent("config.yaml")
        agent.connect()

        # Register callbacks
        agent.on_frame = my_frame_handler
        agent.on_event = my_event_handler

        # Or use synchronous API
        windows = agent.query_windows()
        frame = agent.get_frame(wid=1)
        agent.click(x=100, y=200)
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        self._zmq_ctx = zmq.Context()
        self._connected = False
        self._running = False

        # Callbacks
        self.on_frame: Optional[Callable[[FrameMessage], None]] = None
        self.on_event: Optional[Callable[[EventMessage], None]] = None

        # Latest frame per window (for synchronous access)
        self._latest_frames: Dict[int, FrameMessage] = {}
        self._frame_lock = threading.Lock()

        # Sockets (initialised on connect)
        self._frame_sub = None
        self._event_sub = None
        self._control_req = None

        # Background threads
        self._frame_thread = None
        self._event_thread = None

    def connect(self):
        """Connect to the AI Control Plugin's ZMQ endpoints."""
        cfg = self.config.zmq

        self._frame_sub = self._zmq_ctx.socket(zmq.SUB)
        self._frame_sub.setsockopt(zmq.RCVHWM, cfg.frame_hwm)
        self._frame_sub.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all
        self._frame_sub.connect(cfg.frame_endpoint)

        self._event_sub = self._zmq_ctx.socket(zmq.SUB)
        self._event_sub.setsockopt(zmq.RCVHWM, cfg.event_hwm)
        self._event_sub.setsockopt(zmq.SUBSCRIBE, b"")
        self._event_sub.connect(cfg.event_endpoint)

        self._control_req = self._zmq_ctx.socket(zmq.REQ)
        self._control_req.setsockopt(zmq.RCVTIMEO, 5000)  # 5s timeout
        self._control_req.connect(cfg.control_endpoint)

        self._connected = True
        self._running = True

        # Start receiver threads
        self._frame_thread = threading.Thread(
            target=self._frame_receiver_loop, daemon=True
        )
        self._event_thread = threading.Thread(
            target=self._event_receiver_loop, daemon=True
        )
        self._frame_thread.start()
        self._event_thread.start()

        logger.info("Agent connected to control plugin")

    def disconnect(self):
        """Disconnect and clean up."""
        self._running = False
        if self._frame_thread:
            self._frame_thread.join(timeout=2)
        if self._event_thread:
            self._event_thread.join(timeout=2)
        for sock in (self._frame_sub, self._event_sub, self._control_req):
            if sock:
                sock.close()
        self._zmq_ctx.term()
        self._connected = False
        logger.info("Agent disconnected")

    # ---------------------------------------------------------------
    # Receiver loops
    # ---------------------------------------------------------------

    def _frame_receiver_loop(self):
        """Background thread receiving frame snapshots."""
        poller = zmq.Poller()
        poller.register(self._frame_sub, zmq.POLLIN)

        while self._running:
            try:
                events = dict(poller.poll(timeout=500))
                if self._frame_sub not in events:
                    continue

                raw = self._frame_sub.recv()
                msg = FrameMessage.deserialise(raw)

                # Cache latest frame
                with self._frame_lock:
                    self._latest_frames[msg.wid] = msg

                # Invoke callback
                if self.on_frame:
                    try:
                        self.on_frame(msg)
                    except Exception as e:
                        logger.error("Frame callback error: %s", e)

            except Exception as e:
                if self._running:
                    logger.error("Frame receiver error: %s", e)

    def _event_receiver_loop(self):
        """Background thread receiving window events."""
        poller = zmq.Poller()
        poller.register(self._event_sub, zmq.POLLIN)

        while self._running:
            try:
                events = dict(poller.poll(timeout=500))
                if self._event_sub not in events:
                    continue

                raw = self._event_sub.recv()
                msg = EventMessage.deserialise(raw)

                if self.on_event:
                    try:
                        self.on_event(msg)
                    except Exception as e:
                        logger.error("Event callback error: %s", e)

            except Exception as e:
                if self._running:
                    logger.error("Event receiver error: %s", e)

    # ---------------------------------------------------------------
    # Control API (synchronous, REQ/REP)
    # ---------------------------------------------------------------

    def _send_request(self, req: ControlRequest) -> ControlResponse:
        """Send a control request and wait for response."""
        if not self._connected:
            raise RuntimeError("Agent not connected")

        self._control_req.send(req.serialise())
        raw = self._control_req.recv()
        return ControlResponse.deserialise(raw)

    # --- Actions ---

    def click(self, x: int, y: int, wid: int = 0, button: int = 1) -> bool:
        """Click at coordinates."""
        req = make_action(ActionType.CLICK, wid=wid, x=x, y=y, button=button)
        resp = self._send_request(req)
        return resp.success

    def double_click(self, x: int, y: int, wid: int = 0) -> bool:
        """Double-click at coordinates."""
        req = make_action(ActionType.DOUBLE_CLICK, wid=wid, x=x, y=y)
        resp = self._send_request(req)
        return resp.success

    def right_click(self, x: int, y: int, wid: int = 0) -> bool:
        """Right-click at coordinates."""
        req = make_action(ActionType.RIGHT_CLICK, wid=wid, x=x, y=y)
        resp = self._send_request(req)
        return resp.success

    def mouse_move(self, x: int, y: int) -> bool:
        """Move mouse to coordinates."""
        req = make_action(ActionType.MOUSE_MOVE, x=x, y=y)
        resp = self._send_request(req)
        return resp.success

    def scroll(self, x: int, y: int, dy: int = -3) -> bool:
        """Scroll at coordinates. Negative dy = scroll up."""
        req = make_action(ActionType.SCROLL, x=x, y=y, dy=dy)
        resp = self._send_request(req)
        return resp.success

    def key_press(self, key: str) -> bool:
        """Press and release a key."""
        req = make_action(ActionType.KEY_PRESS, key=key)
        resp = self._send_request(req)
        return resp.success

    def key_down(self, key: str) -> bool:
        """Press a key (hold)."""
        req = make_action(ActionType.KEY_DOWN, key=key)
        resp = self._send_request(req)
        return resp.success

    def key_up(self, key: str) -> bool:
        """Release a key."""
        req = make_action(ActionType.KEY_UP, key=key)
        resp = self._send_request(req)
        return resp.success

    def type_text(self, text: str) -> bool:
        """Type a string of text."""
        req = make_action(ActionType.TYPE_TEXT, text=text)
        resp = self._send_request(req)
        return resp.success

    def set_clipboard(self, text: str) -> bool:
        """Set clipboard contents."""
        req = make_action(ActionType.SET_CLIPBOARD, text=text)
        resp = self._send_request(req)
        return resp.success

    # --- Queries ---

    def query_windows(self) -> List[dict]:
        """Get list of all tracked windows."""
        req = make_query(QueryType.WINDOW_LIST)
        resp = self._send_request(req)
        return resp.data if resp.success else []

    def query_window(self, wid: int) -> Optional[dict]:
        """Get metadata for a specific window."""
        req = make_query(QueryType.WINDOW_INFO, wid=wid)
        resp = self._send_request(req)
        return resp.data if resp.success else None

    def query_focused(self) -> Optional[dict]:
        """Get the currently focused window."""
        req = make_query(QueryType.FOCUSED_WINDOW)
        resp = self._send_request(req)
        return resp.data if resp.success else None

    def query_clipboard(self) -> str:
        """Get clipboard contents."""
        req = make_query(QueryType.CLIPBOARD)
        resp = self._send_request(req)
        return resp.data.get("text", "") if resp.success else ""

    def get_screenshot(self, wid: Optional[int] = None) -> Optional[bytes]:
        """Request a screenshot of a window (or focused window)."""
        kwargs = {"wid": wid} if wid is not None else {}
        req = make_query(QueryType.SCREENSHOT, **kwargs)
        resp = self._send_request(req)
        return resp.data if resp.success else None

    # --- Mode ---

    def set_mode(self, mode: str) -> bool:
        """Change the input gating mode."""
        req = make_mode_change(InputMode(mode))
        resp = self._send_request(req)
        return resp.success

    def get_mode(self) -> str:
        """Get current input mode."""
        req = make_query(QueryType.CURRENT_MODE)
        resp = self._send_request(req)
        return resp.data.get("mode", "unknown") if resp.success else "unknown"

    # ---------------------------------------------------------------
    # Frame access helpers
    # ---------------------------------------------------------------

    def get_frame(self, wid: Optional[int] = None) -> Optional[Image.Image]:
        """
        Get the latest cached frame as a PIL Image.

        If wid is None, returns the frame for the most recently updated window.
        """
        with self._frame_lock:
            if wid is not None:
                msg = self._latest_frames.get(wid)
            else:
                # Most recent frame across all windows
                if not self._latest_frames:
                    return None
                msg = max(self._latest_frames.values(),
                          key=lambda m: m.timestamp)

        if msg is None:
            return None

        try:
            if msg.format == "raw":
                return Image.frombytes("RGB", (msg.width, msg.height), msg.data)
            else:
                return Image.open(io.BytesIO(msg.data))
        except Exception as e:
            logger.error("Failed to decode frame: %s", e)
            return None

    def wait_for_frame(self, wid: Optional[int] = None,
                       timeout: float = 5.0) -> Optional[Image.Image]:
        """Wait for a fresh frame to arrive."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            frame = self.get_frame(wid)
            if frame is not None:
                return frame
            time.sleep(0.1)
        return None


# ---------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AI Agent for Xpra Control")
    parser.add_argument("--config", default=None, help="Config file path")
    parser.add_argument("--mode", default=None,
                        help="Override initial mode")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    agent = AIAgent(args.config)
    agent.connect()

    if args.mode:
        agent.set_mode(args.mode)

    # Register simple logging callbacks
    def on_frame(msg: FrameMessage):
        logger.info("Frame: wid=%d %dx%d %s seq=%d title=%s",
                     msg.wid, msg.width, msg.height, msg.format,
                     msg.sequence, msg.window_title)

    def on_event(msg: EventMessage):
        logger.info("Event: %s wid=%d data=%s",
                     msg.event_type, msg.wid, msg.data)

    agent.on_frame = on_frame
    agent.on_event = on_event

    print(f"AI Agent running, mode={agent.get_mode()}")
    print("Press Ctrl+C to exit")
    print()

    # Query initial state
    windows = agent.query_windows()
    print(f"Active windows: {len(windows)}")
    for w in windows:
        print(f"  wid={w['wid']} title={w.get('title', '?')} "
              f"{w.get('width', 0)}x{w.get('height', 0)}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        agent.disconnect()


if __name__ == "__main__":
    main()
