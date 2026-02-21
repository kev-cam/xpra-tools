"""
Xpra AI Control Plugin

Hooks into the Xpra server's frame and input pipelines to provide an AI
control plane over ZMQ. This is loaded as an Xpra server plugin.

Integration point: Xpra's plugin system loads modules from --plugin= or
from the xpra.server.mixins namespace. This module registers itself via
the standard init_server() entry point.
"""

import os
import time
import json
import logging
import threading
from typing import Dict, Optional, Any

import zmq

from .config import load_config, Config
from .protocol import (
    InputMode, EventType, ActionType, QueryType,
    FrameMessage, EventMessage, ControlRequest, ControlResponse,
)
from .framebuffer import FramebufferManager

logger = logging.getLogger(__name__)


class AIControlPlugin:
    """
    Xpra server plugin that bridges the X11 session to an AI agent.

    Responsibilities:
    - Tap damage/draw events to maintain per-window framebuffers
    - Publish frame snapshots over ZMQ PUB at a configurable rate
    - Publish window lifecycle events over ZMQ PUB
    - Accept control commands (actions, queries, mode changes) over ZMQ REQ/REP
    - Gate human input based on current mode
    - Handle kill switch for emergency mode revert
    """

    def __init__(self, server):
        self.server = server
        self.config = load_config()
        self.mode = InputMode(self.config.control.mode)
        self.framebuffers = FramebufferManager()

        # Window metadata cache
        self._window_meta: Dict[int, dict] = {}
        self._focused_wid: int = 0
        self._human_last_input: float = 0.0
        self._autonomous_start: float = 0.0

        # ZMQ context and sockets
        self._zmq_ctx = zmq.Context()
        self._setup_zmq()

        # Frame delivery thread
        self._running = True
        self._frame_thread = threading.Thread(
            target=self._frame_publisher_loop, daemon=True
        )
        self._control_thread = threading.Thread(
            target=self._control_handler_loop, daemon=True
        )
        self._frame_thread.start()
        self._control_thread.start()

        # Stats
        self._frames_sent = 0
        self._last_stats = time.time()

        logger.info("AI Control Plugin initialised, mode=%s", self.mode.value)

    def _setup_zmq(self):
        """Initialise ZMQ sockets."""
        cfg = self.config.zmq

        self._frame_pub = self._zmq_ctx.socket(zmq.PUB)
        self._frame_pub.setsockopt(zmq.SNDHWM, cfg.frame_hwm)
        self._frame_pub.bind(cfg.frame_endpoint)

        self._event_pub = self._zmq_ctx.socket(zmq.PUB)
        self._event_pub.setsockopt(zmq.SNDHWM, cfg.event_hwm)
        self._event_pub.bind(cfg.event_endpoint)

        self._control_rep = self._zmq_ctx.socket(zmq.REP)
        self._control_rep.bind(cfg.control_endpoint)

        logger.info("ZMQ bound: frames=%s events=%s control=%s",
                     cfg.frame_endpoint, cfg.event_endpoint, cfg.control_endpoint)

    # ---------------------------------------------------------------
    # Xpra hook points
    # ---------------------------------------------------------------

    def on_draw(self, wid: int, x: int, y: int, width: int, height: int,
                coding: str, data: bytes, client_options: dict):
        """
        Called from Xpra's damage/draw pipeline.

        Hook this into WindowSource or the server's _process_draw by
        monkey-patching or subclassing. See install_hooks().
        """
        self.framebuffers.update_region(wid, x, y, width, height, coding, data)

    def on_new_window(self, wid: int, window):
        """Called when a new X11 window is mapped."""
        x, y, w, h = self._get_window_geometry(window)
        title = self._get_window_title(window)
        wm_class = self._get_window_class(window)

        self.framebuffers.create_window(wid, w, h)
        self._window_meta[wid] = {
            "wid": wid,
            "title": title,
            "wm_class": wm_class,
            "x": x, "y": y, "width": w, "height": h,
            "pid": getattr(window, "_NET_WM_PID", 0),
        }

        self._publish_event(EventType.WINDOW_CREATE, wid, self._window_meta[wid])
        logger.debug("Window created: wid=%d title=%s", wid, title)

    def on_window_destroyed(self, wid: int):
        """Called when an X11 window is unmapped/destroyed."""
        self.framebuffers.destroy_window(wid)
        self._window_meta.pop(wid, None)
        self._publish_event(EventType.WINDOW_DESTROY, wid)
        logger.debug("Window destroyed: wid=%d", wid)

    def on_window_resize(self, wid: int, width: int, height: int):
        """Called on window geometry change."""
        self.framebuffers.resize_window(wid, width, height)
        meta = self._window_meta.get(wid, {})
        meta.update(width=width, height=height)
        self._publish_event(EventType.WINDOW_RESIZE, wid,
                            {"width": width, "height": height})

    def on_focus_change(self, wid: int):
        """Called when focus changes to a different window."""
        self._focused_wid = wid
        self._publish_event(EventType.WINDOW_FOCUS, wid)

    def on_title_change(self, wid: int, title: str):
        """Called when a window title changes."""
        meta = self._window_meta.get(wid, {})
        meta["title"] = title
        self._publish_event(EventType.WINDOW_TITLE, wid, {"title": title})

    # ---------------------------------------------------------------
    # Input gating
    # ---------------------------------------------------------------

    def filter_input(self, input_type: str, packet: list) -> bool:
        """
        Decide whether to pass through a human input event.

        Returns True to allow, False to suppress.
        Called from the packet handler for pointer/key events.
        """
        # Kill switch always passes through
        if self._is_kill_switch(input_type, packet):
            self._trigger_kill_switch()
            return True

        if self.mode == InputMode.OBSERVER:
            return True  # Human drives

        if self.mode == InputMode.AUTONOMOUS:
            return False  # AI drives, suppress human

        if self.mode == InputMode.SUPERVISED:
            return True  # Human can override

        if self.mode == InputMode.COLLABORATIVE:
            self._human_last_input = time.time()
            return True  # Both can act

        return True

    def _is_kill_switch(self, input_type: str, packet: list) -> bool:
        """Check if this input event matches the kill switch combo."""
        if input_type != "key-action":
            return False
        try:
            # Xpra key-action packet: [type, wid, keyname, pressed, modifiers, ...]
            keyname = packet[2] if len(packet) > 2 else ""
            modifiers = packet[4] if len(packet) > 4 else []
            ks = self.config.control.kill_switch.lower()
            parts = ks.split("+")
            key_part = parts[-1]
            mod_parts = set(parts[:-1])
            return (keyname.lower() == key_part and
                    mod_parts.issubset(set(m.lower() for m in modifiers)))
        except (IndexError, AttributeError):
            return False

    def _trigger_kill_switch(self):
        """Emergency revert to observer mode."""
        old_mode = self.mode
        self.mode = InputMode.OBSERVER
        self._publish_event(EventType.KILL_SWITCH, 0,
                            {"old_mode": old_mode.value, "new_mode": "observer"})
        logger.warning("KILL SWITCH activated: %s → observer", old_mode.value)

    def ai_can_act(self) -> bool:
        """Check if the AI is currently permitted to inject input."""
        if self.mode in (InputMode.AUTONOMOUS, InputMode.SUPERVISED):
            return True
        if self.mode == InputMode.COLLABORATIVE:
            elapsed = (time.time() - self._human_last_input) * 1000
            return elapsed > self.config.control.human_priority_ms
        return False

    # ---------------------------------------------------------------
    # Frame publisher thread
    # ---------------------------------------------------------------

    def _frame_publisher_loop(self):
        """Background thread that publishes frame snapshots at configured FPS."""
        cfg = self.config.frame_capture
        interval = 1.0 / max(cfg.fps, 0.1)

        while self._running:
            try:
                snapshots = self.framebuffers.get_dirty_snapshots(
                    fmt=cfg.format,
                    quality=cfg.quality,
                    scale=cfg.scale,
                    max_dim=cfg.max_dimension,
                    min_interval=interval if cfg.delta_only else 0,
                )

                for wid, data, seq in snapshots:
                    meta = self._window_meta.get(wid, {})
                    msg = FrameMessage(
                        wid=wid,
                        x=0, y=0,
                        width=meta.get("width", 0),
                        height=meta.get("height", 0),
                        format=cfg.format,
                        data=data,
                        window_title=meta.get("title", ""),
                        window_class=meta.get("wm_class", ""),
                        is_focused=(wid == self._focused_wid),
                        sequence=seq,
                    )
                    self._frame_pub.send(msg.serialise(), zmq.NOBLOCK)
                    self._frames_sent += 1

                # Stats logging
                now = time.time()
                if now - self._last_stats > self.config.logging.stats_interval:
                    logger.info("Frames sent: %d (%.1f/s)",
                                self._frames_sent,
                                self._frames_sent / max(now - self._last_stats, 1))
                    self._frames_sent = 0
                    self._last_stats = now

            except zmq.Again:
                pass  # HWM reached, drop frame
            except Exception as e:
                logger.error("Frame publisher error: %s", e)

            time.sleep(interval)

    # ---------------------------------------------------------------
    # Control handler thread (REQ/REP)
    # ---------------------------------------------------------------

    def _control_handler_loop(self):
        """Background thread handling control requests from the agent."""
        poller = zmq.Poller()
        poller.register(self._control_rep, zmq.POLLIN)

        while self._running:
            try:
                events = dict(poller.poll(timeout=500))
                if self._control_rep not in events:
                    continue

                raw = self._control_rep.recv()
                request = ControlRequest.deserialise(raw)
                response = self._handle_request(request)
                self._control_rep.send(response.serialise())

            except Exception as e:
                logger.error("Control handler error: %s", e)
                # Must send a reply to maintain REQ/REP state
                try:
                    err = ControlResponse(success=False, error=str(e))
                    self._control_rep.send(err.serialise())
                except Exception:
                    pass

    def _handle_request(self, req: ControlRequest) -> ControlResponse:
        """Dispatch a control request."""
        try:
            if req.request_type == "action":
                return self._handle_action(req)
            elif req.request_type == "query":
                return self._handle_query(req)
            elif req.request_type == "mode":
                return self._handle_mode_change(req)
            else:
                return ControlResponse(
                    success=False, request_id=req.request_id,
                    error=f"Unknown request type: {req.request_type}"
                )
        except Exception as e:
            return ControlResponse(
                success=False, request_id=req.request_id, error=str(e)
            )

    def _handle_action(self, req: ControlRequest) -> ControlResponse:
        """Execute an AI-requested input action."""
        if not self.ai_can_act():
            return ControlResponse(
                success=False, request_id=req.request_id,
                error=f"AI cannot act in mode: {self.mode.value}"
            )

        action = req.payload.get("action", "")
        p = req.payload

        try:
            if action == ActionType.CLICK.value:
                self._inject_click(p.get("wid", 0), p["x"], p["y"],
                                   p.get("button", 1))
            elif action == ActionType.DOUBLE_CLICK.value:
                self._inject_click(p.get("wid", 0), p["x"], p["y"],
                                   p.get("button", 1), count=2)
            elif action == ActionType.RIGHT_CLICK.value:
                self._inject_click(p.get("wid", 0), p["x"], p["y"], button=3)
            elif action == ActionType.MOUSE_MOVE.value:
                self._inject_mouse_move(p["x"], p["y"])
            elif action == ActionType.SCROLL.value:
                self._inject_scroll(p["x"], p["y"], p.get("dx", 0), p.get("dy", -3))
            elif action == ActionType.KEY_PRESS.value:
                self._inject_key(p["key"], press=True, release=True)
            elif action == ActionType.KEY_DOWN.value:
                self._inject_key(p["key"], press=True, release=False)
            elif action == ActionType.KEY_UP.value:
                self._inject_key(p["key"], press=False, release=True)
            elif action == ActionType.TYPE_TEXT.value:
                self._inject_text(p["text"])
            elif action == ActionType.SET_CLIPBOARD.value:
                self._set_clipboard(p["text"])
            else:
                return ControlResponse(
                    success=False, request_id=req.request_id,
                    error=f"Unknown action: {action}"
                )

            return ControlResponse(success=True, request_id=req.request_id)

        except Exception as e:
            return ControlResponse(
                success=False, request_id=req.request_id, error=str(e)
            )

    def _handle_query(self, req: ControlRequest) -> ControlResponse:
        """Handle a state query from the agent."""
        query = req.payload.get("query", "")

        if query == QueryType.WINDOW_LIST.value:
            windows = list(self._window_meta.values())
            return ControlResponse(
                success=True, request_id=req.request_id, data=windows
            )

        elif query == QueryType.WINDOW_INFO.value:
            wid = req.payload.get("wid", 0)
            meta = self._window_meta.get(wid)
            return ControlResponse(
                success=meta is not None, request_id=req.request_id,
                data=meta, error="" if meta else f"No window {wid}"
            )

        elif query == QueryType.FOCUSED_WINDOW.value:
            return ControlResponse(
                success=True, request_id=req.request_id,
                data={"wid": self._focused_wid,
                      "meta": self._window_meta.get(self._focused_wid)}
            )

        elif query == QueryType.SCREENSHOT.value:
            wid = req.payload.get("wid", self._focused_wid)
            fb = self.framebuffers.get(wid)
            if fb:
                data = fb.snapshot(
                    self.config.frame_capture.format,
                    self.config.frame_capture.quality,
                    self.config.frame_capture.scale,
                    self.config.frame_capture.max_dimension,
                )
                # Force snapshot even if not dirty
                if data is None:
                    data = fb.snapshot.__wrapped__(fb) if hasattr(fb.snapshot, '__wrapped__') else b""
                return ControlResponse(
                    success=True, request_id=req.request_id, data=data
                )
            return ControlResponse(
                success=False, request_id=req.request_id,
                error=f"No framebuffer for wid {wid}"
            )

        elif query == QueryType.CURRENT_MODE.value:
            return ControlResponse(
                success=True, request_id=req.request_id,
                data={"mode": self.mode.value}
            )

        elif query == QueryType.CLIPBOARD.value:
            text = self._get_clipboard()
            return ControlResponse(
                success=True, request_id=req.request_id, data={"text": text}
            )

        return ControlResponse(
            success=False, request_id=req.request_id,
            error=f"Unknown query: {query}"
        )

    def _handle_mode_change(self, req: ControlRequest) -> ControlResponse:
        """Handle an input mode change request."""
        new_mode_str = req.payload.get("mode", "")
        try:
            new_mode = InputMode(new_mode_str)
        except ValueError:
            return ControlResponse(
                success=False, request_id=req.request_id,
                error=f"Invalid mode: {new_mode_str}"
            )

        old_mode = self.mode
        self.mode = new_mode

        if new_mode == InputMode.AUTONOMOUS:
            self._autonomous_start = time.time()

        self._publish_event(EventType.MODE_CHANGE, 0, {
            "old_mode": old_mode.value, "new_mode": new_mode.value
        })

        logger.info("Mode changed: %s → %s", old_mode.value, new_mode.value)
        return ControlResponse(
            success=True, request_id=req.request_id,
            data={"mode": new_mode.value}
        )

    # ---------------------------------------------------------------
    # Input injection (XTest)
    # ---------------------------------------------------------------

    def _inject_click(self, wid: int, x: int, y: int, button: int = 1,
                      count: int = 1):
        """Inject a mouse click via XTest."""
        try:
            from xpra.x11.bindings.xtest import XTestBindings
            xtest = XTestBindings()
            for _ in range(count):
                xtest.fake_button(button, True, x, y)
                time.sleep(0.02)
                xtest.fake_button(button, False, x, y)
                time.sleep(0.05)
        except ImportError:
            # Fallback: use xdotool
            import subprocess
            for _ in range(count):
                subprocess.run(
                    ["xdotool", "mousemove", str(x), str(y),
                     "click", str(button)],
                    timeout=5
                )

    def _inject_mouse_move(self, x: int, y: int):
        """Inject a mouse move."""
        try:
            from xpra.x11.bindings.xtest import XTestBindings
            xtest = XTestBindings()
            xtest.fake_motion(x, y)
        except ImportError:
            import subprocess
            subprocess.run(["xdotool", "mousemove", str(x), str(y)], timeout=5)

    def _inject_scroll(self, x: int, y: int, dx: int = 0, dy: int = -3):
        """Inject scroll events."""
        try:
            from xpra.x11.bindings.xtest import XTestBindings
            xtest = XTestBindings()
            xtest.fake_motion(x, y)
            # Button 4 = scroll up, 5 = scroll down
            button = 4 if dy < 0 else 5
            for _ in range(abs(dy)):
                xtest.fake_button(button, True, x, y)
                xtest.fake_button(button, False, x, y)
        except ImportError:
            import subprocess
            subprocess.run(["xdotool", "mousemove", str(x), str(y)], timeout=5)
            if dy < 0:
                subprocess.run(["xdotool", "click", "4"] * abs(dy), timeout=5)
            else:
                subprocess.run(["xdotool", "click", "5"] * abs(dy), timeout=5)

    def _inject_key(self, key: str, press: bool = True, release: bool = True):
        """Inject a key press/release via XTest."""
        try:
            from xpra.x11.bindings.xtest import XTestBindings
            from xpra.x11.bindings.keyboard import X11KeyboardBindings
            xtest = XTestBindings()
            kb = X11KeyboardBindings()
            keycode = kb.parse_keycode(key)
            if press:
                xtest.fake_key(keycode, True)
            if release:
                time.sleep(0.02)
                xtest.fake_key(keycode, False)
        except ImportError:
            import subprocess
            if press and release:
                subprocess.run(["xdotool", "key", key], timeout=5)
            elif press:
                subprocess.run(["xdotool", "keydown", key], timeout=5)
            elif release:
                subprocess.run(["xdotool", "keyup", key], timeout=5)

    def _inject_text(self, text: str):
        """Type a string of text."""
        try:
            import subprocess
            subprocess.run(["xdotool", "type", "--clearmodifiers", text], timeout=10)
        except Exception as e:
            logger.error("Failed to type text: %s", e)

    def _set_clipboard(self, text: str):
        """Set the X11 clipboard."""
        try:
            import subprocess
            proc = subprocess.Popen(
                ["xclip", "-selection", "clipboard"],
                stdin=subprocess.PIPE, timeout=5
            )
            proc.communicate(input=text.encode())
        except Exception as e:
            logger.error("Failed to set clipboard: %s", e)

    def _get_clipboard(self) -> str:
        """Get the X11 clipboard contents."""
        try:
            import subprocess
            result = subprocess.run(
                ["xclip", "-selection", "clipboard", "-o"],
                capture_output=True, text=True, timeout=5
            )
            return result.stdout
        except Exception:
            return ""

    # ---------------------------------------------------------------
    # Event publishing
    # ---------------------------------------------------------------

    def _publish_event(self, event_type: EventType, wid: int = 0,
                       data: dict = None):
        """Publish a window/system event."""
        msg = EventMessage(
            event_type=event_type.value,
            wid=wid,
            data=data or {},
        )
        try:
            self._event_pub.send(msg.serialise(), zmq.NOBLOCK)
        except zmq.Again:
            pass

    # ---------------------------------------------------------------
    # Xpra window helpers
    # ---------------------------------------------------------------

    @staticmethod
    def _get_window_geometry(window) -> tuple:
        """Extract geometry from an Xpra window object."""
        try:
            return window.get_geometry()[:4]
        except Exception:
            return (0, 0, 800, 600)

    @staticmethod
    def _get_window_title(window) -> str:
        try:
            return str(window.get_property("title") or "")
        except Exception:
            return ""

    @staticmethod
    def _get_window_class(window) -> str:
        try:
            cls = window.get_property("class-instance")
            return str(cls) if cls else ""
        except Exception:
            return ""

    # ---------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------

    def shutdown(self):
        """Clean up resources."""
        self._running = False
        self._frame_thread.join(timeout=2)
        self._control_thread.join(timeout=2)
        self._frame_pub.close()
        self._event_pub.close()
        self._control_rep.close()
        self._zmq_ctx.term()
        logger.info("AI Control Plugin shut down")


# ---------------------------------------------------------------
# Xpra plugin entry point
# ---------------------------------------------------------------

def install_hooks(server, plugin: AIControlPlugin):
    """
    Monkey-patch Xpra server to route events to the AI plugin.

    This is necessarily version-dependent. Tested against Xpra 4.x/5.x.
    The hook points are:

    1. _process_draw / do_process_draw — frame data
    2. _add_new_window / _add_new_or_window — window creation
    3. _remove_window — window destruction
    4. _process_focus — focus changes
    5. Packet handlers for pointer/key events — input gating
    """

    # --- Hook draw events ---
    if hasattr(server, '_process_draw'):
        original_draw = server._process_draw

        def hooked_draw(proto, packet):
            try:
                wid = packet[1]
                x, y, w, h = packet[2], packet[3], packet[4], packet[5]
                coding = packet[6]
                data = packet[7]
                plugin.on_draw(wid, x, y, w, h, coding, data)
            except Exception as e:
                logger.debug("Draw hook error: %s", e)
            return original_draw(proto, packet)

        server._process_draw = hooked_draw

    # --- Hook window creation ---
    for method_name in ('_add_new_window_common', '_add_new_window', 'add_window'):
        if hasattr(server, method_name):
            original = getattr(server, method_name)

            def make_hook(orig):
                def hooked(*args, **kwargs):
                    result = orig(*args, **kwargs)
                    try:
                        # Try to extract wid and window from args/result
                        if args:
                            window = args[0] if len(args) == 1 else args[-1]
                            wid = getattr(window, 'wid',
                                          getattr(result, 'wid', 0)) if window else 0
                            if wid:
                                plugin.on_new_window(wid, window)
                    except Exception as e:
                        logger.debug("Window creation hook error: %s", e)
                    return result
                return hooked

            setattr(server, method_name, make_hook(original))
            break

    # --- Hook input filtering ---
    for ptype in ("pointer-position", "pointer-button", "key-action"):
        if hasattr(server, '_packet_handlers') and ptype in server._packet_handlers:
            original_handler = server._packet_handlers[ptype]

            def make_input_hook(orig, pt):
                def hooked(proto, packet):
                    if plugin.filter_input(pt, packet):
                        return orig(proto, packet)
                    # Suppressed by AI control
                return hooked

            server._packet_handlers[ptype] = make_input_hook(original_handler, ptype)

    logger.info("AI Control hooks installed on Xpra server")


_plugin_instance: Optional[AIControlPlugin] = None


def init_server(server) -> AIControlPlugin:
    """
    Xpra plugin entry point. Called by the Xpra server during startup.

    Usage:
        xpra start :100 --plugin=xpra_ai_control.plugin
    """
    global _plugin_instance
    _plugin_instance = AIControlPlugin(server)
    install_hooks(server, _plugin_instance)
    return _plugin_instance


def get_plugin() -> Optional[AIControlPlugin]:
    """Get the current plugin instance (for external access)."""
    return _plugin_instance
