"""Tests for protocol messages and framebuffer management."""

import io
import pytest
from PIL import Image

from xpra_ai_control.protocol import (
    FrameMessage, EventMessage, ControlRequest, ControlResponse,
    InputMode, ActionType, QueryType, EventType,
    make_action, make_query, make_mode_change,
)
from xpra_ai_control.framebuffer import WindowFramebuffer, FramebufferManager
from xpra_ai_control.config import load_config, Config


class TestProtocol:
    def test_frame_message_roundtrip(self):
        msg = FrameMessage(
            wid=1, x=0, y=0, width=800, height=600,
            format="jpeg", data=b"fake-jpeg-data",
            window_title="xterm", window_class="XTerm",
            is_focused=True, sequence=42,
        )
        raw = msg.serialise()
        restored = FrameMessage.deserialise(raw)
        assert restored.wid == 1
        assert restored.width == 800
        assert restored.data == b"fake-jpeg-data"
        assert restored.window_title == "xterm"
        assert restored.sequence == 42

    def test_event_message_roundtrip(self):
        msg = EventMessage(
            event_type=EventType.WINDOW_CREATE.value,
            wid=5,
            data={"title": "Firefox", "width": 1024},
        )
        raw = msg.serialise()
        restored = EventMessage.deserialise(raw)
        assert restored.event_type == "window_create"
        assert restored.wid == 5
        assert restored.data["title"] == "Firefox"

    def test_control_request_roundtrip(self):
        req = make_action(ActionType.CLICK, wid=1, x=100, y=200, button=1)
        raw = req.serialise()
        restored = ControlRequest.deserialise(raw)
        assert restored.request_type == "action"
        assert restored.payload["action"] == "click"
        assert restored.payload["x"] == 100

    def test_control_response_roundtrip(self):
        resp = ControlResponse(
            success=True, request_id="req-001",
            data={"mode": "observer"},
        )
        raw = resp.serialise()
        restored = ControlResponse.deserialise(raw)
        assert restored.success is True
        assert restored.data["mode"] == "observer"

    def test_make_mode_change(self):
        req = make_mode_change(InputMode.AUTONOMOUS)
        assert req.request_type == "mode"
        assert req.payload["mode"] == "autonomous"

    def test_make_query(self):
        req = make_query(QueryType.WINDOW_LIST)
        assert req.request_type == "query"
        assert req.payload["query"] == "window_list"


class TestFramebuffer:
    def _make_rgb_data(self, w, h, color=(255, 0, 0)):
        """Create raw RGB data for a solid-color region."""
        return bytes(color) * (w * h)

    def _make_jpeg_data(self, w, h, color=(0, 255, 0)):
        """Create JPEG-encoded data for a solid-color region."""
        img = Image.new("RGB", (w, h), color)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    def test_create_and_snapshot(self):
        fb = WindowFramebuffer(wid=1, width=100, height=80)
        # Initially black, not dirty (no updates)
        assert fb.snapshot() is None

        # Update with RGB data
        fb.update_region(0, 0, 50, 40, "rgb24", self._make_rgb_data(50, 40))
        assert fb.dirty is True

        data = fb.snapshot(fmt="jpeg", quality=70)
        assert data is not None
        assert len(data) > 0

        # After snapshot, not dirty
        assert fb.dirty is False
        assert fb.snapshot() is None

    def test_jpeg_region_update(self):
        fb = WindowFramebuffer(wid=2, width=200, height=150)
        jpeg_data = self._make_jpeg_data(100, 75, (0, 0, 255))
        fb.update_region(10, 10, 100, 75, "jpeg", jpeg_data)
        assert fb.dirty is True

        snap = fb.snapshot(fmt="png")
        assert snap is not None
        img = Image.open(io.BytesIO(snap))
        assert img.size == (200, 150)

    def test_resize(self):
        fb = WindowFramebuffer(wid=3, width=100, height=100)
        fb.update_region(0, 0, 100, 100, "rgb24", self._make_rgb_data(100, 100))
        fb.snapshot()  # Clear dirty

        fb.resize(200, 200)
        assert fb.width == 200
        assert fb.height == 200
        assert fb.dirty is True

    def test_snapshot_scaling(self):
        fb = WindowFramebuffer(wid=4, width=1920, height=1080)
        fb.update_region(0, 0, 100, 100, "rgb24", self._make_rgb_data(100, 100))

        data = fb.snapshot(fmt="jpeg", scale=0.5)
        assert data is not None
        img = Image.open(io.BytesIO(data))
        assert img.size == (960, 540)


class TestFramebufferManager:
    def test_create_and_destroy(self):
        mgr = FramebufferManager()
        mgr.create_window(1, 800, 600)
        mgr.create_window(2, 1024, 768)
        assert set(mgr.window_ids) == {1, 2}

        mgr.destroy_window(1)
        assert mgr.window_ids == [2]

    def test_dirty_snapshots(self):
        mgr = FramebufferManager()
        mgr.create_window(1, 100, 100)
        mgr.create_window(2, 100, 100)

        # Update only window 1
        mgr.update_region(1, 0, 0, 50, 50, "rgb24", bytes((255, 0, 0)) * 2500)

        snaps = mgr.get_dirty_snapshots()
        assert len(snaps) == 1
        assert snaps[0][0] == 1  # wid

        # Second call should return nothing (not dirty)
        snaps = mgr.get_dirty_snapshots()
        assert len(snaps) == 0


class TestConfig:
    def test_default_config(self):
        cfg = Config()
        assert cfg.frame_capture.fps == 3.0
        assert cfg.control.mode == "observer"
        assert cfg.zmq.frame_hwm == 5

    def test_load_missing_file(self):
        cfg = load_config("/nonexistent/path.yaml")
        assert cfg.frame_capture.fps == 3.0  # Falls back to defaults
