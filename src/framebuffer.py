"""
Per-window composited framebuffer management.

Instead of forwarding every damage rect to the AI agent (too much bandwidth),
we maintain a composited framebuffer per window and deliver full-window
snapshots at a configurable rate.
"""

import io
import time
import logging
import threading
from typing import Optional, Dict, Tuple

from PIL import Image

logger = logging.getLogger(__name__)


class WindowFramebuffer:
    """Composited framebuffer for a single X11 window."""

    def __init__(self, wid: int, width: int, height: int):
        self.wid = wid
        self.width = width
        self.height = height
        self.image = Image.new("RGB", (width, height), (0, 0, 0))
        self.dirty = False
        self.last_sent = 0.0
        self.sequence = 0
        self._lock = threading.Lock()

    def update_region(self, x: int, y: int, w: int, h: int,
                      coding: str, data: bytes):
        """Composite a damage region into the framebuffer.

        Handles the common Xpra encodings: rgb24/32, png, jpeg, webp.
        """
        with self._lock:
            try:
                if coding in ("rgb24", "rgb32", "rgbx"):
                    channels = 4 if coding in ("rgb32", "rgbx") else 3
                    mode = "RGBA" if channels == 4 else "RGB"
                    region = Image.frombytes(mode, (w, h), data)
                    if mode == "RGBA":
                        region = region.convert("RGB")
                elif coding in ("png", "jpeg", "webp", "png/L", "png/P"):
                    region = Image.open(io.BytesIO(data)).convert("RGB")
                else:
                    logger.debug("Unsupported encoding %s for wid %d", coding, self.wid)
                    return

                self.image.paste(region, (x, y))
                self.dirty = True
            except Exception as e:
                logger.warning("Failed to composite region for wid %d: %s", self.wid, e)

    def resize(self, width: int, height: int):
        """Handle window resize by creating a new backing."""
        with self._lock:
            old = self.image
            self.image = Image.new("RGB", (width, height), (0, 0, 0))
            # Paste old content at origin (will be clipped if shrunk)
            self.image.paste(old, (0, 0))
            self.width = width
            self.height = height
            self.dirty = True

    def snapshot(self, fmt: str = "jpeg", quality: int = 70,
                 scale: float = 1.0, max_dim: int = 1920) -> Optional[bytes]:
        """Get a compressed snapshot of the current framebuffer.

        Returns None if the buffer hasn't changed since last snapshot.
        """
        with self._lock:
            if not self.dirty:
                return None

            img = self.image.copy()
            self.dirty = False
            self.sequence += 1

        # Downscale if needed
        w, h = img.size
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        elif max(w, h) > max_dim:
            ratio = max_dim / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

        # Encode
        buf = io.BytesIO()
        if fmt == "jpeg":
            img.save(buf, format="JPEG", quality=quality)
        elif fmt == "png":
            img.save(buf, format="PNG")
        elif fmt == "raw":
            return img.tobytes()
        else:
            img.save(buf, format="JPEG", quality=quality)

        return buf.getvalue()


class FramebufferManager:
    """Manages framebuffers for all tracked windows."""

    def __init__(self):
        self._buffers: Dict[int, WindowFramebuffer] = {}
        self._lock = threading.Lock()

    def create_window(self, wid: int, width: int, height: int) -> WindowFramebuffer:
        with self._lock:
            fb = WindowFramebuffer(wid, width, height)
            self._buffers[wid] = fb
            return fb

    def destroy_window(self, wid: int):
        with self._lock:
            self._buffers.pop(wid, None)

    def get(self, wid: int) -> Optional[WindowFramebuffer]:
        with self._lock:
            return self._buffers.get(wid)

    def resize_window(self, wid: int, width: int, height: int):
        fb = self.get(wid)
        if fb:
            fb.resize(width, height)

    def update_region(self, wid: int, x: int, y: int, w: int, h: int,
                      coding: str, data: bytes):
        fb = self.get(wid)
        if fb:
            fb.update_region(x, y, w, h, coding, data)

    def get_dirty_snapshots(self, fmt: str = "jpeg", quality: int = 70,
                            scale: float = 1.0, max_dim: int = 1920,
                            min_interval: float = 0.0
                            ) -> list[Tuple[int, bytes, int]]:
        """Get snapshots for all dirty windows, respecting rate limits.

        Returns list of (wid, data, sequence) tuples.
        """
        now = time.time()
        results = []
        with self._lock:
            buffers = list(self._buffers.values())

        for fb in buffers:
            if min_interval > 0 and (now - fb.last_sent) < min_interval:
                continue
            data = fb.snapshot(fmt, quality, scale, max_dim)
            if data is not None:
                fb.last_sent = now
                results.append((fb.wid, data, fb.sequence))

        return results

    @property
    def window_ids(self) -> list[int]:
        with self._lock:
            return list(self._buffers.keys())
