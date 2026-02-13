"""
Shared message definitions for the AI control plane.

All messages are serialised as msgpack over ZMQ.
"""

import time
import struct
from enum import Enum, auto
from dataclasses import dataclass, field, asdict
from typing import Optional, Any

import msgpack


class InputMode(Enum):
    """AI control input gating modes."""
    OBSERVER = "observer"
    SUPERVISED = "supervised"
    AUTONOMOUS = "autonomous"
    COLLABORATIVE = "collaborative"


class EventType(Enum):
    """Window lifecycle and state events."""
    WINDOW_CREATE = "window_create"
    WINDOW_DESTROY = "window_destroy"
    WINDOW_FOCUS = "window_focus"
    WINDOW_RESIZE = "window_resize"
    WINDOW_TITLE = "window_title"
    WINDOW_ICON = "window_icon"
    MODE_CHANGE = "mode_change"
    CLIPBOARD_UPDATE = "clipboard_update"
    KILL_SWITCH = "kill_switch"


class ActionType(Enum):
    """Actions the AI agent can request."""
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    MOUSE_MOVE = "mouse_move"
    MOUSE_DOWN = "mouse_down"
    MOUSE_UP = "mouse_up"
    SCROLL = "scroll"
    KEY_PRESS = "key_press"
    KEY_DOWN = "key_down"
    KEY_UP = "key_up"
    TYPE_TEXT = "type_text"
    SET_CLIPBOARD = "set_clipboard"


class QueryType(Enum):
    """Queries the agent can make about current state."""
    WINDOW_LIST = "window_list"
    WINDOW_INFO = "window_info"
    FOCUSED_WINDOW = "focused_window"
    CLIPBOARD = "clipboard"
    SCREENSHOT = "screenshot"
    CURRENT_MODE = "current_mode"


# --- Frame Messages (PUB/SUB) ---

@dataclass
class FrameMessage:
    """A captured frame region delivered to the agent."""
    wid: int                    # Xpra window ID
    x: int                      # Region offset X
    y: int                      # Region offset Y
    width: int                  # Region width
    height: int                 # Region height
    format: str                 # "jpeg", "png", "raw"
    data: bytes                 # Compressed image data
    timestamp: float = field(default_factory=time.time)
    window_title: str = ""
    window_class: str = ""
    is_focused: bool = False
    sequence: int = 0           # Monotonic frame counter

    def serialise(self) -> bytes:
        """Serialise to msgpack bytes with topic prefix."""
        d = asdict(self)
        return msgpack.packb(d, use_bin_type=True)

    @classmethod
    def deserialise(cls, data: bytes) -> "FrameMessage":
        d = msgpack.unpackb(data, raw=False)
        return cls(**d)


# --- Event Messages (PUB/SUB) ---

@dataclass
class EventMessage:
    """A window or system event."""
    event_type: str             # EventType value
    wid: int = 0
    timestamp: float = field(default_factory=time.time)
    data: dict = field(default_factory=dict)

    def serialise(self) -> bytes:
        return msgpack.packb(asdict(self), use_bin_type=True)

    @classmethod
    def deserialise(cls, data: bytes) -> "EventMessage":
        d = msgpack.unpackb(data, raw=False)
        return cls(**d)


# --- Control Messages (REQ/REP) ---

@dataclass
class ControlRequest:
    """A request from agent to plugin."""
    request_type: str           # "action", "query", "mode"
    payload: dict = field(default_factory=dict)
    request_id: str = ""

    def serialise(self) -> bytes:
        return msgpack.packb(asdict(self), use_bin_type=True)

    @classmethod
    def deserialise(cls, data: bytes) -> "ControlRequest":
        d = msgpack.unpackb(data, raw=False)
        return cls(**d)


@dataclass
class ControlResponse:
    """A response from plugin to agent."""
    success: bool
    request_id: str = ""
    data: Any = None
    error: str = ""

    def serialise(self) -> bytes:
        return msgpack.packb(asdict(self), use_bin_type=True)

    @classmethod
    def deserialise(cls, data: bytes) -> "ControlResponse":
        d = msgpack.unpackb(data, raw=False)
        return cls(**d)


# --- Helper constructors ---

def make_action(action_type: ActionType, **kwargs) -> ControlRequest:
    """Build an action request."""
    return ControlRequest(
        request_type="action",
        payload={"action": action_type.value, **kwargs},
    )


def make_query(query_type: QueryType, **kwargs) -> ControlRequest:
    """Build a query request."""
    return ControlRequest(
        request_type="query",
        payload={"query": query_type.value, **kwargs},
    )


def make_mode_change(mode: InputMode) -> ControlRequest:
    """Build a mode change request."""
    return ControlRequest(
        request_type="mode",
        payload={"mode": mode.value},
    )
