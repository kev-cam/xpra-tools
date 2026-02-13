"""Configuration loading and defaults."""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FrameCaptureConfig:
    fps: float = 3.0
    format: str = "jpeg"
    quality: int = 70
    scale: float = 1.0
    max_dimension: int = 1920
    delta_only: bool = True


@dataclass
class ControlConfig:
    mode: str = "observer"
    kill_switch: str = "ctrl+Pause"
    autonomous_timeout: int = 300
    human_priority_ms: int = 500


@dataclass
class ZMQConfig:
    frame_endpoint: str = "ipc:///tmp/xpra-ai-frames.sock"
    event_endpoint: str = "ipc:///tmp/xpra-ai-events.sock"
    control_endpoint: str = "ipc:///tmp/xpra-ai-control.sock"
    frame_hwm: int = 5
    event_hwm: int = 100


@dataclass
class AgentConfig:
    model_endpoint: str = "http://localhost:8080/v1/chat/completions"
    model_name: str = "claude-sonnet-4-20250514"
    system_prompt: str = "You are an AI agent controlling an X11 desktop."
    confirm_timeout: int = 30


@dataclass
class LoggingConfig:
    level: str = "INFO"
    stats_interval: int = 30
    file: str = ""


@dataclass
class Config:
    frame_capture: FrameCaptureConfig = field(default_factory=FrameCaptureConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    zmq: ZMQConfig = field(default_factory=ZMQConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(path: Optional[str] = None) -> Config:
    """Load configuration from YAML file, falling back to defaults.

    Config path resolution:
    1. Explicit path argument
    2. XPRA_AI_CONTROL_CONFIG environment variable
    3. ./config.yaml
    4. All defaults
    """
    if path is None:
        path = os.environ.get("XPRA_AI_CONTROL_CONFIG", "config.yaml")

    config = Config()

    if path and os.path.exists(path):
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        if "frame_capture" in raw:
            for k, v in raw["frame_capture"].items():
                if hasattr(config.frame_capture, k):
                    setattr(config.frame_capture, k, v)

        if "control" in raw:
            for k, v in raw["control"].items():
                if hasattr(config.control, k):
                    setattr(config.control, k, v)

        if "zmq" in raw:
            for k, v in raw["zmq"].items():
                if hasattr(config.zmq, k):
                    setattr(config.zmq, k, v)

        if "agent" in raw:
            for k, v in raw["agent"].items():
                if hasattr(config.agent, k):
                    setattr(config.agent, k, v)

        if "logging" in raw:
            for k, v in raw["logging"].items():
                if hasattr(config.logging, k):
                    setattr(config.logging, k, v)

    return config
