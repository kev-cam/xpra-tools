# xpra-tools: AI Control Layer for X11 via Xpra

An AI control layer that inserts into the Xpra stack, enabling autonomous or supervised AI interaction with X11 applications.

## Architecture

```
X11 Apps ↔ Xvfb ↔ Xpra Server + AI Plugin ↔ Xpra Client ↔ Display
                        ↕ ZMQ IPC
                   AI Agent Process
```

The plugin taps Xpra's internal frame and input pipelines, exposing them over a ZMQ control plane to an external AI agent process. This gives the agent:

- **Observation**: Window frames, metadata (title, class, geometry, focus), clipboard
- **Actuation**: Mouse clicks, keyboard input, scroll, window management
- **Gating**: Observer / Supervised / Autonomous / Collaborative input modes

## Components

| File | Purpose |
|------|---------|
| `xpra_ai_control/plugin.py` | Xpra server plugin — frame tap, input injection, ZMQ bridge |
| `xpra_ai_control/agent.py` | Standalone AI agent process with ZMQ client |
| `xpra_ai_control/protocol.py` | Shared message definitions and serialisation |
| `xpra_ai_control/config.py` | Configuration loading and defaults |
| `xpra_ai_control/framebuffer.py` | Per-window composited framebuffer management |
| `examples/simple_agent.py` | Minimal agent that screenshots and clicks |
| `config.yaml` | Default configuration |

## Quick Start

### Prerequisites

```bash
# Xpra with X11 bindings
apt install xpra xvfb python3-xpra

# Python deps
pip install pyzmq pillow numpy pyyaml
```

### Running

```bash
# 1. Start Xpra with the AI control plugin
xpra start :100 \
  --start="xterm" \
  --bind-tcp=0.0.0.0:14500 \
  --env=XPRA_AI_CONTROL_CONFIG=./config.yaml

# 2. In another terminal, run the AI agent
python -m xpra_ai_control.agent --config config.yaml

# 3. Optionally attach a viewer
xpra attach tcp://localhost:14500
```

### Input Modes

| Mode | AI Sees | AI Acts | Human Acts |
|------|---------|---------|------------|
| `observer` | ✓ | ✗ | ✓ |
| `supervised` | ✓ | proposes → human approves | ✓ |
| `autonomous` | ✓ | ✓ | suppressed (kill switch active) |
| `collaborative` | ✓ | ✓ | ✓ (AI yields on conflict) |

**Kill switch**: `Ctrl+Pause` always passes through and forces mode back to `observer`.

## Configuration

See `config.yaml` for all options. Key settings:

```yaml
frame_capture:
  fps: 3.0              # Frame delivery rate to agent
  format: jpeg           # jpeg | png | raw
  quality: 70            # JPEG quality (1-100)
  scale: 1.0             # Downscale factor

control:
  mode: observer         # Default input mode
  kill_switch: ctrl+Pause

zmq:
  frame_endpoint: ipc:///tmp/xpra-ai-frames.sock
  event_endpoint: ipc:///tmp/xpra-ai-events.sock
  control_endpoint: ipc:///tmp/xpra-ai-control.sock
```

## Developing an Agent

```python
from xpra_ai_control.agent import AIAgent

agent = AIAgent("config.yaml")
agent.connect()

# Get current window list
windows = agent.query_windows()

# Grab a frame
frame = agent.get_frame(wid=windows[0]["wid"])
# frame.image is a PIL Image

# Perform actions
agent.click(wid=1, x=100, y=200)
agent.type_text("hello world")
agent.key_press("Return")

# Switch mode
agent.set_mode("autonomous")
```

## License

MIT
