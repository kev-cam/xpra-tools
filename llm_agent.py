#!/usr/bin/env python3
"""
LLM-Driven Vision Agent

Uses Claude's vision capability to observe X11 windows and decide on actions.
Implements a simple observe → think → act loop.

Usage:
    export ANTHROPIC_API_KEY=sk-...
    python examples/llm_agent.py --config config.yaml --goal "Open a terminal and list files"
"""

import io
import sys
import time
import json
import base64
import logging
import argparse
from pathlib import Path
from typing import Optional

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xpra_ai_control.agent import AIAgent

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
)
log = logging.getLogger(__name__)

# Action schema for structured output
ACTION_SCHEMA = """
Respond with a JSON object containing:
{
  "observation": "What you see on the screen",
  "reasoning": "What you think should happen next",
  "action": {
    "type": "click|double_click|right_click|type_text|key_press|scroll|done|wait",
    "x": 100,        // for click/scroll
    "y": 200,        // for click/scroll
    "text": "...",    // for type_text
    "key": "...",     // for key_press (e.g., "Return", "Tab", "ctrl+c")
    "dy": -3          // for scroll (negative = up)
  },
  "done": false       // true when the goal is achieved
}
"""

SYSTEM_PROMPT = """You are an AI agent controlling a Linux desktop via X11.
You receive screenshots of the desktop and can perform mouse/keyboard actions.
Your goal is to complete the user's requested task.

{action_schema}

Important:
- Be precise with click coordinates based on what you see in the screenshot.
- Use keyboard shortcuts when efficient (e.g., Ctrl+L for address bar).
- If something doesn't work, try an alternative approach.
- Set done=true when the task is complete or you've determined it cannot be done.
- Use "wait" action type if you need to wait for something to load.
""".format(action_schema=ACTION_SCHEMA)


class LLMVisionAgent:
    """Agent that uses Claude's vision API to drive desktop actions."""

    def __init__(self, agent: AIAgent, api_key: str,
                 model: str = "claude-sonnet-4-20250514",
                 max_steps: int = 20):
        self.agent = agent
        self.api_key = api_key
        self.model = model
        self.max_steps = max_steps
        self.history = []
        self.client = httpx.Client(timeout=60)

    def run(self, goal: str):
        """Execute the observe-think-act loop until goal is achieved."""
        log.info("Goal: %s", goal)
        self.agent.set_mode("autonomous")

        try:
            for step in range(self.max_steps):
                log.info("--- Step %d/%d ---", step + 1, self.max_steps)

                # 1. Observe
                frame = self.agent.wait_for_frame(timeout=5)
                if frame is None:
                    log.warning("No frame available, waiting...")
                    time.sleep(2)
                    continue

                # Encode frame as base64 JPEG
                buf = io.BytesIO()
                frame.save(buf, format="JPEG", quality=80)
                b64_image = base64.b64encode(buf.getvalue()).decode()

                # 2. Think (call Claude)
                action_json = self._call_llm(goal, b64_image)
                if action_json is None:
                    log.error("LLM returned no action, retrying...")
                    time.sleep(1)
                    continue

                log.info("Observation: %s", action_json.get("observation", "?"))
                log.info("Reasoning: %s", action_json.get("reasoning", "?"))

                # 3. Check if done
                if action_json.get("done", False):
                    log.info("Goal achieved!")
                    break

                # 4. Act
                action = action_json.get("action", {})
                self._execute_action(action)

                # Brief pause for the UI to update
                time.sleep(0.5)
            else:
                log.warning("Max steps reached without completing goal")
        finally:
            self.agent.set_mode("observer")
            log.info("Reverted to observer mode")

    def _call_llm(self, goal: str, b64_image: str) -> Optional[dict]:
        """Call Claude with the current screenshot and get an action."""
        messages = [
            *self.history,
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64_image,
                        },
                    },
                    {
                        "type": "text",
                        "text": f"Goal: {goal}\n\nWhat do you see, and what action should be taken?",
                    },
                ],
            },
        ]

        try:
            resp = self.client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": 1024,
                    "system": SYSTEM_PROMPT,
                    "messages": messages,
                },
            )
            resp.raise_for_status()
            data = resp.json()

            # Extract text response
            text = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    text += block["text"]

            # Parse JSON from response
            # Strip markdown fences if present
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                text = text.rsplit("```", 1)[0]

            result = json.loads(text)

            # Append to conversation history (without image to save tokens)
            self.history.append({
                "role": "user",
                "content": f"[Screenshot shown] Goal: {goal}"
            })
            self.history.append({
                "role": "assistant",
                "content": text,
            })

            # Keep history manageable
            if len(self.history) > 20:
                self.history = self.history[-10:]

            return result

        except Exception as e:
            log.error("LLM call failed: %s", e)
            return None

    def _execute_action(self, action: dict):
        """Execute an action returned by the LLM."""
        action_type = action.get("type", "wait")

        if action_type == "click":
            x, y = action["x"], action["y"]
            log.info("Action: click(%d, %d)", x, y)
            self.agent.click(x=x, y=y)

        elif action_type == "double_click":
            x, y = action["x"], action["y"]
            log.info("Action: double_click(%d, %d)", x, y)
            self.agent.double_click(x=x, y=y)

        elif action_type == "right_click":
            x, y = action["x"], action["y"]
            log.info("Action: right_click(%d, %d)", x, y)
            self.agent.right_click(x=x, y=y)

        elif action_type == "type_text":
            text = action["text"]
            log.info("Action: type_text(%r)", text)
            self.agent.type_text(text)

        elif action_type == "key_press":
            key = action["key"]
            log.info("Action: key_press(%s)", key)
            self.agent.key_press(key)

        elif action_type == "scroll":
            x, y = action.get("x", 0), action.get("y", 0)
            dy = action.get("dy", -3)
            log.info("Action: scroll(%d, %d, dy=%d)", x, y, dy)
            self.agent.scroll(x=x, y=y, dy=dy)

        elif action_type == "wait":
            log.info("Action: wait")
            time.sleep(2)

        elif action_type == "done":
            log.info("Action: done")

        else:
            log.warning("Unknown action type: %s", action_type)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--goal", required=True, help="Task for the AI to accomplish")
    parser.add_argument("--api-key", default=None,
                        help="Anthropic API key (or set ANTHROPIC_API_KEY)")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--max-steps", type=int, default=20)
    args = parser.parse_args()

    import os
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: Set ANTHROPIC_API_KEY or use --api-key", file=sys.stderr)
        sys.exit(1)

    agent = AIAgent(args.config)
    agent.connect()
    time.sleep(1)  # Wait for initial frames

    llm_agent = LLMVisionAgent(
        agent=agent,
        api_key=api_key,
        model=args.model,
        max_steps=args.max_steps,
    )

    try:
        llm_agent.run(args.goal)
    finally:
        agent.disconnect()


if __name__ == "__main__":
    main()
