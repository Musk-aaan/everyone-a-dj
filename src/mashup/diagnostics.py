"""Send rendered audio to a multimodal LLM via OpenRouter and ask what it hears.

Used to debug the renderer perceptually: spectrograms tell you what energy
sits at what frequency, but a model that listens tells you whether the
result actually sounds like the musical structure you intended.

Requires OPENROUTER_API_KEY in the environment.
"""

from __future__ import annotations

import base64
import os
from typing import Optional

import requests


_DEFAULT_PROMPT = """Listen to this audio carefully. It is supposed to be a 60-second mashup demo with this intended structure:

- 0-6s: bass + chord pad fade in (Song B intro)
- 6-26s: bass progression + chord pad as a "groove bed" (Song B)
- 8-12s and 16-20s: a high-pitched 5-note melodic hook (C-D-E-D-C in the upper register) layered on top of the bed - this is supposed to be Song A's "vocal hook"
- 26-30s: crossfade
- 30-60s: a 4-chord progression (C-G-Am-F repeating) as the outro (Song A)
- around 35s: one more melodic hook lands inside the outro

For each section, describe what you actually hear. Be specific about:
1. Whether you hear musical content (chords, melody) or just noise / clicks / beats.
2. Whether the upper-register melodic hooks at ~8s, ~16s, ~35s are audible at all.
3. Whether the 4-chord progression is perceptible as music.
4. Anything that sounds wrong, glitchy, muddy, distorted, or unbalanced.

Be honest. If it sounds like "random beats" rather than music, say so and tell me which timestamps are problematic."""


def listen(
    audio_path: str,
    *,
    model: str = "google/gemini-2.5-flash",
    prompt: str = _DEFAULT_PROMPT,
    api_key: Optional[str] = None,
    timeout: float = 120.0,
) -> Optional[str]:
    """POST `audio_path` to OpenRouter for the named model. Returns the
    model's text response, or None if no API key is configured."""
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None

    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("ascii")
    fmt = audio_path.rsplit(".", 1)[-1].lower() or "mp3"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": fmt},
                    },
                ],
            }
        ],
    }
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Agent-Prod/Agentprod-Backend-Framework",
            "X-Title": "mashup-diagnostics",
        },
        json=payload,
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]
