from __future__ import annotations

import json
import os
import re


class ModelUnavailable(RuntimeError):
    pass


def validate_configs(configs: object, attempts: int) -> list[dict[str, int | str]]:
    if not isinstance(configs, list):
        raise ModelUnavailable("Nemotron response did not contain a configuration list.")
    safe: list[dict[str, int | str]] = []
    allowed_blocks, allowed_warps = {128, 256, 512, 1024}, {4, 8}
    seen: set[tuple[int, int]] = set()
    for item in configs:
        if not isinstance(item, dict):
            continue
        try:
            block, warps = int(item["block_size"]), int(item["num_warps"])
        except (KeyError, TypeError, ValueError):
            continue
        if block not in allowed_blocks or warps not in allowed_warps or (block, warps) in seen:
            continue
        seen.add((block, warps))
        safe.append({"block_size": block, "num_warps": warps, "rationale": re.sub(r"[^\w .,;()/-]", "", str(item.get("rationale", "Nemotron proposal")))[:180]})
    if not safe:
        raise ModelUnavailable("Nemotron returned no launch configuration in the permitted search space.")
    return safe[:attempts]


def propose_configs(*, size: int, attempts: int) -> list[dict[str, int | str]]:
    """Ask Nebius Nemotron for launch parameters only; never execute generated code."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ModelUnavailable("Install requirements.txt to enable the Token Factory client.") from exc

    key = os.getenv("NEBIUS_API_KEY")
    if not key:
        raise ModelUnavailable("NEBIUS_API_KEY is not set.")
    model = os.getenv("NEBIUS_MODEL", "nvidia/Nemotron-3_5-Lightning")
    client = OpenAI(api_key=key, base_url=os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"))
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        max_tokens=500,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You optimize a Triton vector-add benchmark. Return JSON only: {\"configs\":[{\"block_size\": integer, \"num_warps\": integer, \"rationale\": short string}]}. Do not include code. block_size must be one of 128, 256, 512, 1024. num_warps must be one of 4, 8. Give distinct configurations."},
            {"role": "user", "content": f"Suggest up to {attempts} distinct launch configurations for elementwise vector addition with N={size}. Consider memory-bound behavior. Return only the requested JSON."},
        ],
    )
    content = response.choices[0].message.content or ""
    try:
        parsed = json.loads(content)
        configs = parsed["configs"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ModelUnavailable("Nemotron response did not match the expected JSON schema.") from exc

    return validate_configs(configs, attempts)
