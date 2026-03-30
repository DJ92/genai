from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def detect_requested_tools(prompt: str) -> list[dict[str, Any]]:
    tools = []
    match = re.search(r"https?://[^\s]+", prompt)
    if match or "fetch" in prompt.lower():
        payload = {"url": match.group(0)} if match else {}
        tools.append({"tool_name": "web_fetch", "payload": payload})
    return tools


def evaluate_tool_request(tool_name: str) -> dict[str, str]:
    if tool_name == "web_fetch":
        return {
            "decision": "needs_approval",
            "reason": "Network egress is disabled by default and external fetches require approval.",
        }
    return {"decision": "allow", "reason": "Tool is allowed in the current policy scope."}


def validate_tool_specs(spec_dir: str | Path) -> dict[str, Any]:
    spec_path = Path(spec_dir)
    reports = []
    required_top_level = {"name", "description", "input_schema", "output_schema"}
    for path in sorted(spec_path.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        missing = sorted(required_top_level - payload.keys())
        reports.append(
            {
                "file": path.name,
                "valid": not missing,
                "missing": missing,
            }
        )
    return {
        "checked": len(reports),
        "valid": sum(1 for report in reports if report["valid"]),
        "reports": reports,
    }
