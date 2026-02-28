import importlib
import json
import sys
import time
from pathlib import Path

import jsonschema

BASE_DIR = Path(__file__).resolve().parents[2]
SPECS_DIR = BASE_DIR / "specs"
IMPLS_DIR = BASE_DIR / "implementations"


def load_spec(tool_name: str) -> dict:
    spec_path = SPECS_DIR / f"{tool_name}.json"
    if not spec_path.exists():
        raise FileNotFoundError(f"unknown tool spec: {tool_name}")
    return json.loads(spec_path.read_text(encoding="utf-8"))


def run_tool(tool_name: str, args: dict) -> dict:
    module = importlib.import_module(f"tools.implementations.{tool_name}.main")
    if not hasattr(module, "run"):
        raise AttributeError(f"tool implementation missing run(): {tool_name}")
    return module.run(args)


def main() -> None:
    started = time.perf_counter()
    exit_code = 0

    try:
        invocation = json.loads(sys.stdin.read() or "{}")
        tool_name = invocation["tool"]
        args = invocation.get("args", {})

        spec = load_spec(tool_name)
        jsonschema.validate(instance=args, schema=spec["input_schema"])
        data = run_tool(tool_name, args)
        jsonschema.validate(instance=data, schema=spec["output_schema"])

        result = {
            "success": True,
            "data": data,
            "error": None,
            "metadata": {
                "duration_ms": int((time.perf_counter() - started) * 1000),
                "exit_code": 0,
            },
        }
    except Exception as exc:  # noqa: BLE001
        exit_code = 1
        result = {
            "success": False,
            "data": None,
            "error": str(exc),
            "metadata": {
                "duration_ms": int((time.perf_counter() - started) * 1000),
                "exit_code": exit_code,
            },
        }

    print(json.dumps(result))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
