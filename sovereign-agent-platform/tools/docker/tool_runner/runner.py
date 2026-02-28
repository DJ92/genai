from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import jsonschema

ROOT_DIR = Path(__file__).resolve().parents[2]
SPECS_DIR = ROOT_DIR / "specs"
IMPLS_DIR = ROOT_DIR / "implementations"


def load_spec(tool_name: str) -> dict:
    spec_path = SPECS_DIR / f"{tool_name}.json"
    if not spec_path.exists():
        raise FileNotFoundError(f"unknown tool spec: {tool_name}")
    return json.loads(spec_path.read_text(encoding="utf-8"))


def load_implementation(tool_name: str):
    module_path = IMPLS_DIR / tool_name / "main.py"
    if not module_path.exists():
        raise FileNotFoundError(f"missing tool implementation: {tool_name}")

    module_name = f"tool_impl_{tool_name}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module for {tool_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "run"):
        raise AttributeError(f"{tool_name} implementation missing run()")
    return module


def main() -> None:
    started = time.perf_counter()
    try:
        invocation = json.loads(sys.stdin.read() or "{}")
        tool_name = invocation["tool"]
        args = invocation.get("args", {})

        spec = load_spec(tool_name)
        jsonschema.validate(instance=args, schema=spec["input_schema"])
        impl = load_implementation(tool_name)
        data = impl.run(args)
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
        print(json.dumps(result))
        sys.exit(0)
    except Exception as exc:  # noqa: BLE001
        result = {
            "success": False,
            "data": None,
            "error": str(exc),
            "metadata": {
                "duration_ms": int((time.perf_counter() - started) * 1000),
                "exit_code": 1,
            },
        }
        print(json.dumps(result))
        sys.exit(1)


if __name__ == "__main__":
    main()

