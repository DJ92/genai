from __future__ import annotations

import importlib.util
import json
import logging
import subprocess
import time
from pathlib import Path

import jsonschema

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def _detect_root_dir() -> Path:
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (parent / "tools").exists() and (parent / "tools" / "specs").exists():
            return parent
    # Fallback for local development where cwd is project root.
    return Path.cwd()


ROOT_DIR = _detect_root_dir()


class ToolRouter:
    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings
        self._specs_dir = ROOT_DIR / settings.tools_specs_dir
        self._impls_dir = ROOT_DIR / "tools" / "implementations"

    def list_tools(self) -> list[str]:
        if not self._specs_dir.exists():
            return []
        return sorted(path.stem for path in self._specs_dir.glob("*.json"))

    def load_spec(self, tool_name: str) -> dict:
        spec_path = self._specs_dir / f"{tool_name}.json"
        if not spec_path.exists():
            raise FileNotFoundError(f"unknown tool spec: {tool_name}")
        return json.loads(spec_path.read_text(encoding="utf-8"))

    def validate_input(self, tool_name: str, args: dict) -> None:
        spec = self.load_spec(tool_name)
        jsonschema.validate(instance=args, schema=spec["input_schema"])

    def validate_output(self, tool_name: str, result: dict) -> None:
        spec = self.load_spec(tool_name)
        jsonschema.validate(instance=result, schema=spec["output_schema"])

    def execute(self, tool_name: str, args: dict, timeout_seconds: int | None = None) -> dict:
        timeout = timeout_seconds or self._settings.tools_timeout_seconds
        started = time.perf_counter()
        self.validate_input(tool_name, args)

        try:
            data = self._run_local(tool_name, args)
            self.validate_output(tool_name, data)
            return {
                "success": True,
                "data": data,
                "error": None,
                "metadata": {
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                    "execution_mode": "local",
                    "exit_code": 0,
                },
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("tool execution failed: %s", tool_name)
            return {
                "success": False,
                "data": None,
                "error": str(exc),
                "metadata": {
                    "duration_ms": int((time.perf_counter() - started) * 1000),
                    "execution_mode": "local",
                    "exit_code": 1,
                    "timeout_seconds": timeout,
                },
            }

    def _run_local(self, tool_name: str, args: dict) -> dict:
        module_path = self._impls_dir / tool_name / "main.py"
        if not module_path.exists():
            raise FileNotFoundError(f"missing tool implementation: {tool_name}")

        module_name = f"tool_impl_{tool_name}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to load tool module: {tool_name}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "run"):
            raise AttributeError(f"tool implementation missing run(): {tool_name}")
        return module.run(args)

    def execute_via_runner(self, tool_name: str, args: dict, timeout_seconds: int | None = None) -> dict:
        timeout = timeout_seconds or self._settings.tools_timeout_seconds
        invocation = {"tool": tool_name, "args": args, "timeout": timeout}
        runner_path = ROOT_DIR / "tools" / "docker" / "tool_runner" / "runner.py"
        command = ["python", str(runner_path)]
        started = time.perf_counter()

        proc = subprocess.run(
            command,
            input=json.dumps(invocation),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )

        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "runner execution failed")
        payload = json.loads(proc.stdout)
        payload.setdefault("metadata", {})
        payload["metadata"].setdefault("duration_ms", int((time.perf_counter() - started) * 1000))
        return payload
