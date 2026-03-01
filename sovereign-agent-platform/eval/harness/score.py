from __future__ import annotations

import json

try:
    import jsonschema
except ModuleNotFoundError:  # pragma: no cover - optional dependency in minimal env
    jsonschema = None


def _response_text(response: dict) -> str:
    return str(response.get("response", ""))


def _basic_validate_schema(value, schema: dict) -> tuple[bool, str | None]:
    schema_type = schema.get("type")
    if schema_type == "object":
        if not isinstance(value, dict):
            return False, "expected object"
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                return False, f"missing required key: {key}"
        properties = schema.get("properties", {})
        for key, prop_schema in properties.items():
            if key in value and isinstance(prop_schema, dict):
                ok, err = _basic_validate_schema(value[key], prop_schema)
                if not ok:
                    return False, f"{key}: {err}"
        return True, None
    if schema_type == "array":
        if not isinstance(value, list):
            return False, "expected array"
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for idx, item in enumerate(value):
                ok, err = _basic_validate_schema(item, item_schema)
                if not ok:
                    return False, f"item[{idx}]: {err}"
        return True, None
    if schema_type == "string" and not isinstance(value, str):
        return False, "expected string"
    if schema_type == "number" and not isinstance(value, (int, float)):
        return False, "expected number"
    if schema_type == "integer" and not isinstance(value, int):
        return False, "expected integer"
    if schema_type == "boolean" and not isinstance(value, bool):
        return False, "expected boolean"
    return True, None


def _event_payloads(events: list[dict], event_type: str) -> list[dict]:
    payloads: list[dict] = []
    for event in events:
        if event.get("event_type") != event_type:
            continue
        payload = event.get("payload", {})
        if isinstance(payload, str):
            try:
                parsed = json.loads(payload)
                if isinstance(parsed, dict):
                    payload = parsed
                else:
                    payload = {}
            except json.JSONDecodeError:
                payload = {}
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def score_task(task: dict, response: dict, events: list[dict]) -> dict:
    expected = task.get("expected", {})
    failures: list[str] = []

    response_text = _response_text(response)
    citations = response.get("citations", [])

    if expected.get("must_include_citations") and not citations:
        failures.append("missing citations")
    if expected.get("must_include_citations") is False and citations:
        failures.append("unexpected citations")

    if expected.get("must_extract_fact"):
        fact = expected["must_extract_fact"].lower()
        if fact not in response_text.lower():
            failures.append("expected fact missing")

    if expected.get("must_output_valid_json"):
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            failures.append("response is not valid JSON")
            parsed = None
        schema = expected.get("json_schema")
        if schema is not None and parsed is not None:
            if jsonschema is None:
                ok, err = _basic_validate_schema(parsed, schema)
                if not ok:
                    failures.append(f"json schema validation failed: {err}")
            else:
                try:
                    jsonschema.validate(parsed, schema)
                except jsonschema.ValidationError as exc:
                    failures.append(f"json schema validation failed: {exc.message}")

    tool_calls = _event_payloads(events, "tool_call")
    invoked_tools = {item.get("tool_name") for item in tool_calls if item.get("tool_name")}

    prohibited = set(expected.get("must_not_invoke_tools", []))
    bad_invocations = sorted(tool for tool in invoked_tools if tool in prohibited)
    if bad_invocations:
        failures.append(f"prohibited tools invoked: {', '.join(bad_invocations)}")

    expected_tools = set(expected.get("tool_calls_expected", []))
    missing_expected = sorted(tool for tool in expected_tools if tool not in invoked_tools)
    if missing_expected:
        failures.append(f"expected tool calls missing: {', '.join(missing_expected)}")

    if expected.get("policy_decision_expected"):
        required_decision = expected["policy_decision_expected"]
        policy_events = _event_payloads(events, "policy_decision")
        decisions = {event.get("decision") for event in policy_events}
        if required_decision not in decisions:
            failures.append(f"expected policy decision not found: {required_decision}")

    return {"pass": not failures, "failures": failures}
