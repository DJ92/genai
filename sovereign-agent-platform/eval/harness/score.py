def score_task(task: dict, response: dict, events: list[dict]) -> dict:
    expected = task.get("expected", {})
    failures: list[str] = []

    if expected.get("must_include_citations") and not response.get("citations"):
        failures.append("missing citations")

    prohibited = set(expected.get("must_not_invoke_tools", []))
    if prohibited:
        invoked = {event.get("payload", {}).get("tool_name") for event in events if event.get("event_type") == "tool_call"}
        bad = sorted(tool for tool in invoked if tool in prohibited)
        if bad:
            failures.append(f"prohibited tools invoked: {', '.join(bad)}")

    if expected.get("must_extract_fact"):
        fact = expected["must_extract_fact"].lower()
        response_text = str(response.get("response", "")).lower()
        if fact not in response_text:
            failures.append("expected fact missing")

    return {
        "pass": not failures,
        "failures": failures,
    }
