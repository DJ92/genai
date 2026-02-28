from urllib.parse import urlparse

import requests


def run(args: dict) -> dict:
    url = args["url"]
    timeout_seconds = int(args.get("timeout_seconds", 15))
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("url must be http or https")

    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    return {
        "url": url,
        "status_code": response.status_code,
        "body": response.text[:5000],
    }
