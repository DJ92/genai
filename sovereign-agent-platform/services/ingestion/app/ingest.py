import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest documents into sovereign-agent-platform")
    parser.add_argument("--path", required=True, help="Path to folder or file")
    parser.add_argument("--scope", default="personal", help="Scope label for documents")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print({"status": "stub", "path": args.path, "scope": args.scope})


if __name__ == "__main__":
    main()
