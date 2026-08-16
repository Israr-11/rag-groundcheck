"""Command-line interface: `groundcheck`.

Usage:
    groundcheck run example.json
    groundcheck run example.json --json

Where example.json looks like:
    {
        "question": "What is the refund policy?",
        "contexts": ["You can return items within 30 days..."],
        "answer": "The refund policy allows returns within 30 days."
    }

Or a list of such objects, to evaluate a batch in one go.
"""

import argparse
import json
import sys

from groundcheck.evaluate import evaluate


def _run_one(item: dict, as_json: bool) -> dict:
    result = evaluate(
        question=item["question"],
        contexts=item["contexts"],
        answer=item["answer"],
    )
    if as_json:
        return result.to_dict()
    print(result.summary())
    print()
    return result.to_dict()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="groundcheck", description="Evaluate RAG outputs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Evaluate one or more (question, contexts, answer) items from a JSON file.")
    run_parser.add_argument("file", help="Path to a JSON file (single object or list of objects).")
    run_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of a human summary.")

    args = parser.parse_args(argv)

    if args.command == "run":
        try:
            with open(args.file, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Error: file not found: {args.file}", file=sys.stderr)
            return 1
        except json.JSONDecodeError as e:
            print(f"Error: invalid JSON in {args.file}: {e}", file=sys.stderr)
            return 1

        items = data if isinstance(data, list) else [data]
        results = [_run_one(item, args.json) for item in items]

        if args.json:
            print(json.dumps(results if len(results) > 1 else results[0], indent=2))

        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
