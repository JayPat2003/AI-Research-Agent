"""Run the multi-agent dense RAG pipeline from the terminal."""

from __future__ import annotations

import argparse
import sys

from app.pipeline.ingest import ingest_seed
from app.pipeline.runner import run_query


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous research agent (Gemini + Chroma dense RAG)")
    parser.add_argument("question", nargs="?", help="Research question")
    parser.add_argument("--ingest", action="store_true", help="Rebuild the Chroma index from data/seed")
    parser.add_argument("--conversation-id", help="Continue a previous thread")
    parser.add_argument("--repl", action="store_true", help="Interactive follow-up loop")
    args = parser.parse_args()

    if args.ingest or args.repl or args.question:
        if args.ingest:
            n = ingest_seed()
            print(f"Indexed {n} chunks into Chroma.")
            if not args.question and not args.repl:
                return

    if not args.question and not args.repl:
        parser.print_help()
        sys.exit(1)

    cid = args.conversation_id
    if args.question:
        result = run_query(args.question, cid)
        cid = result["conversation_id"]
        print(f"\nconversation_id={cid}  (pass --conversation-id to follow up)")

    if args.repl:
        print("REPL — empty line to exit.")
        while True:
            try:
                line = input("\nquestion> ").strip()
            except EOFError:
                break
            if not line:
                break
            result = run_query(line, cid)
            cid = result["conversation_id"]


if __name__ == "__main__":
    main()
