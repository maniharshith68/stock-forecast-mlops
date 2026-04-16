#!/usr/bin/env python3
"""
Start the FastAPI prediction server.

Usage:
    python3 scripts/run_api.py
    python3 scripts/run_api.py --port 8080
    python3 scripts/run_api.py --host 0.0.0.0 --port 8000
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))


def parse_args():
    parser = argparse.ArgumentParser(description="Start the Stock Forecast API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    import uvicorn
    print(f"\n{'='*55}")
    print("  Stock Forecast MLOps API")
    print(f"  http://{args.host}:{args.port}")
    print(f"  Docs: http://localhost:{args.port}/docs")
    print(f"{'='*55}\n")
    uvicorn.run(
        "src.serving.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
