import argparse
from tinylm.server import run_server


def main():
    parser = argparse.ArgumentParser(description="Run the TinyLM Web Playground")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve playground on (default: 8000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    args = parser.parse_args()

    run_server(port=args.port, host=args.host)


if __name__ == "__main__":
    main()
