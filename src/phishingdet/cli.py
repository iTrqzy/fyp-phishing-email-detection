# src/phishingdet/cli.py
import argparse


def main() -> int:
    parser = argparse.ArgumentParser(prog="phishingdet")
    parser.add_argument("--version", action="store_true", help="Show version and exit")
    args = parser.parse_args()

    if args.version:
        print("phishingdet 0.1.0")
        return 0

    print("phishingdet: starter structure OK ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
