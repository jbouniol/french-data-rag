"""Evaluation script stub (implemented by the eval workstream)."""
import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--out", required=True)
    parser.parse_args()
    raise NotImplementedError("run_eval is not implemented yet")


if __name__ == "__main__":
    main()
