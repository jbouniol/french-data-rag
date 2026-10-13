"""Stub du script d'évaluation (implémenté par le chantier eval)."""
import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--out", required=True)
    parser.parse_args()
    raise NotImplementedError("run_eval n'est pas encore implémenté")


if __name__ == "__main__":
    main()
