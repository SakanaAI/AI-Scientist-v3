"""Baseline program (TODO: replace with the real baseline).

The template evaluator runs this file from a scratch copy of `src/` and scores the
`solution.json` it writes. Replace both sides together.
"""

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="search")
    parser.parse_args()
    with open("solution.json", "w", encoding="utf-8") as handle:
        json.dump({"value": 40.0}, handle)
    print("step 1/1 done")


if __name__ == "__main__":
    main()
