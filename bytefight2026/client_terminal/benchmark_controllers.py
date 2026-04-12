import argparse
import json
import os
import statistics

from game.outcome import Result
from game_runner.gameplay import play_game


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark ByteFight 2026 controllers.")
    parser.add_argument("agent", help="Controller package name in workspace/")
    parser.add_argument("opponent", help="Opponent package name in workspace/")
    parser.add_argument(
        "--maps",
        nargs="*",
        default=["test_map", "boxes", "funnel", "the_future"],
        help="Map names from config/maps.json",
    )
    return parser.parse_args()


def score(result, color):
    if result == Result.TIE:
        return 0.5
    if color == "A":
        return 1.0 if result == Result.PLAYER_1 else 0.0
    return 1.0 if result == Result.PLAYER_2 else 0.0


def main():
    args = parse_args()
    root = os.path.dirname(__file__)
    workspace = os.path.join(root, "workspace")
    with open(os.path.join(root, "config", "maps.json")) as handle:
        maps = json.load(handle)

    results = []
    for map_name in args.maps:
        map_string = maps[map_name]
        for color, a_name, b_name in (
            ("A", args.agent, args.opponent),
            ("B", args.opponent, args.agent),
        ):
            outcome = play_game(
                workspace,
                workspace,
                a_name,
                b_name,
                display_game=False,
                clear_screen=False,
                record=False,
                limit_resources=False,
                map_string=map_string,
            )
            results.append((map_name, color, outcome.result, outcome.reason.name, score(outcome.result, color)))

    average = statistics.mean(item[4] for item in results)
    print(f"{args.agent} vs {args.opponent}: avg_score={average:.3f}")
    for item in results:
        print(
            f"  map={item[0]} color={item[1]} result={item[2].name} "
            f"reason={item[3]} score={item[4]:.1f}"
        )


if __name__ == "__main__":
    main()
