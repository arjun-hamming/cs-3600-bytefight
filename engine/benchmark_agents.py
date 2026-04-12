import argparse
import os
import random
import statistics

from gameplay import play_game
from game.enums import ResultArbiter


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark local ByteFight agents.")
    parser.add_argument("agent", help="Agent name to benchmark")
    parser.add_argument("opponents", nargs="+", help="Opponent agent names")
    parser.add_argument("--games", type=int, default=4, help="Games per color pairing")
    parser.add_argument("--seed", type=int, default=1000, help="Base random seed")
    return parser.parse_args()


def score_from_perspective(winner, color):
    if winner == ResultArbiter.TIE:
        return 0.5
    if color == "A":
        return 1.0 if winner == ResultArbiter.PLAYER_A else 0.0
    return 1.0 if winner == ResultArbiter.PLAYER_B else 0.0


def points_from_perspective(board, color):
    if color == "A":
        return board.player_worker.get_points(), board.opponent_worker.get_points()
    return board.opponent_worker.get_points(), board.player_worker.get_points()


def main():
    args = parse_args()
    play_directory = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "3600-agents")
    )

    for opponent in args.opponents:
        results = []
        seed = args.seed
        for color, a_name, b_name in (
            ("A", args.agent, opponent),
            ("B", opponent, args.agent),
        ):
            for _ in range(args.games):
                random.seed(seed)
                board, *_ = play_game(
                    play_directory,
                    play_directory,
                    a_name,
                    b_name,
                    display_game=False,
                    delay=0.0,
                    clear_screen=False,
                    record=False,
                    limit_resources=False,
                )
                my_points, opp_points = points_from_perspective(board, color)
                results.append(
                    {
                        "seed": seed,
                        "color": color,
                        "winner": board.winner.name,
                        "reason": board.win_reason.name,
                        "score": score_from_perspective(board.winner, color),
                        "my_points": my_points,
                        "opp_points": opp_points,
                    }
                )
                seed += 1

        avg_score = statistics.mean(item["score"] for item in results)
        avg_point_diff = statistics.mean(
            item["my_points"] - item["opp_points"] for item in results
        )
        print(
            f"{args.agent} vs {opponent}: avg_score={avg_score:.3f} "
            f"avg_point_diff={avg_point_diff:.2f}"
        )
        for item in results:
            print(
                "  "
                f"seed={item['seed']} color={item['color']} winner={item['winner']} "
                f"reason={item['reason']} pts={item['my_points']}-{item['opp_points']} "
                f"score={item['score']:.1f}"
            )


if __name__ == "__main__":
    main()
