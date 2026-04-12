import json
import os
import sys

from game_runner.gameplay import validate_submission


def main():
    play_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "workspace"))
    with open(os.path.join(os.path.dirname(__file__), "config", "maps.json")) as handle:
        maps = json.load(handle)

    args = sys.argv[1:]
    limit_resources = False
    if "--limit_resources" in args:
        args.remove("--limit_resources")
        limit_resources = True

    agent = args[0] if args else "arjun_and_gary_2026"
    map_names = args[1:] or ["test_map"]

    for map_name in map_names:
        ok, message = validate_submission(
            play_directory,
            agent,
            limit_resources=limit_resources,
            map_string=maps[map_name],
        )
        print(f"MAP {map_name} OK {ok}")
        print((message or "")[:4000])
        print("---")


if __name__ == "__main__":
    main()
