import os, json, re
from tqdm import tqdm

AGENT_POOL_DIR = "3600-agents"
TARGET_AGENTS = ["0_aggressive", "1_pos_deny", "2_max", "3_phase", "4_aco"]

def get_latest_season_dir():
    existing_seasons = []
    for d in os.listdir(AGENT_POOL_DIR):
        match = re.match(r"matches_season_(\d+)", d)
        if match: existing_seasons.append(int(match.group(1)))
    return os.path.join(AGENT_POOL_DIR, f"matches_season_{max(existing_seasons)}") if existing_seasons else None

def consolidate_to_individual_jsons():
    season_dir = get_latest_season_dir()
    if not season_dir: return

    # Input reads from the raw /individual subfolder
    input_dir = os.path.join(season_dir, "individual")
    if not os.path.exists(input_dir):
        print(f"⚠️ No individual directory found in {season_dir}")
        return

    agent_vaults = {agent: [] for agent in TARGET_AGENTS}
    files = [f for f in os.listdir(input_dir) if f.endswith(".json")]

    for file in tqdm(files, desc="Grouping Matches"):
        path = os.path.join(input_dir, file)
        try:
            with open(path, 'r') as f:
                match_data = json.load(f)
                for agent in TARGET_AGENTS:
                    if agent in file: agent_vaults[agent].append(match_data)
        except Exception: continue

    # Output writes directly to the root season_dir
    for agent, history in agent_vaults.items():
        output_path = os.path.join(season_dir, f"{agent}_full_history.json")
        with open(output_path, 'w') as out_f:
            json.dump(history, out_f, indent=2)

if __name__ == "__main__":
    consolidate_to_individual_jsons()