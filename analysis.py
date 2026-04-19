import os, ollama, sys, pandas as pd, re
from tqdm import tqdm

AGENT_POOL_DIR = "3600-agents"
OLLAMA_MODEL = "gemma4:e4b"
TARGET_AGENTS = ["0_aggressive", "1_pos_deny", "2_max", "3_phase", "4_aco"]

class StrategyScout:
    def __init__(self, model=OLLAMA_MODEL):
        self.model = model
        self.system_rules = "RULES: Priming=+1. Carpeting (n=7: 21 pts). Rat Hit=+4, Miss=-2."

    def get_latest_season_dir(self):
        """Finds the main folder in the latest season."""
        existing = [int(re.search(r"(\d+)", d).group(1)) for d in os.listdir(AGENT_POOL_DIR) if "matches_season" in d]
        if not existing: return None
        return os.path.join(AGENT_POOL_DIR, f"matches_season_{max(existing)}")

    def analyze_agent(self, agent_name, season_dir):
        # Load the summarized data from the main season folder
        data_path = os.path.join(season_dir, f"{agent_name}_full_history.json")
        if not os.path.exists(data_path): return

        # Load Agent Code logic
        code_path = os.path.join(AGENT_POOL_DIR, agent_name, "agent.py")
        with open(code_path, 'r') as f: agent_code = f.read()

        tqdm.write(f"\n{'='*70}\n🔍 SCOUTING: {agent_name}\n{'='*70}")
        prompt = f"Analyze this bot's performance and code:\n{agent_code}\nExplain improvements based on CS3600 rules."

        response = ollama.chat(model=self.model, stream=True, messages=[
            {'role': 'system', 'content': self.system_rules},
            {'role': 'user', 'content': prompt}
        ])
        for chunk in response:
            sys.stdout.write(chunk['message']['content']); sys.stdout.flush()

    def run_batch(self):
        target_dir = self.get_latest_season_dir()
        if not target_dir: return
        for agent in tqdm(TARGET_AGENTS, desc="Analyzing"):
            self.analyze_agent(agent, target_dir)

if __name__ == "__main__":
    scout = StrategyScout()
    scout.run_batch()