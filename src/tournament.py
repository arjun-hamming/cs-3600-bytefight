import os, subprocess, json, math, sys, re, itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm 

# Add engine to path to allow direct import of play_game
sys.path.append(os.path.abspath("engine"))
from gameplay import play_game
from game.enums import ResultArbiter

# --- CONFIGURATION ---
AGENT_POOL_DIR = "3600-agents"
GAMES_PER_PAIR = 10
INITIAL_ELO = 1500  
K_FACTOR = 32       
MAX_WORKERS = max(1, os.cpu_count() - 2) 

# Cache absolute path
PLAY_DIRECTORY = os.path.abspath(AGENT_POOL_DIR)

# --- GLOBAL WORKER FUNCTION ---
# Kept outside the class to prevent heavy pickling delays.
# Removed stdout redirection so the engine's subprocesses don't choke.
def execute_match(task):
    agent_a, agent_b, output_dir = task
    try:
        final_board, _, _, _, _, _ = play_game(
            PLAY_DIRECTORY, output_dir, agent_a, agent_b,
            display_game=False, record=True, limit_resources=False
        )

        p_worker, o_worker = final_board.player_worker, final_board.opponent_worker
        a_pts = p_worker.get_points() if p_worker.is_player_a else o_worker.get_points()
        b_pts = o_worker.get_points() if p_worker.is_player_a else p_worker.get_points()

        result_val = final_board.winner.value if final_board.winner is not None else ResultArbiter.TIE.value
        
        match_data = {
            "agent_a": agent_a, "agent_b": agent_b,
            "a_pts": a_pts, "b_pts": b_pts,
            "result": result_val
        }

        # The tournament runner is responsible for saving the match summary, which is
        # then consumed by the `convert_matches.py` script.
        import uuid
        filename = f"match_{agent_a}_vs_{agent_b}_{uuid.uuid4().hex[:8]}.json"
        with open(os.path.join(output_dir, filename), "w") as f:
            json.dump(match_data, f)

        return match_data
    except Exception as e:
        return {"error": f"Match failed between {agent_a} and {agent_b}: {e}"}

class ELOTournament:
    def __init__(self):
        self.agents = [d for d in os.listdir(AGENT_POOL_DIR) 
                       if os.path.isdir(os.path.join(AGENT_POOL_DIR, d)) 
                       and os.path.exists(os.path.join(AGENT_POOL_DIR, d, "agent.py"))]
        
        self.elo_scores = {agent: INITIAL_ELO for agent in self.agents}
        self.stats = {agent: {"wins": 0, "losses": 0, "draws": 0} for agent in self.agents}
        
        self.matches_dir = self._get_next_season_dir()
        self.individual_dir = os.path.join(self.matches_dir, "individual")

        # Create both the season directory and the individual subdirectory
        if not os.path.exists(self.individual_dir):
            os.makedirs(self.individual_dir)
    def _get_next_season_dir(self) -> str:
        existing_seasons = []
        for d in os.listdir(AGENT_POOL_DIR):
            match = re.match(r"matches_season_(\d+)", d)
            if match:
                existing_seasons.append(int(match.group(1)))
        
        next_season = max(existing_seasons) + 1 if existing_seasons else 1
        return os.path.join(AGENT_POOL_DIR, f"matches_season_{next_season}")

    def calculate_expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))

    def update_elo(self, agent_a: str, agent_b: str, score_a: float):
        expected_a = self.calculate_expected_score(self.elo_scores[agent_a], self.elo_scores[agent_b])
        self.elo_scores[agent_a] += K_FACTOR * (score_a - expected_a)
        self.elo_scores[agent_b] += K_FACTOR * ((1.0 - score_a) - (1.0 - expected_a))

    def generate_pairings(self):
        for a, b in itertools.combinations(self.agents, 2):
            for _ in range(GAMES_PER_PAIR // 2):
                yield (a, b)
                yield (b, a)

    def run_match_batch(self):
        # We safely cast the generator to a list since 500 matches uses less than 1MB of RAM
        pairings = list(self.generate_pairings())
        total_matches = len(pairings)

        print(f"🌟 --- STARTING NEW TOURNAMENT SEASON --- 🌟")
        print(f"⚔️  Season Started: {self.matches_dir}")
        print(f"📊 {len(self.agents)} agents, {total_matches} total matches.")
        
        # Swapped back to ProcessPoolExecutor to allow the engine to spawn child processes safely
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            tasks = [(p[0], p[1], self.individual_dir) for p in pairings]
            futures = [executor.submit(execute_match, t) for t in tasks]
            
            for future in tqdm(as_completed(futures), total=total_matches, desc="Executing Matches", unit="game"):
                match_data = future.result()
                if "error" in match_data:
                    tqdm.write(match_data["error"])
                    continue

                agent_a, agent_b = match_data["agent_a"], match_data["agent_b"]
                winner = match_data["result"]

                # ELO & Stats updates
                if winner == ResultArbiter.PLAYER_A.value:
                    self.update_elo(agent_a, agent_b, 1.0)
                    self.stats[agent_a]["wins"] += 1; self.stats[agent_b]["losses"] += 1
                elif winner == ResultArbiter.PLAYER_B.value:
                    self.update_elo(agent_a, agent_b, 0.0)
                    self.stats[agent_b]["wins"] += 1; self.stats[agent_a]["losses"] += 1
                else: 
                    a_pts, b_pts = match_data["a_pts"], match_data["b_pts"]
                    if a_pts > b_pts:
                        self.update_elo(agent_a, agent_b, 1.0)
                        self.stats[agent_a]["wins"] += 1; self.stats[agent_b]["losses"] += 1
                    elif b_pts > a_pts:
                        self.update_elo(agent_a, agent_b, 0.0)
                        self.stats[agent_b]["wins"] += 1; self.stats[agent_a]["losses"] += 1
                    else:
                        self.update_elo(agent_a, agent_b, 0.5)
                        self.stats[agent_a]["draws"] += 1; self.stats[agent_b]["draws"] += 1

    def display_leaderboard(self):
        sorted_ranks = sorted(self.elo_scores.items(), key=lambda x: x[1], reverse=True)
        print("\n" + "="*50 + f"\nELO RANKINGS: {os.path.basename(self.matches_dir)}\n" + "-"*50)
        
        for i, (name, elo) in enumerate(sorted_ranks, 1):
            s = self.stats[name]
            print(f"{i:<3} {name:<18} | ELO: {int(elo):<5} | W/L/D: {s['wins']}/{s['losses']}/{s['draws']}")
        
        print("="*50 + "\n")
        
        with open(os.path.join(self.matches_dir, "leaderboard.json"), "w") as f:
            json.dump(self.elo_scores, f, indent=4)

if __name__ == "__main__":
    tourney = ELOTournament()
    tourney.run_match_batch()
    tourney.display_leaderboard()
