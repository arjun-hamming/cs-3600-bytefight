import sys
import os

# Add the root and src directories to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from tournament import ELOTournament, INITIAL_ELO
from convert_matches import consolidate_to_individual_jsons
#from analysis import StrategyScout
# Import modules to allow for monkey-patching
import convert_matches
#import analysis

def run_season():
    print("🌟 --- STARTING NEW TOURNAMENT SEASON --- 🌟")

    # Define the specific bots to include in the tournament
    target_bots = ["Bihar", "0_car", "old", "Yolanda", "0_car_mutate_0", "0_car_mutate_1", "0_car_mutate_2"]
    
    # Step 1: Run the Tournament
    # This creates the matches_season_X and the /individual subfolder
    tourney = ELOTournament()

    # Filter the tournament to only include the target bots
    # This ensures that pairings, ELO updates, and leaderboards are scoped to them.
    tourney.agents = [agent for agent in tourney.agents if agent in target_bots]
    tourney.elo_scores = {agent: INITIAL_ELO for agent in tourney.agents}
    tourney.stats = {agent: {"wins": 0, "losses": 0, "draws": 0} for agent in tourney.agents}

    tourney.run_match_batch()
    tourney.display_leaderboard()
    
    # Step 2: Consolidate Matches
    # This combines individual JSONs into agent summaries in /individual
    print("\n📦 --- CONSOLIDATING MATCH DATA --- 📦")
    # Monkey-patch the list of agents for the consolidation script
    convert_matches.TARGET_AGENTS = tourney.agents
    consolidate_to_individual_jsons()
    
    # Step 3: Run AI Analysis
    # This reads the summarized JSONs from the latest season's /individual folder
    #  print("\n🧠 --- STARTING AI STRATEGY SCOUT --- 🧠")
    # Monkey-patch the list of agents for the analysis script
    # analysis.TARGET_AGENTS = tourney.agents
    # scout = StrategyScout()
    # scout.run_batch()

    print("\n🏁 --- SEASON WORKFLOW COMPLETE --- 🏁")

if __name__ == "__main__":
    run_season()
