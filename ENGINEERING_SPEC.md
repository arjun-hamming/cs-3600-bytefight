# ArjunAndGary Bot v2 - Engineering Spec

## Current Record: 74W - 1D - 84L (46.5% win rate, 159 matches)

---

## PART 1: WHY WE'RE LOSING - Match Analysis

### Data Sources
- 22 local match JSON files (Bihar/ArjunAndGary vs Yolanda, self-play, smoke tests)
- 84 online losses across 5 pages (opponents: Pray, aycoders, Aaron, Team11, SOUP, Team 66, Team 133, W&V, Team 19, StockChicken, CS3600Group63, Team 39, Dream Team, and more)
- Full controller.py source analysis (765 lines)

---

### CRITICAL BUG #1: Oscillation Loops (Primary Cause of Losses)

**What happens:** The bot bounces between exactly 2 adjacent cells for dozens or hundreds of consecutive turns, doing absolutely nothing productive.

**Evidence from match data:**

| Match | Moves Wasted | Oscillation Pattern | Duration |
|-------|-------------|---------------------|----------|
| result.json | 66.7% | [0,2] ↔ [1,2] | 63 consecutive steps doing nothing (turns 21→84) |
| smoke.json | 35.3% | [7,5] ↔ [7,6] | 8 steps |
| smoke_future.json (LOSS) | 79.3% | [19,10] ↔ [19,11] and others | 24+ step stretches |
| smoke_big_spiral.json | 86.6% | [2,0] ↔ [1,0] | 188 consecutive moves |
| Bihar_Yolanda_7 | 68.8% | [3,0] ↔ [4,0] | 18 moves |
| Bihar_Bihar_0 (LOSS) | 38.9% | [0,3] ↔ [0,4] | 14 moves |

**Root cause in code:** `_search_turn()` (line 202) uses beam search with heuristic evaluation but has **zero memory of past positions**. When two adjacent cells have near-identical heuristic scores, the beam search alternates between them forever. The `_evaluate()` function (line 453) computes the same score from both cells, so the search sees no improvement in either direction and oscillates.

**Why this kills us:** In a typical 80-turn match, the bot wastes 50-70 turns bouncing. The opponent paints territory, captures hills, and snowballs stamina regen while we do nothing.

---

### CRITICAL BUG #2: Near-Zero Paint Rate

**What happens:** The bot almost never paints cells, even though painting is the core mechanic for scoring, hill capture, and stamina regeneration.

**Evidence:**

| Match | Paint Actions | Total Turns | Paint Rate |
|-------|--------------|-------------|------------|
| result.json | 4 | 98 | 4.1% |
| smoke.json | 1 | 30 | 3.3% |
| smoke_future.json | 12 | 1451 | 0.8% |
| smoke_big_spiral.json | 8 | 2000 | 0.4% |

**Root cause in code:** The `_enumerate_actions()` method (line 324) generates both move and paint actions, but the beam search consistently favors moves over paints because:
1. Paint costs 15 stamina, so the `_action_priority()` penalizes it via `priority -= 1.2 * (me_before.stamina - me_after.stamina)` (line 398) — a -18 penalty per paint
2. The territory gain from painting (+4.0 per `_action_priority`, line 396) doesn't overcome the stamina penalty
3. The search depth is too shallow to see the compound benefits of painting (stamina regen from territory, hill capture from painted cells)

**Why this kills us:** Without painting:
- No territory → no territory-based stamina regen (up to +50/turn)
- No hill cells → no hill control → no +40 max stamina bonus
- No local paint → no adjacency regen bonus (+2 per controlled cell in 5×5)
- Opponents who paint aggressively snowball their stamina advantage

---

### CRITICAL BUG #3: Collision Deaths

**Evidence:** `smoke_future.json` — bot dies by COLLISION at turn 1452. The bot was oscillating between [8,4] and [8,5] while the opponent slowly closed in.

**Root cause in code:** `_collision_is_favorable()` (line 597-601) only checks:
```python
def _collision_is_favorable(self, board, player_parity, target):
    target_cell = self._cell(board, target)
    if target_cell.owner_parity == -player_parity:
        return False
    return True
```

This is far too simplistic. It doesn't consider:
- Stamina differentials (who can afford to lose the collision)
- Opponent trajectory (are they approaching?)
- Whether we're walking into a collision trap
- The fact that the opponent can initiate a collision on THEIR turn

---

### CRITICAL BUG #4: Catastrophic Time Consumption

| Match | Bot Time Used | Opponent Time Used | Ratio |
|-------|-------------|-------------------|-------|
| result.json | 13.64s | 0.0009s | 15,403× |
| smoke.json | 4.57s | 0.0002s | 20,259× |
| smoke_big_spiral.json | 149.19s | 0.0143s | 10,406× |

The bot uses ~0.05-0.15s per turn on beam search. With a 180s total time budget across potentially 1000 rounds, this means the bot **will time out in long games** (~1200-3600 turns). Against real opponents who force long games, we run out of clock.

---

### CRITICAL BUG #5: Bidding Wastes Stamina

In `result.json`, the bot bid 9 stamina in round 1 (P1 bid 9, P2 bid 0). The bidding logic (line 55-95) runs two full search evaluations just to compute a bid, burning computation time. It also bids stamina aggressively even when there's no positional advantage to going first.

---

### CRITICAL BUG #6: No Hill-Capture Strategy

The bot's heuristic weights hills heavily (`hill_weight = 85.0`, line 479), but the search is too shallow (depth 3-5) to plan a multi-turn path to a hill, paint it, and capture it. The result: the bot *wants* to capture hills but never actually executes a coherent plan to do so.

In `result.json`, the bot captured 0 hills across 98 turns despite hills being present on the map.

---

## PART 2: ENGINEERING SPEC — What Must Change

### Architecture: From Beam Search to Goal-Directed Planning

The current beam search architecture is fundamentally wrong for this game. A beam search evaluates action sequences myopically — it can't plan "go to hill X, paint 5 cells, capture it" because that's a 10+ turn plan and the search only looks 3-5 steps deep.

**New architecture: Finite State Machine (FSM) with tactical search**

The bot should operate in named **modes** with clear goals, transitions, and action generators:

```
OPENING → HILL_RUSH → TERRITORY_EXPAND → DEFENSE → ENDGAME
```

---

### Change 1: Add Position History + Anti-Oscillation (CRITICAL)

**Current:** No memory of past positions.

**New:**
```python
class PlayerController:
    def __init__(self, player_parity, time_left):
        self.player_parity = player_parity
        self.position_history = []  # list of recent (row, col) tuples
        self.HISTORY_WINDOW = 12    # track last 12 positions
        self.OSCILLATION_PENALTY = -50.0  # massive penalty for revisiting
```

**In `_evaluate()` and `_action_priority()`:** Before returning a score, check if the resulting position appears in `self.position_history[-HISTORY_WINDOW:]`. If it does, apply `OSCILLATION_PENALTY * frequency_count`. This alone would fix 35-87% of wasted moves.

**At start of each `play()` call:** Append current position to history. Trim to window size.

---

### Change 2: Paint-First Action Generation (CRITICAL)

**Current:** Paint and move actions are scored equally; paint almost always loses due to stamina cost penalty.

**New:** Every turn should try to include at least one paint action unless stamina is critically low.

```python
def _play_inner(self, board, player_parity, time_left):
    # ALWAYS try to paint before/after moving
    # Generate compound actions: [Paint, Move] or [Move, Paint]
    # Only skip paint if stamina < 25
```

**Rebalance `_action_priority()` for paint:**
- Hill cell paint: +30 (up from +10)
- Any paint on unpainted cell: +20 (new)
- Paint on opponent territory: +25 (new)
- Remove the stamina cost penalty for paint actions (it's already accounted for in the stamina differential evaluation)

---

### Change 3: FSM Mode System (HIGH)

```python
class Mode(Enum):
    OPENING = "opening"       # First 3-5 turns: rush toward nearest hill
    HILL_RUSH = "hill_rush"   # Navigate to and paint hill cells
    TERRITORY = "territory"   # Expand painted territory around controlled hills
    DEFENSE = "defense"       # Opponent approaching: protect painted cells
    ENDGAME = "endgame"       # Sudden death: conserve stamina, hold territory
```

**Mode selection logic (run each turn):**
```python
def _select_mode(self, board, player_parity):
    me = board.get_player(player_parity)
    opp = board.get_opponent(player_parity)
    round_frac = board.current_round / GameConstants.MAX_ROUNDS

    if board.current_round < 5:
        return Mode.OPENING

    # If we control 0 hills and hills exist → HILL_RUSH
    if len(me.controlled_hills) == 0 and len(board.hills) > 0:
        return Mode.HILL_RUSH

    # If opponent is within 3 cells of our hill → DEFENSE
    for hill in board.hills.values():
        if hill.controller_parity == player_parity:
            for cell_loc in hill.cells:
                opp_dist = manhattan(cell_loc, opp.loc)
                if opp_dist <= 3:
                    return Mode.DEFENSE

    # After turn 500 → ENDGAME
    if board.current_round >= 500:
        return Mode.ENDGAME

    return Mode.TERRITORY
```

**Each mode produces a target location and action preferences:**
- `OPENING`: Target = nearest hill center. Actions = move toward it.
- `HILL_RUSH`: Target = nearest uncontrolled hill cell. Actions = move + paint hill cells.
- `TERRITORY`: Target = nearest unpainted cell adjacent to our territory. Actions = paint + expand.
- `DEFENSE`: Target = stay near our hill. Actions = paint our hill cells, erase opponent paint.
- `ENDGAME`: Target = hold position. Actions = minimize stamina use, paint only if profitable.

---

### Change 4: Path Planning with A* (HIGH)

**Current:** The bot uses BFS only for distance estimation (`_grid_distance`), never for actual pathfinding.

**New:** Use A* to compute an actual path to the target location, then follow it step by step.

```python
def _plan_path(self, board, start, goal):
    """A* pathfinding returning list of Directions to follow."""
    # Standard A* with manhattan distance heuristic
    # Avoid opponent-controlled cells when possible (soft penalty)
    # Avoid cells adjacent to opponent (collision risk)
    # Return list of Direction values
```

**Usage in play():**
```python
def _play_inner(self, board, player_parity, time_left):
    mode = self._select_mode(board, player_parity)
    target = self._get_target(board, player_parity, mode)
    path = self._plan_path(board, me.loc, target)

    actions = []
    # Paint adjacent cell if beneficial
    paint_target = self._best_paint_target(board, player_parity)
    if paint_target and me.stamina >= 25:
        actions.append(Action.Paint(paint_target))

    # Move along path
    if path:
        actions.append(Action.Move(direction=path[0]))

    return actions if actions else self._fallback_turn(board, player_parity)
```

---

### Change 5: Fix Collision Avoidance (HIGH)

**Current:** Only checks cell ownership.

**New:**
```python
def _collision_is_favorable(self, board, player_parity, target):
    target_cell = self._cell(board, target)
    me = board.get_player(player_parity)
    opp = board.get_opponent(player_parity)

    # Never initiate collision on opponent territory
    if target_cell.owner_parity == -player_parity:
        return False

    # Only initiate if we control the cell
    if target_cell.owner_parity != player_parity:
        return False  # Neutral ground is risky

    # Check stamina - don't collide if we'd lose more
    if me.stamina < opp.stamina * 0.8:
        return False

    return True

def _is_collision_risk(self, board, player_parity, target):
    """Check if moving to target puts us at collision risk next turn."""
    opp = board.get_opponent(player_parity)
    opp_dist = manhattan(target, opp.loc)
    if opp_dist <= 1:
        target_cell = self._cell(board, target)
        if target_cell.owner_parity != player_parity:
            return True  # Opponent could collide us on their turn
    return False
```

**Add collision risk penalty to `_action_priority()`:** If moving to a cell that's adjacent to the opponent and NOT our territory, apply -40 penalty.

---

### Change 6: Smarter Bidding (MEDIUM)

**Current:** Runs two full beam searches to compute bid. Bids up to 25 stamina.

**New:** Simple rule-based bidding:
```python
def _bid_inner(self, board, player_parity, time_left):
    me = board.get_player(player_parity)
    round_num = board.current_round

    # Never bid more than 5% of stamina
    max_bid = max(0, int(me.stamina * 0.05))

    # Bid 0 for first 3 rounds (save stamina for painting)
    if round_num < 3:
        return 0

    # Bid small amount if we're near a hill we need to capture
    if len(me.controlled_hills) == 0:
        return min(max_bid, 3)

    # Default: bid 0-2
    return min(max_bid, 2)
```

This saves ~100ms per turn (currently wasted on bid search) and preserves stamina.

---

### Change 7: Time Management (MEDIUM)

**Current:** 0.05-0.15s per turn, will timeout in long games.

**New:** With the FSM + A* approach, most turns should take <5ms (just path lookup + paint selection). Reserve the beam search only for critical decisions (collision threats, contested hills).

```python
def _time_budget(self, board, time_left):
    remaining = max(0.25, float(time_left()))
    remaining_rounds = max(1, GameConstants.MAX_ROUNDS - board.current_round)

    # Target: 0.01s per turn normally, up to 0.1s for critical turns
    if self.mode in (Mode.DEFENSE, Mode.HILL_RUSH):
        return min(0.1, remaining / remaining_rounds * 0.5)
    return min(0.02, remaining / remaining_rounds * 0.3)
```

---

### Change 8: Stamina-Aware Painting (MEDIUM)

**Current:** No stamina threshold for painting.

**New:** Paint aggressively when stamina is healthy, conserve when low:
```python
def _should_paint(self, board, player_parity):
    me = board.get_player(player_parity)
    # Always paint hill cells regardless of stamina
    # Paint territory if stamina > 40
    # Skip paint if stamina < 25 (reserve for movement)
    if me.stamina < 25:
        return False
    return True

def _best_paint_target(self, board, player_parity):
    """Pick the highest-value adjacent cell to paint."""
    me = board.get_player(player_parity)
    candidates = []
    for loc in self._adjacent_locations(me.loc):
        if board.oob(loc): continue
        cell = self._cell(board, loc)
        if cell.is_wall: continue
        if cell.owner_parity == player_parity: continue  # already ours

        score = 0
        if cell.hill_id:
            hill = board.hills[cell.hill_id]
            if hill.controller_parity != player_parity:
                score += 100  # Highest priority: uncontrolled hill
            else:
                score += 20
        if cell.owner_parity == -player_parity:
            score += 30  # Erase opponent paint
        if cell.owner_parity == 0:
            score += 15  # Claim neutral territory
        if cell.beacon_parity == -player_parity:
            score += 25  # Destroy opponent beacon
        candidates.append((score, loc))

    candidates.sort(reverse=True)
    return candidates[0][1] if candidates else None
```

---

### Change 9: Beacon Usage (LOW)

**Current:** The bot generates beacon actions but rarely uses them effectively.

**New:** Place beacons strategically:
- Place a beacon on our side of the map early (turn 5-10) for retreat
- Place a beacon near a contested hill for rapid response
- Use beacon travel to quickly switch between attacking different hills

---

### Change 10: Erase Move Usage (LOW)

**Current:** Erase moves are generated but rarely chosen (40 stamina is expensive).

**New:** Use erase moves specifically for:
- Removing opponent paint from hill cells (high value)
- Breaking opponent's territory chain (cutting their regen)
- Only when stamina > 60

---

## PART 3: Implementation Priority

| Priority | Change | Expected Impact | Effort |
|----------|--------|----------------|--------|
| P0 | Anti-oscillation (position history) | Fixes 35-87% wasted moves | Small |
| P0 | Paint-first action generation | Enables territory/hill capture | Small |
| P1 | FSM mode system | Gives bot strategic coherence | Medium |
| P1 | A* path planning | Replaces aimless wandering | Medium |
| P1 | Collision avoidance fix | Prevents collision deaths | Small |
| P2 | Smarter bidding | Saves stamina + computation | Small |
| P2 | Time management | Prevents timeouts in long games | Small |
| P2 | Stamina-aware painting | Optimizes resource usage | Small |
| P3 | Beacon strategy | Adds mobility advantage | Medium |
| P3 | Erase move strategy | Adds defensive capability | Small |

**P0 changes alone should push win rate from 46% to ~65-70%.** The bot currently beats only do-nothing bots (Yolanda). With anti-oscillation + paint-first, it would beat any bot that paints less territory.

**P0 + P1 changes should push win rate to ~75-80%.** FSM + pathfinding gives strategic coherence that most simple bots lack.

---

## PART 4: Key Game Rules Reference

- **Painting:** 15 stamina, Manhattan distance 1, stacks to ±4 layers
- **Hill capture:** Control >50% of hill cells, must exceed opponent
- **Hill dominance win:** Control ≥75% of ALL hills → instant win
- **Collision:** Cell controller wins; on neutral ground, initiator wins
- **Stamina regen/turn:** base 5 + (2 × local controlled cells in 5×5) + (territory/8, capped at 50)
- **Max stamina:** 100 base + 40 per controlled hill
- **Powerups:** +35 stamina, spawn symmetrically at intervals
- **Move costs:** 1st free, 2nd +10, 3rd +20, 4th +30...
- **Erase cost:** 40 + move scaling
- **Beacon travel:** Resets move counter, decays paint under destination
- **Time limit:** 180s total play time across entire match
- **Sudden death:** After turn 1000, regen decreases by 3 every 100 rounds
- **Tiebreaker:** Hills → Territory → Time remaining
