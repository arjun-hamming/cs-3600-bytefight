"""Bihar - CS3600 Spring 2026 tournament agent.

Architecture:
- HMM belief tracker over the rat (numpy vector of length 64).
- Negamax search with alpha-beta pruning over physical moves.
- Search moves (rat guesses) considered at the root and folded in as an
  expected-value action alongside the top physical move.
- Heuristic: score differential + carpet-roll potential of primed runs +
  worker proximity to primes + rat-knowledge bonus.

The bot was designed from scratch for this game, inspired architecturally
by a ByteFight paint-game bot ("biharbot"). The two games are different,
so this is a clean implementation that reuses only the engineering
playbook (minimax + HMM + rich heuristic).
"""

from collections.abc import Callable
from typing import List, Optional, Tuple

import numpy as np

from game import board as _board_mod  # noqa: F401 - side effect import
from game.board import Board
from game.enums import (
    BOARD_SIZE,
    CARPET_POINTS_TABLE,
    Cell,
    Direction,
    MoveType,
    Noise,
)
from game.move import Move


# ---------------------------------------------------------------------------
# Constants and small helpers
# ---------------------------------------------------------------------------

N_CELLS = BOARD_SIZE * BOARD_SIZE
HEADSTART_MOVES = 1000

# Emission model (must match game/rat.py)
NOISE_PROBS = {
    Cell.BLOCKED: (0.5, 0.3, 0.2),
    Cell.SPACE: (0.7, 0.15, 0.15),
    Cell.PRIMED: (0.1, 0.8, 0.1),
    Cell.CARPET: (0.1, 0.1, 0.8),
}

# Distance emission: observed = clip(actual + offset, 0, inf)
DIST_ERROR_OFFSETS = (-1, 0, 1, 2)
DIST_ERROR_PROBS = (0.12, 0.7, 0.12, 0.06)

INF = 10 ** 9


def _pos_to_idx(x: int, y: int) -> int:
    return y * BOARD_SIZE + x


def _idx_to_pos(i: int) -> Tuple[int, int]:
    return (i % BOARD_SIZE, i // BOARD_SIZE)


def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# Precompute Manhattan distance tables for every worker position once.
# _DIST_TABLE[worker_idx, cell_idx] = manhattan distance.
_DIST_TABLE = np.zeros((N_CELLS, N_CELLS), dtype=np.int16)
for _wi in range(N_CELLS):
    _wx, _wy = _idx_to_pos(_wi)
    for _ci in range(N_CELLS):
        _cx, _cy = _idx_to_pos(_ci)
        _DIST_TABLE[_wi, _ci] = abs(_wx - _cx) + abs(_wy - _cy)

# Precompute distance-likelihood table: _DIST_LL[actual, observed]
_MAX_DIST = 2 * (BOARD_SIZE - 1)
_DIST_LL = np.zeros((_MAX_DIST + 1, _MAX_DIST + 4), dtype=np.float64)
for _actual in range(_MAX_DIST + 1):
    for _off, _p in zip(DIST_ERROR_OFFSETS, DIST_ERROR_PROBS):
        _val = _actual + _off
        if _val < 0:
            _val = 0
        if _val <= _MAX_DIST + 3:
            _DIST_LL[_actual, _val] += _p


def _dist_likelihood(actual: int, observed: int) -> float:
    if observed < 0:
        return 0.0
    if observed > _MAX_DIST + 3:
        return 0.0
    return float(_DIST_LL[actual, observed])


# ---------------------------------------------------------------------------
# Rat belief tracker (HMM)
# ---------------------------------------------------------------------------


class RatTracker:
    """Maintains a posterior belief over the rat's current cell."""

    def __init__(self, T_raw) -> None:
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            T = np.array(T_raw, dtype=np.float64, copy=True)
            T = np.nan_to_num(T, nan=0.0, posinf=0.0, neginf=0.0)
            T = np.maximum(T, 0.0)
            # Numerical hygiene: ensure rows sum to 1
            row_sums = T.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums <= 1e-18, 1.0, row_sums)
            self.T = np.ascontiguousarray(T / row_sums, dtype=np.float64)

            # post_spawn_belief = delta_(0,0) @ T^HEADSTART_MOVES
            v = np.zeros(N_CELLS, dtype=np.float64)
            v[0] = 1.0
            for _ in range(HEADSTART_MOVES):
                nv = np.dot(v, self.T)
                nv = np.nan_to_num(nv, nan=0.0, posinf=0.0, neginf=0.0)
                nv = np.maximum(nv, 0.0)
                s = nv.sum()
                if not np.isfinite(s) or s < 1e-18:
                    nv = np.full(N_CELLS, 1.0 / N_CELLS, dtype=np.float64)
                else:
                    nv = nv / s
                v = nv
            # Precompute the belief as of the moment T^1001 (one rat.move past spawn).
            self.post_spawn_belief = v
            pp = np.dot(v, self.T)
            pp = np.nan_to_num(pp, nan=0.0, posinf=0.0, neginf=0.0)
            pp = np.maximum(pp, 0.0)
            sp = pp.sum()
            if not np.isfinite(sp) or sp < 1e-18:
                pp = np.full(N_CELLS, 1.0 / N_CELLS, dtype=np.float64)
            else:
                pp = pp / sp
            self.post_spawn_plus_one = pp

        # Running belief: posterior at the time of my most recent observation.
        self.belief: Optional[np.ndarray] = None
        self.last_turn_count: Optional[int] = None

    # ------------------------------------------------------------------
    # Observation update
    # ------------------------------------------------------------------
    def _sanitize(self, v: np.ndarray) -> np.ndarray:
        """Replace NaN/inf with 0 and renormalize; fall back to uniform if empty."""
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        v = np.maximum(v, 0.0)
        s = v.sum()
        if not np.isfinite(s) or s < 1e-12:
            return np.full(N_CELLS, 1.0 / N_CELLS, dtype=np.float64)
        return v / s

    def _observation_update(
        self,
        belief: np.ndarray,
        board: Board,
        worker_pos: Tuple[int, int],
        noise: int,
        distance: int,
    ) -> np.ndarray:
        # Build noise likelihood using board cell types.
        cell_types = np.empty(N_CELLS, dtype=np.int8)
        for ci in range(N_CELLS):
            cell_types[ci] = int(board.get_cell(_idx_to_pos(ci)))

        noise_ll = np.empty(N_CELLS, dtype=np.float64)
        for ci in range(N_CELLS):
            probs = NOISE_PROBS[Cell(cell_types[ci])]
            noise_ll[ci] = probs[noise]

        worker_idx = _pos_to_idx(*worker_pos)
        dist_row = _DIST_TABLE[worker_idx]
        obs = int(distance)
        dist_ll = np.empty(N_CELLS, dtype=np.float64)
        for ci in range(N_CELLS):
            dist_ll[ci] = _dist_likelihood(int(dist_row[ci]), obs)

        belief = self._sanitize(belief)
        post = belief * noise_ll * dist_ll
        return self._sanitize(post)

    # ------------------------------------------------------------------
    # Belief maintenance across turns
    # ------------------------------------------------------------------
    def update_for_my_turn(
        self,
        board: Board,
        noise: int,
        distance: int,
        worker_pos: Tuple[int, int],
        current_turn_count: int,
        opp_search: Tuple,
        player_search: Tuple,
    ) -> None:
        opp_loc, opp_result = opp_search
        my_prev_loc, my_prev_result = player_search

        opp_caught = (opp_loc is not None) and bool(opp_result)
        my_caught = (my_prev_loc is not None) and bool(my_prev_result)

        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            if self.belief is None:
                # First play call - rat has moved HEADSTART + (turn_count + 1) times total.
                if current_turn_count == 0:
                    v = self.post_spawn_plus_one.copy()  # T^1001
                else:
                    v = self.post_spawn_belief.copy()
                    for _ in range(current_turn_count + 1):
                        v = self._sanitize(np.dot(v, self.T))
                self.belief = self._sanitize(v)
            else:
                if opp_caught:
                    # Rat respawned on the opponent's turn (last). Then my rat.move() ran.
                    self.belief = self.post_spawn_plus_one.copy()
                elif my_caught:
                    v = self.post_spawn_plus_one.copy()
                    if opp_loc is not None:
                        vi = _pos_to_idx(*opp_loc)
                        v[vi] = 0.0
                        v = self._sanitize(v)
                    v = self._sanitize(np.dot(v, self.T))
                    self.belief = v
                else:
                    cur = self._sanitize(self.belief)
                    if my_prev_loc is not None and not my_prev_result:
                        vi = _pos_to_idx(*my_prev_loc)
                        cur = cur.copy()
                        cur[vi] = 0.0
                        cur = self._sanitize(cur)
                    v = self._sanitize(np.dot(cur, self.T))
                    if opp_loc is not None:
                        vi = _pos_to_idx(*opp_loc)
                        v[vi] = 0.0
                        v = self._sanitize(v)
                    v = self._sanitize(np.dot(v, self.T))
                    self.belief = v

            self.belief = self._observation_update(
                self.belief, board, worker_pos, noise, distance
            )
            self.last_turn_count = current_turn_count

    def max_belief(self) -> float:
        if self.belief is None:
            return 0.0
        return float(self.belief.max())

    def argmax(self) -> int:
        if self.belief is None:
            return 0
        return int(np.argmax(self.belief))

    def top_k(self, k: int) -> List[int]:
        if self.belief is None:
            return []
        k = max(1, min(k, N_CELLS))
        idxs = np.argpartition(-self.belief, k - 1)[:k]
        # sort descending by probability
        idxs = idxs[np.argsort(-self.belief[idxs])]
        return idxs.tolist()


# ---------------------------------------------------------------------------
# Heuristic
# ---------------------------------------------------------------------------


def _carpet_potential(board: Board) -> float:
    """Sum of CARPET_POINTS_TABLE[k] for maximal runs of primed cells."""
    primed = board._primed_mask
    total = 0.0

    # Horizontal runs
    for y in range(BOARD_SIZE):
        run = 0
        for x in range(BOARD_SIZE):
            bit = 1 << (y * BOARD_SIZE + x)
            if primed & bit:
                run += 1
            else:
                if run >= 2:
                    total += CARPET_POINTS_TABLE.get(min(run, 7), 0)
                run = 0
        if run >= 2:
            total += CARPET_POINTS_TABLE.get(min(run, 7), 0)

    # Vertical runs
    for x in range(BOARD_SIZE):
        run = 0
        for y in range(BOARD_SIZE):
            bit = 1 << (y * BOARD_SIZE + x)
            if primed & bit:
                run += 1
            else:
                if run >= 2:
                    total += CARPET_POINTS_TABLE.get(min(run, 7), 0)
                run = 0
        if run >= 2:
            total += CARPET_POINTS_TABLE.get(min(run, 7), 0)

    return total


def _dist_to_nearest_prime(board: Board, worker_pos: Tuple[int, int]) -> int:
    primed = board._primed_mask
    if primed == 0:
        return 0
    wx, wy = worker_pos
    best = BOARD_SIZE * 2
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            bit = 1 << (y * BOARD_SIZE + x)
            if primed & bit:
                d = abs(x - wx) + abs(y - wy)
                if d < best:
                    best = d
    return best if best < BOARD_SIZE * 2 else 0


def _cell_potential(board: Board, worker_pos: Tuple[int, int]) -> float:
    """
    "Potential" of each cell = the longest contiguous run of non-blocked,
    non-carpet cells it belongs to in either horizontal or vertical axis.
    Weighted by CARPET_POINTS_TABLE[run] to reflect future carpet value, and
    by inverse Manhattan distance from the worker (closer = more useful).
    """
    blocked = board._blocked_mask
    carpet = board._carpet_mask
    unavail = blocked | carpet  # cells that can never become a prime
    wx, wy = worker_pos
    total = 0.0

    # Horizontal max-run per cell
    h_run = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    for y in range(BOARD_SIZE):
        run = 0
        segs: List[Tuple[int, int]] = []  # (x_start, length)
        for x in range(BOARD_SIZE):
            bit = 1 << (y * BOARD_SIZE + x)
            if unavail & bit:
                if run > 0:
                    segs.append((x - run, run))
                run = 0
            else:
                run += 1
        if run > 0:
            segs.append((BOARD_SIZE - run, run))
        for xs, L in segs:
            for x in range(xs, xs + L):
                h_run[y][x] = L

    # Vertical max-run per cell
    v_run = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    for x in range(BOARD_SIZE):
        run = 0
        segs = []
        for y in range(BOARD_SIZE):
            bit = 1 << (y * BOARD_SIZE + x)
            if unavail & bit:
                if run > 0:
                    segs.append((y - run, run))
                run = 0
            else:
                run += 1
        if run > 0:
            segs.append((BOARD_SIZE - run, run))
        for ys, L in segs:
            for y in range(ys, ys + L):
                v_run[y][x] = L

    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            bit = 1 << (y * BOARD_SIZE + x)
            if unavail & bit:
                continue
            run = max(h_run[y][x], v_run[y][x])
            if run < 2:
                continue
            val = CARPET_POINTS_TABLE.get(min(run, 7), 0)
            d = abs(x - wx) + abs(y - wy)
            total += val / (1.0 + 0.6 * d)
    return total


def heuristic(board: Board, belief_max_p: float) -> float:
    """Evaluate the position from the perspective of board.player_worker."""
    me = board.player_worker
    opp = board.opponent_worker

    # Score differential (primary driver)
    score = 5.0 * (me.get_points() - opp.get_points())

    # Carpet potential: primes in lines => huge future value
    score += 0.55 * _carpet_potential(board)

    # Cell potential: future prime-line opportunities weighted by proximity.
    # This is the key ingredient behind Carrie's 90% advanced heuristic.
    score += 0.22 * _cell_potential(board, me.get_location())
    # Opponent's future potential is a mild negative for us.
    score -= 0.12 * _cell_potential(board, opp.get_location())

    # Proximity to my primes (to roll them). Cheap, only matters when there are primes.
    if board._primed_mask:
        d = _dist_to_nearest_prime(board, me.get_location())
        score -= 0.35 * d

    # Mobility: more legal moves is better (prevents getting cornered)
    try:
        n_moves = len(board.get_valid_moves(enemy=False, exclude_search=True))
    except Exception:
        n_moves = 4
    score += 0.15 * min(n_moves, 20)

    # Rat knowledge: being able to search profitably is valuable.
    search_ev = 6.0 * belief_max_p - 2.0
    if search_ev > 0:
        score += 1.5 * search_ev

    return score


# ---------------------------------------------------------------------------
# Searcher (negamax with alpha-beta)
# ---------------------------------------------------------------------------


def _move_priority(m: Move) -> Tuple[int, int]:
    """Sort key for move ordering (higher = try first)."""
    if m.move_type == MoveType.CARPET:
        roll = m.roll_length
        pts = CARPET_POINTS_TABLE.get(min(roll, 7), 0)
        return (3, pts)
    if m.move_type == MoveType.PRIME:
        return (2, 0)
    if m.move_type == MoveType.PLAIN:
        return (1, 0)
    return (0, 0)


def _filter_moves(moves: List[Move]) -> List[Move]:
    """Drop strictly-bad moves (e.g. 1-length carpet rolls) and sort."""
    out: List[Move] = []
    for m in moves:
        if m.move_type == MoveType.CARPET and m.roll_length < 2:
            continue
        out.append(m)
    out.sort(key=_move_priority, reverse=True)
    return out


class Negamax:
    def __init__(self, time_budget_s: float, get_time_left: Callable[[], float]) -> None:
        self.time_budget_s = time_budget_s
        self.get_time_left = get_time_left
        self.start_time: float = 0.0
        self.deadline: float = 0.0
        self.aborted = False

    def _time_up(self) -> bool:
        if self.aborted:
            return True
        try:
            remaining = self.get_time_left()
        except Exception:
            remaining = 5.0
        if remaining < 0.5:
            self.aborted = True
            return True
        # Also respect per-move budget
        if self.deadline and _perf() >= self.deadline:
            self.aborted = True
            return True
        return False

    def negamax(
        self,
        board: Board,
        depth: int,
        alpha: float,
        beta: float,
        belief_max_p: float,
    ) -> float:
        if depth == 0 or board.is_game_over():
            return heuristic(board, belief_max_p)
        if self._time_up():
            return heuristic(board, belief_max_p)

        moves = _filter_moves(board.get_valid_moves(exclude_search=True))
        if not moves:
            return heuristic(board, belief_max_p)

        best = -float("inf")
        for m in moves:
            next_board = board.forecast_move(m, check_ok=False)
            if next_board is None:
                continue
            next_board.reverse_perspective()
            val = -self.negamax(next_board, depth - 1, -beta, -alpha, belief_max_p)
            if val > best:
                best = val
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
            if self._time_up():
                break
        if best == -float("inf"):
            return heuristic(board, belief_max_p)
        return best

    def best_move_iddfs(
        self,
        board: Board,
        max_depth: int,
        belief_max_p: float,
    ) -> Tuple[Optional[Move], float]:
        """Iterative-deepening search over physical moves with root PV ordering."""
        moves = _filter_moves(board.get_valid_moves(exclude_search=True))
        if not moves:
            return None, -float("inf")

        # Seed root ordering with a shallow heuristic eval on each child.
        root_scores: List[Tuple[float, Move]] = []
        for m in moves:
            nb = board.forecast_move(m, check_ok=False)
            if nb is None:
                continue
            nb.reverse_perspective()
            # -h(child) since child is from opponent's perspective
            root_scores.append((-heuristic(nb, belief_max_p), m))
        root_scores.sort(key=lambda t: t[0], reverse=True)
        ordered_moves = [m for _, m in root_scores]
        if not ordered_moves:
            return moves[0], -float("inf")

        best_move = ordered_moves[0]
        best_score = -float("inf")

        self.start_time = _perf()
        self.deadline = self.start_time + self.time_budget_s
        self.aborted = False

        for depth in range(1, max_depth + 1):
            if self._time_up():
                break
            current_best_move = best_move
            current_best_score = -float("inf")
            alpha = -float("inf")
            beta = float("inf")
            # Re-order so previously-best move is tried first.
            ordered = [best_move] + [m for m in ordered_moves if m is not best_move]
            new_scored: List[Tuple[float, Move]] = []
            for m in ordered:
                next_board = board.forecast_move(m, check_ok=False)
                if next_board is None:
                    continue
                next_board.reverse_perspective()
                val = -self.negamax(
                    next_board, depth - 1, -beta, -alpha, belief_max_p
                )
                new_scored.append((val, m))
                if val > current_best_score:
                    current_best_score = val
                    current_best_move = m
                if current_best_score > alpha:
                    alpha = current_best_score
                if self._time_up():
                    break
            if not self.aborted:
                best_move = current_best_move
                best_score = current_best_score
                # Use this depth's scores to seed next depth's ordering.
                new_scored.sort(key=lambda t: t[0], reverse=True)
                ordered_moves = [m for _, m in new_scored]
            else:
                break

        return best_move, best_score


def _perf() -> float:
    import time

    return time.perf_counter()


# ---------------------------------------------------------------------------
# PlayerAgent
# ---------------------------------------------------------------------------


class PlayerAgent:
    def __init__(self, board: Board, transition_matrix=None, time_left: Callable = None):
        self.rat = RatTracker(transition_matrix) if transition_matrix is not None else None
        self._last_turn_count = -1
        # Remember cells I've searched and missed within the last few turns,
        # so I don't blindly re-guess the same cell while belief diffuses back.
        # Map: (x, y) -> last turn_count at which we missed there.
        self._recent_misses: dict = {}

    def commentate(self) -> str:
        return "Bihar out."

    # ------------------------------------------------------------------
    def _budget_for_turn(self, board: Board) -> float:
        turns_left = max(1, board.player_worker.turns_left)
        time_left = max(0.5, board.player_worker.time_left)
        # Use ~85% of the average time slice for this move, leave headroom.
        return min(5.5, 0.85 * time_left / turns_left)

    def _search_ev_bonus(self) -> float:
        """Expected immediate points from searching the single best cell."""
        if self.rat is None or self.rat.belief is None:
            return -2.0
        return 6.0 * self.rat.max_belief() - 2.0

    def play(
        self,
        board: Board,
        sensor_data: Tuple,
        time_left: Callable,
    ) -> Move:
        try:
            return self._play_impl(board, sensor_data, time_left)
        except Exception:
            # Failsafe: never crash; return any legal move.
            try:
                moves = board.get_valid_moves(exclude_search=True)
                for m in moves:
                    if not (m.move_type == MoveType.CARPET and m.roll_length < 2):
                        return m
                if moves:
                    return moves[0]
            except Exception:
                pass
            return Move.plain(Direction.RIGHT)

    def _play_impl(
        self,
        board: Board,
        sensor_data: Tuple,
        time_left: Callable,
    ) -> Move:
        noise_raw, distance = sensor_data
        noise_idx = int(noise_raw)

        worker_pos = board.player_worker.get_location()

        # Update HMM belief
        if self.rat is not None:
            self.rat.update_for_my_turn(
                board=board,
                noise=noise_idx,
                distance=int(distance),
                worker_pos=worker_pos,
                current_turn_count=board.turn_count,
                opp_search=board.opponent_search,
                player_search=board.player_search,
            )

        belief_max = self.rat.max_belief() if self.rat is not None else 0.0

        # Budget time for this turn
        budget = self._budget_for_turn(board)

        # Run iterative-deepening negamax on physical moves.
        searcher = Negamax(time_budget_s=budget, get_time_left=time_left)
        max_depth = 6
        best_physical_move, best_physical_score = searcher.best_move_iddfs(
            board, max_depth=max_depth, belief_max_p=belief_max
        )

        # Score a root-level search action as well.
        # Approximation: post-search board state is identical (no worker move), so
        # the opponent reply value ~= the value we'd get playing a "pass" (which
        # doesn't exist), hence we estimate continuation via one level of opponent
        # negamax starting from a board where we pretend we searched (turn advanced).
        search_action: Optional[Move] = None
        search_score = -float("inf")
        if self.rat is not None and self.rat.belief is not None:
            # Record if my previous search missed (via board.player_search).
            my_prev_loc, my_prev_result = board.player_search
            if my_prev_loc is not None and not my_prev_result:
                self._recent_misses[tuple(my_prev_loc)] = board.turn_count
            # Forget misses older than 3 turns.
            self._recent_misses = {
                loc: t
                for loc, t in self._recent_misses.items()
                if board.turn_count - t <= 3
            }

            # Pick the highest-probability cell that has NOT been recently missed.
            top_candidates = self.rat.top_k(6)
            chosen_idx = -1
            chosen_p = 0.0
            for idx in top_candidates:
                loc = _idx_to_pos(idx)
                if loc in self._recent_misses:
                    continue
                chosen_idx = idx
                chosen_p = float(self.rat.belief[idx])
                break

            # Only consider searching when EV is clearly positive.
            # 6p - 2 > 0  <=>  p > 1/3. Use 0.44 for a safety margin.
            if chosen_idx >= 0 and chosen_p > 0.44:
                top_loc = _idx_to_pos(chosen_idx)
                p = chosen_p
                search_move = Move.search(top_loc)
                # Forecast a search: the board's apply_move treats SEARCH as a no-op
                # except for end_turn() advancing the counter and turns_left.
                next_board = board.forecast_move(search_move, check_ok=False)
                if next_board is not None:
                    next_board.reverse_perspective()
                    # Evaluate opponent's best reply (our continuation value).
                    cont_val = -searcher.negamax(
                        next_board, max(0, max_depth - 1), -float("inf"), float("inf"),
                        belief_max,
                    )
                    # Immediate expected reward from the guess.
                    ev_reward = 6.0 * p - 2.0  # 4p - 2(1-p)
                    # Scale by the heuristic's points weight (5.0) for apples-to-apples.
                    search_score = cont_val + 5.0 * ev_reward
                    search_action = search_move

        # Compare and pick the winner.
        if best_physical_move is None and search_action is None:
            # No legal non-carpet-1 move found - fall back to any legal move.
            moves = board.get_valid_moves(exclude_search=True)
            return moves[0] if moves else Move.plain(Direction.RIGHT)

        if search_action is not None and search_score > best_physical_score + 0.01:
            return search_action

        return best_physical_move if best_physical_move is not None else search_action
