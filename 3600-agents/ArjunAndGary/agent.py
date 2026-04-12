"""
ArjunAndGary - CS3600 Spring 2026 Tournament Agent
==================================================

This is the tournament submission for team ArjunAndGary. It plays the 8x8
"carpet / rat" game where two workers compete over 40 turns each to prime
squares, roll them up into scoring carpet lines, and catch a hidden rat that
moves according to a game-specific transition matrix.

High-level architecture
-----------------------
1. Hidden Markov Model (HMM) for rat localization:
   - A 64-dim probability vector `belief[i]` = P(rat currently in cell i).
   - Transition update uses the provided 64x64 row-stochastic matrix T.
   - Observation update combines a noise-type likelihood and a noisy-distance
     likelihood (both provided by the rules).
   - Belief is carefully rolled forward through both my and my opponent's
     rat moves between my consecutive turns, and respawns are handled
     explicitly when either side successfully caught the rat.

2. Negamax alpha-beta search over *physical* moves (plain/prime/carpet):
   - Iterative deepening so we always return something sensible even if we
     run out of time at a deeper depth.
   - Root move ordering by a shallow heuristic eval of each child and then
     re-ordered by prior-depth score (principal variation first).
   - A separate continuation estimate for a "search the rat" action at the
     root is compared against the best physical move using an apples-to-apples
     points-weighted expected value.

3. Evaluation function:
   - Score differential is the biggest weight.
   - Primed-line score recognises 2+ length primed runs (they can become high
     value carpets on the next roll).
   - Open-line reach rewards empty-corridor potential near my worker and
     penalizes it near the opponent's worker.
   - Rat knowledge bonus: a positive-EV search is worth something.
   - Mobility bonus so the search does not walk itself into a corner.

Notes for the reader
--------------------
- All arithmetic on the belief vector is wrapped in `np.errstate` so that the
  noisy BLAS matmul path does not spam RuntimeWarnings.
- Every code block that mutates the belief calls `_normalize_distribution`
  which also defends against NaN/inf/0-sum catastrophes with a uniform fallback.
- The engine exposes internal bitmasks (`_primed_mask`, `_carpet_mask`,
  `_blocked_mask`); we use them directly for fast bit-level feature extraction.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Dict, List, Optional, Tuple

import numpy as np

# The engine imports its game package via a path append in player_process.py.
# Pulling `game.board` in keeps the engine's sys.path hook happy.
from game import board as _board_mod  # noqa: F401 - side-effect import
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


# ===========================================================================
# Section 1: Game constants, precomputed lookup tables, small helpers.
# ===========================================================================

# Total number of cells on the 8x8 board.
NUM_CELLS: int = BOARD_SIZE * BOARD_SIZE

# The rules: the rat executes this many moves on the spawn tile before the
# actual game starts. We precompute the post-headstart belief once at init
# time so we don't pay 1000 matrix-vector products in every game.
HEADSTART_STEPS: int = 1000

# Rat noise emission probabilities P(noise | cell type). The indices into the
# tuples match the Noise enum: 0 = squeak, 1 = scratch, 2 = squeal.
NOISE_EMISSION: Dict[Cell, Tuple[float, float, float]] = {
    Cell.BLOCKED: (0.5, 0.3, 0.2),
    Cell.SPACE:   (0.7, 0.15, 0.15),
    Cell.PRIMED:  (0.1, 0.8, 0.1),
    Cell.CARPET:  (0.1, 0.1, 0.8),
}

# Noisy-distance emission: observed = max(0, actual + offset). The offsets
# and probabilities come directly from the assignment PDF §4 ("The Rat").
DISTANCE_OFFSETS: Tuple[int, ...] = (-1, 0, 1, 2)
DISTANCE_PROBS: Tuple[float, ...] = (0.12, 0.7, 0.12, 0.06)


def _flat_index(x: int, y: int) -> int:
    """Flatten an (x, y) coordinate into its 0..63 cell index."""
    return y * BOARD_SIZE + x


def _cell_xy(flat: int) -> Tuple[int, int]:
    """Inverse of `_flat_index`: flat id -> (x, y)."""
    return (flat % BOARD_SIZE, flat // BOARD_SIZE)


def _manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Standard L1 distance between two grid cells."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ---------------------------------------------------------------------------
# Precomputed tables.
#
# Both tables are computed once at module import time. They only depend on
# the board geometry and the rat distance-noise model, so we never have to
# rebuild them during play.
# ---------------------------------------------------------------------------

# DISTANCE_MATRIX[w, c] = L1 distance between worker cell `w` and cell `c`.
# Shape (64, 64), int16 is plenty.
DISTANCE_MATRIX: np.ndarray = np.zeros((NUM_CELLS, NUM_CELLS), dtype=np.int16)
for _worker_idx in range(NUM_CELLS):
    _wx, _wy = _cell_xy(_worker_idx)
    for _target_idx in range(NUM_CELLS):
        _tx, _ty = _cell_xy(_target_idx)
        DISTANCE_MATRIX[_worker_idx, _target_idx] = abs(_wx - _tx) + abs(_wy - _ty)

# DISTANCE_LIKELIHOOD[actual, observed] = P(observed | actual).
# Accounts for the clip-to-zero behavior: any offset that would produce a
# negative observation contributes its probability mass to observation 0.
_MAX_L1_DIST: int = 2 * (BOARD_SIZE - 1)  # largest possible actual distance (14)
# We pad the "observed" axis a little so the indexing below can't run off.
DISTANCE_LIKELIHOOD: np.ndarray = np.zeros(
    (_MAX_L1_DIST + 1, _MAX_L1_DIST + 4), dtype=np.float64
)
for _actual in range(_MAX_L1_DIST + 1):
    for _offset, _prob in zip(DISTANCE_OFFSETS, DISTANCE_PROBS):
        _raw = _actual + _offset
        _clipped = _raw if _raw >= 0 else 0  # rules: never report negative
        if _clipped <= _MAX_L1_DIST + 3:
            DISTANCE_LIKELIHOOD[_actual, _clipped] += _prob


def _distance_likelihood(actual_dist: int, observed_dist: int) -> float:
    """Look up P(observed_dist | actual_dist) with safe bounds checking."""
    if observed_dist < 0 or observed_dist > _MAX_L1_DIST + 3:
        return 0.0
    return float(DISTANCE_LIKELIHOOD[actual_dist, observed_dist])


# Precomputed per-cell noise emission rows indexed by Cell integer value.
# This is a (4, 3) table so we can do a single fancy-index into it instead
# of a Python loop each time we need the noise likelihood vector.
_NOISE_ROW_BY_CELL_INT: np.ndarray = np.array(
    [
        [0.0, 0.0, 0.0],  # placeholder; we'll fill in the real entries next
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)
for _cell, _row in NOISE_EMISSION.items():
    _NOISE_ROW_BY_CELL_INT[int(_cell)] = _row


# ===========================================================================
# Section 2: Rat belief model (Hidden Markov Model).
# ===========================================================================


class RatBeliefModel:
    """
    Posterior distribution over the rat's current cell.

    The model maintains `self.belief`, a length-64 probability vector whose
    entry `i` is our current best estimate of P(rat is in cell i | history).

    Because the game interleaves my moves and the opponent's moves - with a
    rat transition before each player's move - the belief has to be advanced
    exactly the right number of times between observations. See
    `advance_and_observe` for the per-turn bookkeeping.
    """

    def __init__(self, transition_matrix_raw) -> None:
        # `transition_matrix_raw` can be a jax array, a numpy array, or any
        # array-like. We copy it into a contiguous float64 numpy array and
        # scrub any NaNs/negatives that might sneak in.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            T_dense = np.array(transition_matrix_raw, dtype=np.float64, copy=True)
            T_dense = np.nan_to_num(T_dense, nan=0.0, posinf=0.0, neginf=0.0)
            T_dense = np.maximum(T_dense, 0.0)

            # Re-normalize each row so that each row is a valid probability
            # distribution (sums to 1). If a row sums to ~0 for some reason,
            # replace its divisor with 1 so we don't blow up with NaNs.
            row_totals = T_dense.sum(axis=1, keepdims=True)
            row_totals = np.where(row_totals <= 1e-18, 1.0, row_totals)
            # `ascontiguousarray` tends to let BLAS take a faster code path.
            self.T: np.ndarray = np.ascontiguousarray(
                T_dense / row_totals, dtype=np.float64
            )

            # Precompute the belief after the 1000-step headstart. We start
            # from a delta at (0,0) (the rules say the rat spawns there) and
            # push it through T ^ HEADSTART_STEPS.
            v = np.zeros(NUM_CELLS, dtype=np.float64)
            v[0] = 1.0
            for _ in range(HEADSTART_STEPS):
                v = self._safe_matmul(v, self.T)
            # State at time T^1000 (before any player turn).
            self._spawn_belief: np.ndarray = v
            # State at time T^1001: one additional rat step which happens
            # just before the *next* player moves. This covers both the
            # "my very first turn" case and the "opponent just caught it"
            # case where the rat respawns and then takes its usual 1-step
            # move right before my turn.
            self._spawn_plus_one: np.ndarray = self._safe_matmul(v, self.T)

        # Running posterior, updated in place each turn. Starts as None
        # because we haven't observed anything yet.
        self.belief: Optional[np.ndarray] = None

    # ----- low-level utilities ---------------------------------------------

    def _safe_matmul(self, vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Compute vec @ matrix and renormalize defensively."""
        out = np.dot(vec, matrix)
        return self._normalize_distribution(out)

    def _normalize_distribution(self, vec: np.ndarray) -> np.ndarray:
        """
        Clean up a vector so it is a valid probability distribution.

        - Replace NaN/inf with 0.
        - Clip away any small negatives (floating-point noise).
        - If everything is zero or non-finite, fall back to the uniform
          distribution rather than divide-by-zeroing.
        - Otherwise renormalize so `vec.sum() == 1`.
        """
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        vec = np.maximum(vec, 0.0)
        total = vec.sum()
        if not np.isfinite(total) or total < 1e-12:
            # Uniform prior is the most honest "I know nothing" state.
            return np.full(NUM_CELLS, 1.0 / NUM_CELLS, dtype=np.float64)
        return vec / total

    # ----- per-turn observation update -------------------------------------

    def _apply_sensor_observation(
        self,
        prior: np.ndarray,
        board: Board,
        worker_xy: Tuple[int, int],
        noise_id: int,
        observed_dist: int,
    ) -> np.ndarray:
        """
        Bayes update: posterior ∝ prior * P(noise | cell type) * P(obs_dist | dist).

        We compute the two likelihood vectors in a vectorized way instead of
        looping in Python: the noise likelihood is just a fancy-index into
        our precomputed 4x3 table, and the distance likelihood is a lookup
        into our 64x64 distance matrix followed by a row slice.
        """
        # Build a per-cell integer array of the current board's cell types.
        cell_type_ints = np.empty(NUM_CELLS, dtype=np.int8)
        for ci in range(NUM_CELLS):
            cell_type_ints[ci] = int(board.get_cell(_cell_xy(ci)))

        # Vectorized noise likelihood: rows indexed by Cell int, then column
        # by the reported noise id.
        noise_likelihood = _NOISE_ROW_BY_CELL_INT[cell_type_ints, noise_id]

        # Vectorized distance likelihood: for each cell, look up the actual
        # distance from the worker, then look up P(observed | actual).
        worker_flat = _flat_index(*worker_xy)
        actual_distances = DISTANCE_MATRIX[worker_flat]  # shape (64,)

        obs_int = int(observed_dist)
        # Cap the observation into the valid column range to avoid indexing
        # errors from wildly out-of-range reports.
        if obs_int < 0:
            obs_int = 0
        if obs_int > _MAX_L1_DIST + 3:
            obs_int = _MAX_L1_DIST + 3
        distance_likelihood_vec = DISTANCE_LIKELIHOOD[actual_distances, obs_int]

        # Combine likelihoods and normalize.
        sanitized_prior = self._normalize_distribution(prior)
        posterior_unnorm = sanitized_prior * noise_likelihood * distance_likelihood_vec
        return self._normalize_distribution(posterior_unnorm)

    # ----- public entry point ---------------------------------------------

    def advance_and_observe(
        self,
        board: Board,
        noise_id: int,
        observed_dist: int,
        worker_xy: Tuple[int, int],
        turn_count: int,
        opp_search: Tuple,
        player_search: Tuple,
    ) -> None:
        """
        Advance the belief to the current turn and fold in the sensor reading.

        The game loop orders events as follows:
            for each player, in alternating order:
                rat.move()                # one T step
                samples = rat.sample()    # noise + distance we observe
                player.play()             # our move
                board.apply_move()        # may catch the rat -> respawn

        So between my consecutive observations, the rat has typically made
        exactly two T-steps (one before opponent's observation, one before
        mine). The exceptions are the very first turn (the rat has only made
        HEADSTART + 1 moves since spawn) and the recapture cases where the
        rat was caught and respawned in the middle.
        """
        opp_guess_loc, opp_guess_hit = opp_search
        my_prev_guess_loc, my_prev_guess_hit = player_search

        opponent_caught_rat = (opp_guess_loc is not None) and bool(opp_guess_hit)
        i_caught_rat_last_turn = (my_prev_guess_loc is not None) and bool(my_prev_guess_hit)

        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            if self.belief is None:
                # This is the first call to `play`. How many rat transitions
                # have occurred so far?
                #   - If I'm player A (turn_count == 0): HEADSTART + 1 = T^1001.
                #   - If I'm player B (turn_count == 1): HEADSTART + 2 = T^1002.
                # In practice we only ever see turn_count in {0, 1} here, but
                # we handle the general case by taking extra steps if needed.
                if turn_count == 0:
                    working = self._spawn_plus_one.copy()
                else:
                    working = self._spawn_belief.copy()
                    for _ in range(turn_count + 1):
                        working = self._safe_matmul(working, self.T)
                self.belief = self._normalize_distribution(working)
            else:
                if opponent_caught_rat:
                    # The opponent caught the rat on their previous turn, so
                    # a new rat spawned and ran HEADSTART steps, then took one
                    # more T step before my current observation. That is
                    # exactly `self._spawn_plus_one`.
                    self.belief = self._spawn_plus_one.copy()
                elif i_caught_rat_last_turn:
                    # I caught the rat on my previous turn. The new rat ran
                    # HEADSTART steps then took one T step to the opponent's
                    # observation moment. Between the opponent's observation
                    # and mine, one more T step happens.
                    working = self._spawn_plus_one.copy()
                    if opp_guess_loc is not None:
                        # If the opponent also guessed (and presumably missed,
                        # since we see a new rat in place), we can zero out
                        # their guess cell from the belief at *their*
                        # observation moment.
                        zero_idx = _flat_index(*opp_guess_loc)
                        working[zero_idx] = 0.0
                        working = self._normalize_distribution(working)
                    # One more T step to reach my observation moment.
                    working = self._safe_matmul(working, self.T)
                    self.belief = working
                else:
                    # Normal case: neither side caught anything, two T steps
                    # occurred between my last observation and this one.
                    #
                    # Sequence we need to unwind:
                    #   my_last_obs (= self.belief)
                    #   -> maybe filter out cells I searched & missed
                    #   -> T step (rat moves before opponent's obs)
                    #   -> maybe filter out cells opponent searched & missed
                    #   -> T step (rat moves before my new obs)
                    #   -> apply new sensor observation (handled below)
                    current = self._normalize_distribution(self.belief)

                    if my_prev_guess_loc is not None and not my_prev_guess_hit:
                        # If I searched at `my_prev_guess_loc` last turn and
                        # missed, the rat certainly was NOT there at my
                        # observation moment.
                        zero_idx = _flat_index(*my_prev_guess_loc)
                        current = current.copy()
                        current[zero_idx] = 0.0
                        current = self._normalize_distribution(current)

                    current = self._safe_matmul(current, self.T)

                    if opp_guess_loc is not None:
                        # Opponent searched and we got told the result. If
                        # they hit, we would have taken the opponent_caught
                        # branch above, so here we know they missed.
                        zero_idx = _flat_index(*opp_guess_loc)
                        current[zero_idx] = 0.0
                        current = self._normalize_distribution(current)

                    current = self._safe_matmul(current, self.T)
                    self.belief = current

            # Whatever branch we took, fold the sensor reading into the
            # belief now.
            self.belief = self._apply_sensor_observation(
                self.belief, board, worker_xy, noise_id, observed_dist
            )

    # ----- read-only queries -----------------------------------------------

    def peak_probability(self) -> float:
        """Return the maximum belief mass over any single cell."""
        if self.belief is None:
            return 0.0
        return float(self.belief.max())

    def best_cell_index(self) -> int:
        """Return the argmax cell index under the current belief."""
        if self.belief is None:
            return 0
        return int(np.argmax(self.belief))

    def top_k_cells(self, k: int) -> List[int]:
        """Return the `k` highest-belief cell indices in descending order."""
        if self.belief is None:
            return []
        k = max(1, min(k, NUM_CELLS))
        # argpartition is O(n); we then sort just the k candidates.
        rough = np.argpartition(-self.belief, k - 1)[:k]
        ordered = rough[np.argsort(-self.belief[rough])]
        return ordered.tolist()


# ===========================================================================
# Section 3: Evaluation function and its helper features.
# ===========================================================================


def _primed_line_score(board: Board) -> float:
    """
    Sum the CARPET_POINTS_TABLE reward over every maximal run of primed cells
    (both horizontal and vertical) that has length >= 2.

    A single primed cell is worth very little on its own (roll length 1 is
    actually -1 points), so we skip runs of length < 2. Runs of length 2+
    represent latent carpet-roll reward and get the matching table value.
    """
    primed_bits = board._primed_mask
    cumulative = 0.0

    # Horizontal runs.
    for row in range(BOARD_SIZE):
        run_length = 0
        for col in range(BOARD_SIZE):
            mask_bit = 1 << (row * BOARD_SIZE + col)
            if primed_bits & mask_bit:
                run_length += 1
            else:
                if run_length >= 2:
                    cumulative += CARPET_POINTS_TABLE.get(min(run_length, 7), 0)
                run_length = 0
        # Flush the run that hits the edge of the row.
        if run_length >= 2:
            cumulative += CARPET_POINTS_TABLE.get(min(run_length, 7), 0)

    # Vertical runs.
    for col in range(BOARD_SIZE):
        run_length = 0
        for row in range(BOARD_SIZE):
            mask_bit = 1 << (row * BOARD_SIZE + col)
            if primed_bits & mask_bit:
                run_length += 1
            else:
                if run_length >= 2:
                    cumulative += CARPET_POINTS_TABLE.get(min(run_length, 7), 0)
                run_length = 0
        if run_length >= 2:
            cumulative += CARPET_POINTS_TABLE.get(min(run_length, 7), 0)

    return cumulative


def _closest_prime_distance(board: Board, worker_xy: Tuple[int, int]) -> int:
    """Return the Manhattan distance from `worker_xy` to the nearest primed cell.

    Returns 0 if there are no primed cells at all (so the penalty evaporates).
    """
    primed_bits = board._primed_mask
    if primed_bits == 0:
        return 0

    wx, wy = worker_xy
    closest = BOARD_SIZE * 2
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if primed_bits & (1 << (row * BOARD_SIZE + col)):
                d = abs(col - wx) + abs(row - wy)
                if d < closest:
                    closest = d
    return closest if closest < BOARD_SIZE * 2 else 0


def _open_line_reach(board: Board, worker_xy: Tuple[int, int]) -> float:
    """
    "Open line reach" feature. This is a Carrie-style advanced heuristic that
    estimates the *future* prime-line potential of each open cell, weighted
    by how quickly our worker could actually get there.

    For each cell that is neither blocked nor already carpeted (so it can
    still be primed and then rolled):

        let L = length of the longest contiguous run of still-primable cells
                that this cell belongs to (horizontal or vertical).
        if L >= 2, add CARPET_POINTS_TABLE[L] / (1 + 0.6 * manhattan_to_worker)

    The 0.6 distance discount is a tunable knob that punishes cells that are
    very far away without eliminating their value entirely.
    """
    blocked_bits = board._blocked_mask
    carpet_bits = board._carpet_mask
    unavailable_bits = blocked_bits | carpet_bits  # cells that cannot prime
    wx, wy = worker_xy
    cumulative = 0.0

    # ---- horizontal max-run length per cell ------------------------------
    # For each row, walk left-to-right, remember each open segment's start
    # and length, then stamp that length into every cell of the segment.
    horizontal_run: List[List[int]] = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    for row in range(BOARD_SIZE):
        run_length = 0
        segments: List[Tuple[int, int]] = []
        for col in range(BOARD_SIZE):
            if unavailable_bits & (1 << (row * BOARD_SIZE + col)):
                if run_length > 0:
                    segments.append((col - run_length, run_length))
                run_length = 0
            else:
                run_length += 1
        if run_length > 0:
            segments.append((BOARD_SIZE - run_length, run_length))
        for start_col, length in segments:
            for col in range(start_col, start_col + length):
                horizontal_run[row][col] = length

    # ---- vertical max-run length per cell --------------------------------
    vertical_run: List[List[int]] = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    for col in range(BOARD_SIZE):
        run_length = 0
        segments = []
        for row in range(BOARD_SIZE):
            if unavailable_bits & (1 << (row * BOARD_SIZE + col)):
                if run_length > 0:
                    segments.append((row - run_length, run_length))
                run_length = 0
            else:
                run_length += 1
        if run_length > 0:
            segments.append((BOARD_SIZE - run_length, run_length))
        for start_row, length in segments:
            for row in range(start_row, start_row + length):
                vertical_run[row][col] = length

    # ---- accumulate distance-discounted reward ---------------------------
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if unavailable_bits & (1 << (row * BOARD_SIZE + col)):
                continue
            best_line = max(horizontal_run[row][col], vertical_run[row][col])
            if best_line < 2:
                continue  # single-cell segments do not yield net reward
            roll_value = CARPET_POINTS_TABLE.get(min(best_line, 7), 0)
            discount = 1.0 + 0.6 * (abs(col - wx) + abs(row - wy))
            cumulative += roll_value / discount
    return cumulative


def evaluate_position(board: Board, peak_belief: float) -> float:
    """
    Heuristic evaluation from the perspective of `board.player_worker`.

    The coefficients below are hand-tuned; the relative scale is more
    important than the absolute numbers. Because the score differential uses
    a 5x multiplier, downstream features that represent potential points
    also use a roughly 0.1 - 0.5 scale so that they can shift the decision
    when raw points are tied.
    """
    me = board.player_worker
    opp = board.opponent_worker

    # (1) Score differential is the biggest single factor. We multiply by 5
    # so that the various "potential" features below don't drown out a real
    # +1 point swing.
    evaluation = 5.0 * (me.get_points() - opp.get_points())

    # (2) Primed-line score: every 2+ length primed run adds its rolled
    # carpet value at a reduced weight. These aren't guaranteed to convert,
    # but they are extremely likely to.
    evaluation += 0.55 * _primed_line_score(board)

    # (3) Open line reach (Carrie-style). The closer I am to long open
    # corridors, the more *future* points I can create.
    evaluation += 0.22 * _open_line_reach(board, me.get_location())
    # Symmetric penalty for the opponent's reach; a slightly smaller weight
    # since we care more about our own upside.
    evaluation -= 0.12 * _open_line_reach(board, opp.get_location())

    # (4) Distance-to-nearest-prime penalty. Small but prevents my worker
    # from wandering away from primed cells it could have just rolled.
    if board._primed_mask:
        evaluation -= 0.35 * _closest_prime_distance(board, me.get_location())

    # (5) Mobility: having more legal moves is better, up to a cap so the
    # heuristic doesn't reward opening up huge empty spaces needlessly.
    try:
        legal_count = len(board.get_valid_moves(enemy=False, exclude_search=True))
    except Exception:
        legal_count = 4
    evaluation += 0.15 * min(legal_count, 20)

    # (6) Rat-knowledge bonus. If the current belief is peaked enough that a
    # single search has positive expected value, credit that optimistically.
    search_ev = 6.0 * peak_belief - 2.0  # 4*p - 2*(1-p) after simplification
    if search_ev > 0:
        evaluation += 1.5 * search_ev

    return evaluation


# ===========================================================================
# Section 4: Alpha-beta negamax searcher with iterative deepening.
# ===========================================================================


def _move_sort_key(move: Move) -> Tuple[int, int]:
    """
    Produce a sort key for move ordering. Higher tuples sort first.

    - Carpet rolls come first, broken by how many points they score.
    - Prime steps come second (they set up future carpets).
    - Plain steps come last.
    - Anything else (defensive default) sorts last.
    """
    if move.move_type == MoveType.CARPET:
        return (3, CARPET_POINTS_TABLE.get(min(move.roll_length, 7), 0))
    if move.move_type == MoveType.PRIME:
        return (2, 0)
    if move.move_type == MoveType.PLAIN:
        return (1, 0)
    return (0, 0)


def _prune_and_order_moves(candidates: List[Move]) -> List[Move]:
    """
    Drop strictly-dominated moves and return the rest in our preferred
    search order.

    In particular, a carpet roll of length 1 scores -1, so it is always
    worse than doing nothing and we filter those out entirely.
    """
    filtered: List[Move] = []
    for mv in candidates:
        if mv.move_type == MoveType.CARPET and mv.roll_length < 2:
            continue  # negative-value roll, never worth considering
        filtered.append(mv)
    filtered.sort(key=_move_sort_key, reverse=True)
    return filtered


class AlphaBetaSearcher:
    """
    Negamax-with-alpha-beta search over physical moves, plus iterative
    deepening with a soft time budget.

    The negamax formulation lets us evaluate from the current player's
    perspective and then simply negate on recursion. We only search over
    *physical* moves (plain/prime/carpet); the search (rat-guess) action is
    evaluated separately in the agent because its payoff is an expected
    value that does not fit cleanly into a minimax node.
    """

    def __init__(
        self,
        per_move_budget_s: float,
        remaining_time_getter: Callable[[], float],
    ) -> None:
        self.per_move_budget_s = per_move_budget_s
        self.remaining_time_getter = remaining_time_getter
        # Populated when `best_move_iddfs` kicks off.
        self._started_at: float = 0.0
        self._deadline: float = 0.0
        self._aborted: bool = False

    # -----------------------------------------------------------------
    # Time management helpers.
    # -----------------------------------------------------------------

    def _check_time_exhausted(self) -> bool:
        """
        Return True if we should stop searching. Conservative: we abort well
        before the hard engine deadline so we always have time to finish
        bookkeeping and return a move.
        """
        if self._aborted:
            return True
        try:
            global_remaining = self.remaining_time_getter()
        except Exception:
            global_remaining = 5.0
        if global_remaining < 0.5:
            self._aborted = True
            return True
        if self._deadline and _now() >= self._deadline:
            self._aborted = True
            return True
        return False

    # -----------------------------------------------------------------
    # Core negamax recursion.
    # -----------------------------------------------------------------

    def negamax(
        self,
        board: Board,
        depth: int,
        alpha: float,
        beta: float,
        peak_belief: float,
    ) -> float:
        """
        Standard negamax body. `alpha` is the best value already guaranteed
        for the maximizer at or above this node; `beta` is the ceiling
        imposed by the opponent. Pruning when alpha >= beta is safe.
        """
        # Terminal or quiesce conditions.
        if depth == 0 or board.is_game_over():
            return evaluate_position(board, peak_belief)
        if self._check_time_exhausted():
            return evaluate_position(board, peak_belief)

        child_moves = _prune_and_order_moves(board.get_valid_moves(exclude_search=True))
        if not child_moves:
            return evaluate_position(board, peak_belief)

        best_value = -float("inf")
        for mv in child_moves:
            next_board = board.forecast_move(mv, check_ok=False)
            if next_board is None:
                continue
            next_board.reverse_perspective()
            child_value = -self.negamax(
                next_board, depth - 1, -beta, -alpha, peak_belief
            )
            if child_value > best_value:
                best_value = child_value
            if best_value > alpha:
                alpha = best_value
            if alpha >= beta:
                break  # alpha-beta cutoff: opponent would never allow this
            if self._check_time_exhausted():
                break

        if best_value == -float("inf"):
            # No legal child produced a meaningful value (e.g. all were
            # filtered out). Fall back to the static eval so we don't
            # return -inf.
            return evaluate_position(board, peak_belief)
        return best_value

    # -----------------------------------------------------------------
    # Iterative deepening driver.
    # -----------------------------------------------------------------

    def best_move_iddfs(
        self,
        board: Board,
        max_depth: int,
        peak_belief: float,
    ) -> Tuple[Optional[Move], float]:
        """
        Run iterative-deepening negamax up to `max_depth`, respecting the
        per-move time budget. Between depth iterations we re-order the root
        move list using the scores we just learned, which dramatically
        improves alpha-beta pruning at the next depth.
        """
        root_moves = _prune_and_order_moves(board.get_valid_moves(exclude_search=True))
        if not root_moves:
            return None, -float("inf")

        # ---- Seed the root order with a cheap depth-1 eval per child. ----
        # We score each child by -heuristic(child) since the child is from
        # the opponent's perspective (reverse_perspective was called).
        seeded: List[Tuple[float, Move]] = []
        for mv in root_moves:
            forecast = board.forecast_move(mv, check_ok=False)
            if forecast is None:
                continue
            forecast.reverse_perspective()
            seeded.append((-evaluate_position(forecast, peak_belief), mv))
        seeded.sort(key=lambda t: t[0], reverse=True)
        ordered_moves = [mv for _score, mv in seeded]
        if not ordered_moves:
            return root_moves[0], -float("inf")

        # Default result in case we get aborted before completing depth 1.
        best_move: Move = ordered_moves[0]
        best_score: float = -float("inf")

        # Set up the time budget for this search.
        self._started_at = _now()
        self._deadline = self._started_at + self.per_move_budget_s
        self._aborted = False

        # ---- Iterative deepening loop. ------------------------------------
        for depth in range(1, max_depth + 1):
            if self._check_time_exhausted():
                break

            depth_best_move = best_move
            depth_best_score = -float("inf")
            alpha = -float("inf")
            beta = float("inf")

            # Always try the previously-best move first: that's the single
            # most effective alpha-beta move-ordering trick.
            move_queue = [best_move] + [mv for mv in ordered_moves if mv is not best_move]

            # Track (score, move) pairs so we can reorder for the next depth.
            depth_scores: List[Tuple[float, Move]] = []
            for mv in move_queue:
                forecast = board.forecast_move(mv, check_ok=False)
                if forecast is None:
                    continue
                forecast.reverse_perspective()
                mv_value = -self.negamax(
                    forecast, depth - 1, -beta, -alpha, peak_belief
                )
                depth_scores.append((mv_value, mv))

                if mv_value > depth_best_score:
                    depth_best_score = mv_value
                    depth_best_move = mv
                if depth_best_score > alpha:
                    alpha = depth_best_score
                if self._check_time_exhausted():
                    break

            if not self._aborted:
                # Commit this depth's result.
                best_move = depth_best_move
                best_score = depth_best_score
                # Seed the next depth with these up-to-date scores.
                depth_scores.sort(key=lambda t: t[0], reverse=True)
                ordered_moves = [mv for _s, mv in depth_scores]
            else:
                # Keep the last fully-completed depth's result.
                break

        return best_move, best_score


def _now() -> float:
    """Monotonic wall-clock getter; imported lazily so we don't pay at import."""
    import time as _time
    return _time.perf_counter()


# ===========================================================================
# Section 5: The PlayerAgent class the engine will instantiate and drive.
# ===========================================================================


class PlayerAgent:
    """
    Top-level agent object that the engine instantiates once per game and
    calls `play(...)` on each turn. The engine expects:

        class PlayerAgent:
            def __init__(self, board, transition_matrix, time_left_func): ...
            def play(self, board, sensor_data, time_left_func) -> Move: ...
            def commentate(self) -> str: ...
    """

    TEAM_NAME: str = "ArjunAndGary"

    # How many turns back we remember a missed search location. While the
    # HMM's belief diffuses back to a missed cell over a few T steps, we
    # refuse to re-guess it. Tunable.
    MISS_MEMORY_TURNS: int = 3

    # How many top cells to consider when selecting our search target. The
    # true argmax may be a recently-missed cell, in which case we fall
    # through to the next best option.
    SEARCH_TOPK: int = 6

    # Minimum belief probability at which a search is worth doing. Raw EV
    # breaks even at p = 1/3 (6p - 2 > 0). We use a safety margin so we
    # only search when the best cell clearly dominates.
    SEARCH_PROB_THRESHOLD: float = 0.44

    def __init__(
        self,
        board: Board,
        transition_matrix=None,
        time_left: Callable = None,
    ) -> None:
        # The HMM belief tracker. `transition_matrix` may be None in weird
        # test harnesses, in which case we degrade gracefully.
        self.rat_model: Optional[RatBeliefModel] = (
            RatBeliefModel(transition_matrix) if transition_matrix is not None else None
        )

        # { (x, y) -> turn_count } of recent misses so we don't blindly
        # re-guess the same spot while the belief re-concentrates.
        self._missed_search_cells: Dict[Tuple[int, int], int] = {}

    # -----------------------------------------------------------------
    # Optional hook the engine uses for post-game commentary.
    # -----------------------------------------------------------------

    def commentate(self) -> str:
        return f"{self.TEAM_NAME} out."

    # -----------------------------------------------------------------
    # Time budgeting.
    # -----------------------------------------------------------------

    def _turn_time_budget(self, board: Board) -> float:
        """
        Allocate at most ~85% of the average per-move slice, capped at
        5.5 seconds so that a single unlucky turn can't burn our entire
        4-minute pool.
        """
        turns_remaining = max(1, board.player_worker.turns_left)
        time_remaining = max(0.5, board.player_worker.time_left)
        return min(5.5, 0.85 * time_remaining / turns_remaining)

    # -----------------------------------------------------------------
    # Public entry point the engine calls each turn.
    # -----------------------------------------------------------------

    def play(
        self,
        board: Board,
        sensor_data: Tuple,
        time_left: Callable,
    ) -> Move:
        """
        Decide on a move for the current turn. Wraps `_decide_move` in a
        try/except so that any unexpected exception falls back to a safe
        legal move instead of crashing the engine (which would forfeit).
        """
        try:
            return self._decide_move(board, sensor_data, time_left)
        except Exception:
            return self._safe_fallback_move(board)

    # -----------------------------------------------------------------
    # The brain.
    # -----------------------------------------------------------------

    def _decide_move(
        self,
        board: Board,
        sensor_data: Tuple,
        time_left: Callable,
    ) -> Move:
        # Unpack the sensor tuple. The engine sends (noise, noisy_distance).
        reported_noise, reported_distance = sensor_data
        noise_as_int = int(reported_noise)
        distance_as_int = int(reported_distance)

        my_location = board.player_worker.get_location()

        # Phase 1: update the HMM belief with the latest observation.
        if self.rat_model is not None:
            self.rat_model.advance_and_observe(
                board=board,
                noise_id=noise_as_int,
                observed_dist=distance_as_int,
                worker_xy=my_location,
                turn_count=board.turn_count,
                opp_search=board.opponent_search,
                player_search=board.player_search,
            )

        peak_belief = (
            self.rat_model.peak_probability() if self.rat_model is not None else 0.0
        )

        # Phase 2: run iterative-deepening negamax over physical moves.
        time_budget = self._turn_time_budget(board)
        searcher = AlphaBetaSearcher(
            per_move_budget_s=time_budget,
            remaining_time_getter=time_left,
        )
        MAX_SEARCH_DEPTH = 6
        best_physical_move, best_physical_score = searcher.best_move_iddfs(
            board, max_depth=MAX_SEARCH_DEPTH, peak_belief=peak_belief
        )

        # Phase 3: consider the "search for the rat" action as a separate
        # candidate. We estimate its full value as the continuation value
        # (opponent's best reply after we search) plus the expected points
        # from the immediate guess, scaled to the same 5x points weighting
        # the heuristic uses.
        search_move, search_total_value = self._score_search_action(
            board, searcher, peak_belief, MAX_SEARCH_DEPTH
        )

        # Phase 4: compare and return the winner.
        if best_physical_move is None and search_move is None:
            return self._safe_fallback_move(board)

        # Require a meaningful edge before preferring the search action,
        # since its value estimate is an approximation.
        if search_move is not None and search_total_value > best_physical_score + 0.01:
            return search_move

        if best_physical_move is not None:
            return best_physical_move

        # If only the search action was valid, take it.
        return search_move  # type: ignore[return-value]

    # -----------------------------------------------------------------
    # Search-action scoring.
    # -----------------------------------------------------------------

    def _score_search_action(
        self,
        board: Board,
        searcher: AlphaBetaSearcher,
        peak_belief: float,
        max_depth: int,
    ) -> Tuple[Optional[Move], float]:
        """
        Evaluate the best rat-guess we could make right now. Returns
        (move, total_value) or (None, -inf) if no guess is worth making.
        """
        if self.rat_model is None or self.rat_model.belief is None:
            return None, -float("inf")

        # Bookkeeping: remember any miss we just learned about (so we don't
        # loop back onto it), and expire very old misses.
        previous_guess_loc, previous_guess_hit = board.player_search
        if previous_guess_loc is not None and not previous_guess_hit:
            self._missed_search_cells[tuple(previous_guess_loc)] = board.turn_count

        # Prune expired misses.
        self._missed_search_cells = {
            cell: turn
            for cell, turn in self._missed_search_cells.items()
            if board.turn_count - turn <= self.MISS_MEMORY_TURNS
        }

        # Find the highest-probability cell among the top-k that is NOT on
        # the recent-misses list.
        top_candidates = self.rat_model.top_k_cells(self.SEARCH_TOPK)
        selected_flat_idx = -1
        selected_prob = 0.0
        for flat_idx in top_candidates:
            candidate_xy = _cell_xy(flat_idx)
            if candidate_xy in self._missed_search_cells:
                continue
            selected_flat_idx = flat_idx
            selected_prob = float(self.rat_model.belief[flat_idx])
            break

        if selected_flat_idx < 0 or selected_prob <= self.SEARCH_PROB_THRESHOLD:
            # Either no viable candidate or the best candidate is too weak
            # for a positive-EV search.
            return None, -float("inf")

        # Forecast the search move. In the engine, a SEARCH move doesn't
        # move the worker; it only advances the turn counter and decrements
        # turns_left. So the continuation value is just "what does the
        # opponent do next from this same physical position?"
        target_xy = _cell_xy(selected_flat_idx)
        search_move = Move.search(target_xy)
        next_board = board.forecast_move(search_move, check_ok=False)
        if next_board is None:
            return None, -float("inf")
        next_board.reverse_perspective()

        # Continuation value: opponent's best reply, evaluated from our
        # perspective via negation.
        continuation_value = -searcher.negamax(
            next_board,
            depth=max(0, max_depth - 1),
            alpha=-float("inf"),
            beta=float("inf"),
            peak_belief=peak_belief,
        )

        # Expected immediate reward from the guess:
        #   hit  (prob = p):   +4
        #   miss (prob = 1-p): -2
        # => EV = 4p - 2(1-p) = 6p - 2
        expected_reward = 6.0 * selected_prob - 2.0

        # Scale by the heuristic's 5x points weight so we're comparing
        # apples to apples with `best_physical_score`.
        total_value = continuation_value + 5.0 * expected_reward
        return search_move, total_value

    # -----------------------------------------------------------------
    # Last-ditch safe move (used when the brain throws an exception).
    # -----------------------------------------------------------------

    def _safe_fallback_move(self, board: Board) -> Move:
        """
        Return any legal move that is not a net-negative carpet roll of 1.
        If even that fails, fall all the way back to a plain-right step.
        """
        try:
            candidates = board.get_valid_moves(exclude_search=True)
            for mv in candidates:
                if mv.move_type == MoveType.CARPET and mv.roll_length < 2:
                    continue
                return mv
            if candidates:
                return candidates[0]
        except Exception:
            pass
        return Move.plain(Direction.RIGHT)
