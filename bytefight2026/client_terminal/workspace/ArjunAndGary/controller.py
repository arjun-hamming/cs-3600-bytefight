from __future__ import annotations

from collections.abc import Callable, Iterable
from enum import Enum
import math
from typing import Optional, Union

from game import Action, Board, Direction, GameConstants, Location, MoveType, Parity
from game.board import Hill
from game.outcome import Result


class Mode(Enum):
    OPENING = 0
    HILL_RUSH = 1
    TERRITORY = 2
    DEFENSE = 3
    ENDGAME = 4


class PlayerController:
    """
    FSM + goal-directed paint bot for ByteFight 2026.

    Key improvements over v1 beam-search bot:
    - Anti-oscillation via position history tracking
    - Paint-first strategy (paint every turn when possible)
    - FSM mode system for strategic coherence
    - Greedy pathfinding with collision avoidance
    - Conservative bidding to preserve stamina
    """

    HISTORY_SIZE = 20
    OSCILLATION_PENALTY = 65.0
    PAINT_STAM_MIN = 25

    def __init__(self, player_parity: int, time_left: Callable):
        self.player_parity = player_parity
        self.pos_history: list[tuple[int, int]] = []
        self.mode = Mode.OPENING
        self.turn_count = 0

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #

    def bid(self, board: Board, player_parity: int, time_left: Callable) -> int:
        try:
            return self._bid_inner(board, player_parity)
        except Exception:
            return 0

    def play(
        self,
        board: Board,
        player_parity: int,
        time_left: Callable,
    ) -> Union[Action.Move, Action.Paint, Iterable[Action.Move | Action.Paint]]:
        try:
            return self._play_inner(board, player_parity, time_left)
        except Exception:
            return self._emergency_move(board, player_parity)

    def commentate(self, board: Board, player_parity: int, time_left: Callable) -> str:
        me = board.get_player(player_parity)
        return (
            f"v2 mode={self.mode.name} stam={me.stamina}/{me.max_stamina} "
            f"terr={board.get_territory_count(player_parity)} "
            f"hills={len(me.controlled_hills)}"
        )

    # ------------------------------------------------------------------ #
    #  Bidding                                                             #
    # ------------------------------------------------------------------ #

    def _bid_inner(self, board: Board, pp: int) -> int:
        me = board.get_player(pp)
        if board.current_round < 3:
            return 0
        cap = max(0, int(me.stamina * 0.04))
        if len(me.controlled_hills) == 0 and len(board.hills) > 0:
            return min(3, cap)
        return min(2, cap)

    # ------------------------------------------------------------------ #
    #  Core play logic                                                     #
    # ------------------------------------------------------------------ #

    def _play_inner(
        self, board: Board, pp: int, time_left: Callable
    ) -> Union[Action.Move, Action.Paint, Iterable[Action.Move | Action.Paint]]:
        me = board.get_player(pp)

        # Track position for anti-oscillation
        self.pos_history.append((me.loc.r, me.loc.c))
        if len(self.pos_history) > self.HISTORY_SIZE:
            self.pos_history = self.pos_history[-self.HISTORY_SIZE:]
        self.turn_count += 1

        # Determine strategy
        self.mode = self._select_mode(board, pp)
        target = self._get_target(board, pp)

        # Build validated action sequence
        actions = self._build_actions(board, pp, target)
        if actions:
            return actions if len(actions) > 1 else actions[0]

        # Fallback: try just the best move
        d = self._best_move_dir(board, pp, target)
        if d is not None:
            mv = Action.Move(direction=d)
            _, ok = board.forecast_turn(pp, [mv])
            if ok:
                return mv

        return self._emergency_move(board, pp)

    # ------------------------------------------------------------------ #
    #  Mode selection                                                      #
    # ------------------------------------------------------------------ #

    def _select_mode(self, board: Board, pp: int) -> Mode:
        me = board.get_player(pp)
        opp = board.get_opponent(pp)

        if self.turn_count <= 2:
            return Mode.OPENING

        if board.current_round >= 500:
            return Mode.ENDGAME

        # Defense: opponent near one of our controlled hills
        for hid in list(me.controlled_hills):
            hill = board.hills.get(hid)
            if hill is None:
                continue
            for cl in hill.cells:
                if self._mdist(cl, opp.loc) <= 3:
                    return Mode.DEFENSE

        # Hill rush: we control nothing
        if len(me.controlled_hills) == 0 and len(board.hills) > 0:
            return Mode.HILL_RUSH

        return Mode.TERRITORY

    # ------------------------------------------------------------------ #
    #  Target selection per mode                                           #
    # ------------------------------------------------------------------ #

    def _get_target(self, board: Board, pp: int) -> Optional[Location]:
        if self.mode in (Mode.OPENING, Mode.HILL_RUSH):
            return self._nearest_needed_hill_cell(board, pp)
        if self.mode == Mode.DEFENSE:
            return self._threatened_hill_cell(board, pp)
        if self.mode == Mode.ENDGAME:
            return self._nearest_powerup(board, pp) or self._nearest_neutral(board, pp)
        # TERRITORY
        return self._territory_target(board, pp)

    def _nearest_needed_hill_cell(self, board: Board, pp: int) -> Optional[Location]:
        """Nearest hill cell we need to paint (prioritize uncontrolled hills)."""
        me = board.get_player(pp)
        best, best_d = None, 999999
        for hill in board.hills.values():
            bonus = 0 if hill.controller_parity != pp else 10000
            for loc in hill.cells:
                if board.oob(loc):
                    continue
                cell = board.cells[loc.r][loc.c]
                # Skip cells we already own
                if cell.owner_parity == pp:
                    continue
                d = self._mdist(me.loc, loc) + bonus
                if d < best_d:
                    best_d = d
                    best = loc
        return best or self._nearest_powerup(board, pp) or self._nearest_neutral(board, pp)

    def _threatened_hill_cell(self, board: Board, pp: int) -> Optional[Location]:
        """Our hill cell closest to opponent — go defend it."""
        opp = board.get_opponent(pp)
        me = board.get_player(pp)
        best, best_d = None, 999999
        for hid in list(me.controlled_hills):
            hill = board.hills.get(hid)
            if hill is None:
                continue
            for loc in hill.cells:
                d = self._mdist(loc, opp.loc)
                if d < best_d:
                    best_d = d
                    best = loc
        return best or self._nearest_needed_hill_cell(board, pp)

    def _territory_target(self, board: Board, pp: int) -> Optional[Location]:
        me = board.get_player(pp)
        # Priority 1: uncontrolled hill cells within reach
        ht = self._nearest_needed_hill_cell(board, pp)
        if ht and self._mdist(me.loc, ht) <= 12:
            return ht
        # Priority 2: powerup
        pu = self._nearest_powerup(board, pp)
        if pu and self._mdist(me.loc, pu) <= 6:
            return pu
        # Priority 3: expand paint
        return self._nearest_neutral(board, pp) or me.loc

    def _nearest_powerup(self, board: Board, pp: int) -> Optional[Location]:
        me = board.get_player(pp)
        best, best_d = None, 999999
        for r, row in enumerate(board.cells):
            for c, cell in enumerate(row):
                if cell.powerup:
                    loc = Location(r, c)
                    d = self._mdist(me.loc, loc)
                    if d < best_d:
                        best_d = d
                        best = loc
        return best

    def _nearest_neutral(self, board: Board, pp: int) -> Optional[Location]:
        me = board.get_player(pp)
        best, best_d = None, 999999
        for r, row in enumerate(board.cells):
            for c, cell in enumerate(row):
                if cell.is_wall or cell.owner_parity != 0:
                    continue
                loc = Location(r, c)
                d = self._mdist(me.loc, loc)
                if d < best_d:
                    best_d = d
                    best = loc
        return best

    # ------------------------------------------------------------------ #
    #  Action building                                                     #
    # ------------------------------------------------------------------ #

    def _build_actions(
        self, board: Board, pp: int, target: Optional[Location]
    ) -> Optional[list]:
        me = board.get_player(pp)
        opp = board.get_opponent(pp)

        move_dir = self._best_move_dir(board, pp, target)
        if move_dir is None:
            return None

        dest = me.loc + move_dir
        dest_cell = board.cells[dest.r][dest.c]

        # Decide move type: erase on opponent hill cells if worthwhile
        use_erase = (
            dest_cell.owner_parity == -pp
            and dest_cell.hill_id
            and me.stamina >= 70
            and abs(dest_cell.paint_value) >= 2
        )
        move_type = MoveType.ERASE if use_erase else MoveType.REGULAR
        move_action = Action.Move(direction=move_dir, move_type=move_type)
        move_cost = 40 if use_erase else 0

        # Paint targets: before move (adjacent to current pos) and after (adjacent to dest)
        pb = self._best_paint(board, pp, me.loc)
        pa = self._best_paint(board, pp, dest)
        # Ensure they differ
        if pa is not None and pb is not None and pa == pb:
            pa = None

        paint_cost = GameConstants.PAINT_STAMINA_COST  # 15

        # Try progressively simpler action combos
        combos = []
        if pb and pa and me.stamina >= paint_cost + move_cost + paint_cost:
            combos.append([Action.Paint(pb), move_action, Action.Paint(pa)])
        if pb and me.stamina >= paint_cost + move_cost:
            combos.append([Action.Paint(pb), move_action])
        if pa and me.stamina >= move_cost + paint_cost:
            combos.append([move_action, Action.Paint(pa)])
        combos.append([move_action])

        for combo in combos:
            _, ok = board.forecast_turn(pp, combo)
            if ok:
                return combo
        return None

    def _best_paint(
        self, board: Board, pp: int, from_loc: Location
    ) -> Optional[Location]:
        """Best cell to paint from a given location."""
        best_s, best_l = -1, None
        for d in Direction.cardinals():
            loc = from_loc + d
            if board.oob(loc):
                continue
            cell = board.cells[loc.r][loc.c]
            if cell.is_wall:
                continue
            if cell.owner_parity == -pp:
                continue  # Can't paint opponent cells
            if cell.beacon_parity == pp:
                continue  # Can't paint on own beacon

            s = 0
            if cell.hill_id:
                hill = board.hills[cell.hill_id]
                if hill.controller_parity != pp:
                    s += 100  # Uncontrolled hill — top priority
                elif abs(cell.paint_value) < GameConstants.MAX_PAINT_VALUE:
                    s += 12   # Strengthen our hill
                else:
                    s += 1
            if cell.owner_parity == 0:
                s += 20       # New territory
            elif cell.owner_parity == pp:
                if abs(cell.paint_value) < GameConstants.MAX_PAINT_VALUE:
                    s += 3    # Strengthen existing
            if cell.beacon_parity == -pp:
                s += 35       # Destroy opponent beacon

            if s > best_s:
                best_s = s
                best_l = loc
        return best_l if best_s > 0 else None

    # ------------------------------------------------------------------ #
    #  Move direction scoring                                              #
    # ------------------------------------------------------------------ #

    def _best_move_dir(
        self, board: Board, pp: int, target: Optional[Location]
    ) -> Optional[Direction]:
        me = board.get_player(pp)
        opp = board.get_opponent(pp)
        cands: list[tuple[float, Direction]] = []

        for d in Direction.cardinals():
            nl = me.loc + d
            if board.oob(nl):
                continue
            cell = board.cells[nl.r][nl.c]
            if cell.is_wall:
                continue

            sc = 0.0

            # ---- Goal proximity ----
            if target is not None:
                old = self._mdist(me.loc, target)
                new = self._mdist(nl, target)
                sc += (old - new) * 12.0

            # ---- Anti-oscillation (CRITICAL) ----
            pos = (nl.r, nl.c)
            for i, prev in enumerate(reversed(self.pos_history)):
                if prev == pos:
                    w = 1.0 / (1.0 + i * 0.25)
                    sc -= self.OSCILLATION_PENALTY * w

            # ---- Collision avoidance ----
            od = self._mdist(nl, opp.loc)
            if od == 0:
                # Direct collision
                if cell.owner_parity == pp:
                    sc += 80.0   # We win collision on our turf
                else:
                    sc -= 250.0  # Deadly on non-friendly ground
            elif od == 1:
                if cell.owner_parity != pp:
                    sc -= 30.0   # Risky adjacency

            # ---- Incentives ----
            if cell.powerup:
                sc += 50.0
            if cell.hill_id:
                hill = board.hills[cell.hill_id]
                if hill.controller_parity != pp:
                    sc += 14.0
                    if cell.owner_parity == -pp:
                        sc += 6.0   # Erases opponent paint on entry
            if cell.owner_parity == 0:
                sc += 3.0           # Explore neutral
            elif cell.owner_parity == pp:
                sc += 0.5           # Safe ground

            # ---- Paintability bonus: prefer moving to spot with good paint targets ----
            paint_nearby = 0
            for d2 in Direction.cardinals():
                adj = nl + d2
                if board.oob(adj):
                    continue
                ac = board.cells[adj.r][adj.c]
                if ac.is_wall or ac.owner_parity == -pp:
                    continue
                if ac.beacon_parity == pp:
                    continue
                if ac.hill_id and ac.owner_parity != pp:
                    paint_nearby += 8
                elif ac.owner_parity == 0:
                    paint_nearby += 2
            sc += paint_nearby * 0.5

            cands.append((sc, d))

        if not cands:
            return None
        cands.sort(key=lambda x: x[0], reverse=True)
        return cands[0][1]

    # ------------------------------------------------------------------ #
    #  Utilities                                                           #
    # ------------------------------------------------------------------ #

    def _mdist(self, a: Location, b: Location) -> int:
        return abs(a.r - b.r) + abs(a.c - b.c)

    def _emergency_move(self, board: Board, pp: int) -> Action.Move:
        me = board.get_player(pp)
        for d in Direction.cardinals():
            nl = me.loc + d
            if board.oob(nl):
                continue
            if board.cells[nl.r][nl.c].is_wall:
                continue
            mv = Action.Move(direction=d)
            _, ok = board.forecast_turn(pp, [mv])
            if ok:
                return mv
        return Action.Move(Direction.UP)
