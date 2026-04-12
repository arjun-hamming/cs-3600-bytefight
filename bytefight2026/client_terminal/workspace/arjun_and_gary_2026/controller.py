from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import math
import time
from typing import Optional, Union

from game import Action, Board, Direction, GameConstants, Location, MoveType, Parity
from game.board import Hill
from game.outcome import Result


WIN_SCORE = 1_000_000_000.0


@dataclass
class SearchNode:
    board: Board
    actions: tuple[Action.Move | Action.Paint, ...]
    move_count: int
    moved: bool
    heuristic: float


class PlayerController:
    """
    Search-based controller for ByteFight 2026 Paint.

    This is an iteration on the prior bot's approach:
    - explicit forward simulation
    - heuristic evaluation
    - shallow adversarial lookahead

    The main change is that the search now reasons over action sequences in
    the current Paint ruleset instead of single moves in the older rat game.
    """

    MAX_DEPTH = 5
    ROOT_BEAM = 14
    REPLY_BEAM = 8
    TOP_CANDIDATES = 6
    MAX_MOVE_ACTIONS = 3
    MAX_ACTION_OPTIONS = 12

    def __init__(self, player_parity: int, time_left: Callable):
        self.player_parity = player_parity

    def bid(self, board: Board, player_parity: int, time_left: Callable) -> int:
        deadline = time.perf_counter() + 0.08
        our_eval, _ = self._search_turn(
            board=board,
            turn_parity=player_parity,
            perspective_parity=player_parity,
            deadline=deadline,
            max_depth=3,
            beam_width=8,
            return_candidates=False,
        )
        opp_eval, _ = self._search_turn(
            board=board,
            turn_parity=-player_parity,
            perspective_parity=player_parity,
            deadline=deadline + 0.03,
            max_depth=3,
            beam_width=8,
            return_candidates=False,
        )

        me = board.get_player(player_parity)
        swing = our_eval - opp_eval
        hill_count = len(board.hills)
        bid = 4 + min(6, hill_count)
        if swing > 0:
            bid += int(min(12, swing / 14.0))

        nearest_power = self._nearest_powerup_distance(board, player_parity)
        if nearest_power is not None and nearest_power <= 2:
            bid += 2

        return max(0, min(int(me.stamina // 3), min(25, bid)))

    def play(
        self,
        board: Board,
        player_parity: int,
        time_left: Callable,
    ) -> Union[Action.Move, Action.Paint, Iterable[Action.Move | Action.Paint]]:
        budget = self._time_budget(board, time_left)
        deadline = time.perf_counter() + budget

        _, candidates = self._search_turn(
            board=board,
            turn_parity=player_parity,
            perspective_parity=player_parity,
            deadline=deadline,
            max_depth=self.MAX_DEPTH,
            beam_width=self.ROOT_BEAM,
            return_candidates=True,
        )

        if not candidates:
            return self._fallback_turn(board, player_parity)

        best_actions = candidates[0][1]
        best_value = -float("inf")

        for our_value, our_actions, our_board in candidates[: self.TOP_CANDIDATES]:
            if time.perf_counter() >= deadline:
                break

            if our_board.get_winner() is not None:
                if our_value > best_value:
                    best_value = our_value
                    best_actions = our_actions
                continue

            reply_deadline = min(deadline, time.perf_counter() + max(0.03, budget * 0.28))
            reply_value, _ = self._search_turn(
                board=our_board,
                turn_parity=-player_parity,
                perspective_parity=player_parity,
                deadline=reply_deadline,
                max_depth=3,
                beam_width=self.REPLY_BEAM,
                return_candidates=False,
            )

            if reply_value > best_value:
                best_value = reply_value
                best_actions = our_actions

        if not best_actions:
            return self._fallback_turn(board, player_parity)
        if len(best_actions) == 1:
            return best_actions[0]
        return list(best_actions)

    def commentate(self, board: Board, player_parity: int, time_left: Callable) -> str:
        return "beam-search paint bot"

    def _time_budget(self, board: Board, time_left: Callable) -> float:
        try:
            remaining = max(0.25, float(time_left()))
        except Exception:
            remaining = 15.0
        remaining_rounds = max(1, GameConstants.MAX_ROUNDS - board.current_round)
        average_slice = remaining / remaining_rounds
        return min(0.75, max(0.07, 0.7 * average_slice))

    def _search_turn(
        self,
        board: Board,
        turn_parity: int,
        perspective_parity: int,
        deadline: float,
        max_depth: int,
        beam_width: int,
        return_candidates: bool,
    ) -> tuple[float, list[tuple[float, tuple[Action.Move | Action.Paint, ...], Board]]]:
        initial = SearchNode(
            board=board.get_copy(),
            actions=(),
            move_count=0,
            moved=False,
            heuristic=self._evaluate_partial(board, perspective_parity),
        )
        frontier = [initial]
        terminals: list[tuple[float, tuple[Action.Move | Action.Paint, ...], Board]] = []

        for _depth in range(max_depth):
            if time.perf_counter() >= deadline:
                break

            next_frontier: list[SearchNode] = []
            for node in frontier:
                if time.perf_counter() >= deadline:
                    break

                if node.moved:
                    finalized = self._finalize_turn(node.board)
                    terminals.append(
                        (
                            self._evaluate_terminal(finalized, perspective_parity),
                            node.actions,
                            finalized,
                        )
                    )

                expansions = self._expand_node(
                    node=node,
                    turn_parity=turn_parity,
                    perspective_parity=perspective_parity,
                    deadline=deadline,
                )
                next_frontier.extend(expansions)

            if not next_frontier:
                break

            next_frontier.sort(
                key=lambda item: self._search_key(item.heuristic, turn_parity, perspective_parity),
                reverse=True,
            )
            frontier = next_frontier[:beam_width]

        for node in frontier:
            if node.moved:
                finalized = self._finalize_turn(node.board)
                terminals.append(
                    (
                        self._evaluate_terminal(finalized, perspective_parity),
                        node.actions,
                        finalized,
                    )
                )

        if not terminals:
            fallback = self._fallback_turn(board, turn_parity)
            fallback_actions = (fallback,) if isinstance(fallback, (Action.Move, Action.Paint)) else tuple(fallback)
            forecast, ok = board.forecast_turn(turn_parity, list(fallback_actions))
            if ok:
                terminals = [
                    (
                        self._evaluate_terminal(forecast, perspective_parity),
                        fallback_actions,
                        forecast,
                    )
                ]
            else:
                terminals = [(self._evaluate_terminal(board, perspective_parity), (), board.get_copy())]

        terminals.sort(
            key=lambda item: self._search_key(item[0], turn_parity, perspective_parity),
            reverse=True,
        )
        best_value = terminals[0][0]
        return best_value, terminals if return_candidates else []

    def _expand_node(
        self,
        node: SearchNode,
        turn_parity: int,
        perspective_parity: int,
        deadline: float,
    ) -> list[SearchNode]:
        options = self._enumerate_actions(node.board, turn_parity, node.move_count)
        if not options:
            return []

        expansions: list[SearchNode] = []
        for local_priority, action, next_board in options[: self.MAX_ACTION_OPTIONS]:
            if time.perf_counter() >= deadline:
                break

            moved = node.moved or isinstance(action, Action.Move)
            move_count = node.move_count + (1 if isinstance(action, Action.Move) else 0)
            heuristic = self._evaluate_partial(next_board, perspective_parity)
            heuristic += local_priority
            heuristic -= 0.8 * len(node.actions)

            expansions.append(
                SearchNode(
                    board=next_board,
                    actions=node.actions + (action,),
                    move_count=move_count,
                    moved=moved,
                    heuristic=heuristic,
                )
            )
        return expansions

    def _enumerate_actions(
        self,
        board: Board,
        player_parity: int,
        move_count: int,
    ) -> list[tuple[float, Action.Move | Action.Paint, Board]]:
        player = board.get_player(player_parity)
        actions: list[tuple[float, Action.Move | Action.Paint, Board]] = []

        for loc in self._adjacent_locations(player.loc):
            paint = Action.Paint(loc)
            next_board, ok = board.forecast_action(player_parity, paint)
            if ok:
                actions.append((self._action_priority(board, next_board, player_parity, paint), paint, next_board))

        if move_count < self.MAX_MOVE_ACTIONS:
            for direction in Direction.cardinals():
                for move_type in (MoveType.REGULAR, MoveType.ERASE):
                    move = Action.Move(direction=direction, move_type=move_type)
                    next_board, ok = board.forecast_action(player_parity, move)
                    if ok:
                        actions.append((self._action_priority(board, next_board, player_parity, move), move, next_board))

                    beacon_move = Action.Move(
                        direction=direction,
                        move_type=move_type,
                        place_beacon=True,
                    )
                    beacon_board, beacon_ok = board.forecast_action(player_parity, beacon_move)
                    if beacon_ok:
                        actions.append(
                            (
                                self._action_priority(board, beacon_board, player_parity, beacon_move),
                                beacon_move,
                                beacon_board,
                            )
                        )

            if self._cell(board, player.loc).beacon_parity == player_parity:
                for target in self._own_beacons(board, player_parity):
                    move = Action.Move(
                        direction=None,
                        move_type=MoveType.BEACON_TRAVEL,
                        beacon_target=target,
                    )
                    next_board, ok = board.forecast_action(player_parity, move)
                    if ok:
                        actions.append((self._action_priority(board, next_board, player_parity, move), move, next_board))

        actions.sort(key=lambda item: item[0], reverse=True)
        return actions

    def _action_priority(
        self,
        before: Board,
        after: Board,
        player_parity: int,
        action: Action.Move | Action.Paint,
    ) -> float:
        me_before = before.get_player(player_parity)
        me_after = after.get_player(player_parity)
        opp_before = before.get_opponent(player_parity)
        opp_after = after.get_opponent(player_parity)

        priority = 0.0
        winner = after.get_winner()
        if winner is not None:
            result, _ = winner
            if self._result_for_parity(result, player_parity) > 0:
                return WIN_SCORE

        priority += 4.0 * (
            after.get_territory_count(player_parity) - before.get_territory_count(player_parity)
        )
        priority -= 1.2 * (me_before.stamina - me_after.stamina)
        priority += 1.5 * (opp_before.stamina - opp_after.stamina)
        priority += 24.0 * (
            len(me_after.controlled_hills) - len(me_before.controlled_hills)
        )
        priority -= 18.0 * (
            len(opp_after.controlled_hills) - len(opp_before.controlled_hills)
        )

        if isinstance(action, Action.Paint):
            cell = self._cell(after, action.location)
            if cell.hill_id:
                priority += 10.0
            if cell.beacon_parity == -player_parity:
                priority += 6.0
            priority += 0.8 * abs(cell.paint_value)
        else:
            target = me_after.loc
            target_cell = self._cell(after, target)
            if target_cell.powerup:
                priority += 8.0
            if target_cell.hill_id:
                priority += 10.0
            if target == opp_before.loc and self._collision_is_favorable(before, player_parity, target):
                priority += 120.0
            if action.move_type == MoveType.ERASE and target_cell.hill_id:
                priority += 8.0
            if action.move_type == MoveType.BEACON_TRAVEL:
                priority += 6.0
            if action.place_beacon:
                priority += 12.0
            priority += self._strategic_cell_bonus(after, player_parity, target)

        return priority

    def _evaluate_partial(self, board: Board, perspective_parity: int) -> float:
        return self._evaluate(board, perspective_parity, include_turn_end=False)

    def _evaluate_terminal(self, board: Board, perspective_parity: int) -> float:
        return self._evaluate(board, perspective_parity, include_turn_end=True)

    def _evaluate(self, board: Board, perspective_parity: int, include_turn_end: bool) -> float:
        winner = board.get_winner()
        if winner is not None:
            result, _ = winner
            if result == Result.TIE:
                return 0.0
            return WIN_SCORE if self._result_for_parity(result, perspective_parity) > 0 else -WIN_SCORE

        me = board.get_player(perspective_parity)
        opp = board.get_opponent(perspective_parity)

        territory_diff = board.get_territory_count(perspective_parity) - board.get_territory_count(-perspective_parity)
        stamina_diff = me.stamina - opp.stamina
        max_stamina_diff = me.max_stamina - opp.max_stamina
        hill_control_diff = len(me.controlled_hills) - len(opp.controlled_hills)
        hill_progress_diff = self._hill_progress(board, perspective_parity) - self._hill_progress(board, -perspective_parity)
        paint_mass_diff = self._paint_mass(board, perspective_parity) - self._paint_mass(board, -perspective_parity)
        local_control_diff = self._local_control(board, perspective_parity) - self._local_control(board, -perspective_parity)
        power_score_diff = self._powerup_score(board, perspective_parity) - self._powerup_score(board, -perspective_parity)
        beacon_diff = self._beacon_score(board, perspective_parity) - self._beacon_score(board, -perspective_parity)
        pressure_diff = self._collision_pressure(board, perspective_parity) - self._collision_pressure(board, -perspective_parity)

        score = 0.0
        score += 85.0 * hill_control_diff
        score += 9.0 * hill_progress_diff
        score += 2.4 * stamina_diff
        score += 1.5 * max_stamina_diff
        score += 1.8 * territory_diff
        score += 0.45 * paint_mass_diff
        score += 0.85 * local_control_diff
        score += 7.5 * power_score_diff
        score += 4.0 * beacon_diff
        score += 12.0 * pressure_diff

        score += 0.4 * self._strategic_cell_bonus(board, perspective_parity, me.loc)
        score -= 0.25 * self._strategic_cell_bonus(board, -perspective_parity, opp.loc)

        if include_turn_end:
            score += 0.15 * (board.current_round / max(1, GameConstants.MAX_ROUNDS))

        return score

    def _result_for_parity(self, result: Result, player_parity: int) -> int:
        if result == Result.TIE:
            return 0
        if player_parity == 1:
            return 1 if result == Result.PLAYER_1 else -1
        return 1 if result == Result.PLAYER_2 else -1

    def _finalize_turn(self, board: Board) -> Board:
        finalized = board.get_copy()
        finalized.end_turn()
        return finalized

    def _hill_progress(self, board: Board, player_parity: int) -> float:
        progress = 0.0
        for hill in board.hills.values():
            my_cells = self._hill_cells_for_parity(hill, player_parity)
            opp_cells = self._hill_cells_for_parity(hill, -player_parity)
            required = math.ceil(len(hill.cells) * GameConstants.HILL_CONTROL_THRESHOLD)
            progress += 0.8 * (my_cells - opp_cells)
            if my_cells >= required:
                progress += 3.0
            if opp_cells >= required:
                progress -= 3.0
        return progress

    def _hill_cells_for_parity(self, hill: Hill, player_parity: int) -> int:
        if player_parity > 0:
            return hill.control_positive
        return -hill.control_negative

    def _paint_mass(self, board: Board, player_parity: int) -> int:
        total = 0
        for row in board.cells:
            for cell in row:
                if cell.owner_parity == player_parity:
                    total += abs(cell.paint_value)
        return total

    def _local_control(self, board: Board, player_parity: int) -> int:
        player = board.get_player(player_parity)
        count = 0
        radius = GameConstants.ADJACENCY_RADIUS
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                loc = Location(player.loc.r + dr, player.loc.c + dc)
                if board.oob(loc):
                    continue
                if self._cell(board, loc).owner_parity == player_parity:
                    count += 1
        return count

    def _powerup_score(self, board: Board, player_parity: int) -> float:
        player = board.get_player(player_parity)
        score = 0.0
        for r, row in enumerate(board.cells):
            for c, cell in enumerate(row):
                if not cell.powerup:
                    continue
                dist = self._grid_distance(board, player.loc, Location(r, c))
                if dist is None:
                    continue
                score += 10.0 / (1.0 + dist)
        return score

    def _beacon_score(self, board: Board, player_parity: int) -> float:
        total = 0.0
        for row in board.cells:
            for cell in row:
                if cell.beacon_parity == player_parity:
                    total += 2.5
                    if cell.owner_parity == player_parity:
                        total += 0.5 * abs(cell.paint_value)
        if self._cell(board, board.get_player(player_parity).loc).beacon_parity == player_parity:
            total += 2.0
        return total

    def _collision_pressure(self, board: Board, player_parity: int) -> float:
        player = board.get_player(player_parity)
        opponent = board.get_opponent(player_parity)
        for direction in Direction.cardinals():
            if player.loc + direction != opponent.loc:
                continue
            if self._collision_is_favorable(board, player_parity, opponent.loc):
                return 1.0
            return -1.0
        return 0.0

    def _collision_is_favorable(self, board: Board, player_parity: int, target: Location) -> bool:
        target_cell = self._cell(board, target)
        if target_cell.owner_parity == -player_parity:
            return False
        return True

    def _strategic_cell_bonus(self, board: Board, player_parity: int, loc: Location) -> float:
        if board.oob(loc):
            return -5.0
        cell = self._cell(board, loc)
        bonus = 0.0
        if cell.hill_id:
            bonus += 8.0
            hill = board.hills[cell.hill_id]
            required = math.ceil(len(hill.cells) * GameConstants.HILL_CONTROL_THRESHOLD)
            my_cells = self._hill_cells_for_parity(hill, player_parity)
            if my_cells < required:
                bonus += 4.0
        if cell.powerup:
            bonus += 10.0
        nearest_power = self._nearest_powerup_distance(board, player_parity)
        if nearest_power is not None:
            bonus += 3.0 / (1.0 + nearest_power)
        return bonus

    def _nearest_powerup_distance(self, board: Board, player_parity: int) -> Optional[int]:
        player = board.get_player(player_parity)
        best: Optional[int] = None
        for r, row in enumerate(board.cells):
            for c, cell in enumerate(row):
                if not cell.powerup:
                    continue
                dist = self._grid_distance(board, player.loc, Location(r, c))
                if dist is None:
                    continue
                if best is None or dist < best:
                    best = dist
        return best

    def _grid_distance(self, board: Board, start: Location, target: Location) -> Optional[int]:
        if start == target:
            return 0
        frontier = [start]
        visited = {start}
        depth = 0
        while frontier and depth < 12:
            depth += 1
            next_frontier: list[Location] = []
            for loc in frontier:
                for direction in Direction.cardinals():
                    nxt = loc + direction
                    if board.oob(nxt):
                        continue
                    if nxt in visited:
                        continue
                    if self._cell(board, nxt).is_wall:
                        continue
                    if nxt == target:
                        return depth
                    visited.add(nxt)
                    next_frontier.append(nxt)
            frontier = next_frontier
        return None

    def _adjacent_locations(self, loc: Location) -> tuple[Location, ...]:
        return tuple(loc + direction for direction in Direction.cardinals())

    def _own_beacons(self, board: Board, player_parity: int) -> list[Location]:
        beacons: list[Location] = []
        for r, row in enumerate(board.cells):
            for c, cell in enumerate(row):
                if cell.beacon_parity == player_parity:
                    beacons.append(Location(r, c))
        return beacons

    def _cell(self, board: Board, loc: Location):
        return board.cells[loc.r][loc.c]

    def _search_key(self, score: float, turn_parity: int, perspective_parity: int) -> float:
        return score if turn_parity == perspective_parity else -score

    def _fallback_turn(self, board: Board, player_parity: int) -> list[Action.Move | Action.Paint]:
        player = board.get_player(player_parity)
        paint_targets: list[Location] = []
        for loc in self._adjacent_locations(player.loc):
            if board.oob(loc):
                continue
            cell = self._cell(board, loc)
            if cell.is_wall:
                continue
            if cell.owner_parity in (0, player_parity) and cell.beacon_parity != player_parity:
                paint_targets.append(loc)

        for loc in paint_targets[:2]:
            action = Action.Paint(loc)
            forecast, ok = board.forecast_turn(player_parity, [action, Action.Move(Direction.UP)])
            if ok:
                return [action, Action.Move(Direction.UP)]

        for direction in Direction.cardinals():
            move = Action.Move(direction=direction)
            forecast, ok = board.forecast_turn(player_parity, [move])
            if ok:
                return [move]

        return [Action.Move(Direction.UP)]
