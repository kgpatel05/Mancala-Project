import numpy as np
import sys
import time

transposition_table = {}

# Set the time limit (in seconds) for computing a move.
TIME_LIMIT = 0.90
# Safety margin in seconds to prevent timeout.
TIME_MARGIN = 0.01

class GameState:
    def __init__(self, p1_pits: np.array, p2_pits: np.array, p1_store: int, p2_store: int, turn: int, currentplayer: int):
        self.board = [p1_pits, p2_pits]
        self.stores = [p1_store, p2_store]
        self.turn_number = turn
        self.current_player = currentplayer
        # Flag indicating if the last move resulted in an extra turn.
        self.extra_turn = False

    def duplicate(self):
        new_state = GameState(self.board[0].copy(), self.board[1].copy(),
                              self.stores[0], self.stores[1],
                              self.turn_number, self.current_player)
        new_state.extra_turn = self.extra_turn
        return new_state

def process_input(input_str: str) -> GameState:
    """
    Expects a single-line input in the format:
    STATE <N> <p1_pits...> <p2_pits...> <p1_store> <p2_store> <turn> <player>
    """
    tokens = input_str.split()
    if tokens[0] != "STATE":
        raise ValueError("Invalid input format, expected 'STATE' at the beginning.")
    
    N = int(tokens[1])
    p1_pits = np.array(list(map(int, tokens[2:N+2])))
    p2_pits = np.array(list(map(int, tokens[N+2:2*N+2])))
    p1_store = int(tokens[2*N+2])
    p2_store = int(tokens[2*N+3])
    turn = int(tokens[2*N+4])
    currentplayer = int(tokens[2*N+5])
    
    return GameState(p1_pits, p2_pits, p1_store, p2_store, turn, currentplayer)

def nextmove(game_state: GameState) -> str:
    """
    Uses iterative deepening with negamax (with alpha-beta pruning) until the time limit is reached.
    Returns a pit selection (1-indexed) or "PIE" if applicable.
    """
    transposition_table.clear()
    
    # For Player 2's first turn, return the PIE move.
    if game_state.turn_number == 1 and game_state.current_player == 2:
        return "PIE"
    
    root_player = game_state.current_player
    total_stones = int(np.sum(game_state.board[0]) + np.sum(game_state.board[1]))
    if total_stones > 20:
        max_possible_depth = 6  # Early game
    elif total_stones > 10:
        max_possible_depth = 8  # Mid game
    else:
        max_possible_depth = 14 # End game

    best_move = None
    start_time = time.time()
    end_time = start_time + TIME_LIMIT
    depth = 1

    while time.time() < end_time - TIME_MARGIN and depth <= max_possible_depth:
        current_best_move = None
        current_best_value = float("-inf")
        alpha, beta = float("-inf"), float("inf")
        moves = get_valid_moves(game_state)
        # Order moves using a quick evaluation from the root's perspective.
        moves.sort(key=lambda m: evaluate_for_root(simulate_move(game_state, m), root_player), reverse=True)
        for move in moves:
            if time.time() >= end_time - TIME_MARGIN:
                break
            child_state = simulate_move(game_state, move)
            value = -negamax(child_state, depth - 1, -beta, -alpha, root_player, end_time)
            if value > current_best_value:
                current_best_value = value
                current_best_move = move
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        if time.time() < end_time - TIME_MARGIN and current_best_move is not None:
            best_move = current_best_move
        depth += 1

    # If no move was computed (should not happen), return a default move.
    if best_move is None:
        best_move = 0
    if best_move == -1:
        return "PIE"
    return str(best_move + 1)

def negamax(state: GameState, depth: int, alpha: float, beta: float, root_player: int, end_time: float) -> float:
    if time.time() >= end_time - TIME_MARGIN:
        return evaluate_state(state) if state.current_player == root_player else -evaluate_state(state)
    
    cache_key = get_state_key(state, depth, alpha, beta, root_player)
    if cache_key in transposition_table:
        return transposition_table[cache_key]
    
    if depth == 0 or is_terminal(state):
        val = evaluate_state(state)
        if state.current_player != root_player:
            val = -val
        transposition_table[cache_key] = val
        return val

    max_val = float("-inf")
    moves = get_valid_moves(state)
    moves.sort(key=lambda m: evaluate_for_root(simulate_move(state, m), root_player), reverse=True)
    for move in moves:
        if time.time() >= end_time - TIME_MARGIN:
            break
        child = simulate_move(state, move)
        score = -negamax(child, depth - 1, -beta, -alpha, root_player, end_time)
        max_val = max(max_val, score)
        alpha = max(alpha, score)
        if alpha >= beta:
            break

    transposition_table[cache_key] = max_val
    return max_val

def get_valid_moves(state: GameState) -> list:
    current_pits = state.board[state.current_player - 1]
    moves = [i for i in range(len(current_pits)) if current_pits[i] > 0]
    if state.turn_number == 1 and state.current_player == 2:
        moves.append(-1)
    return moves

def simulate_move(state: GameState, move) -> GameState:
    if move == -1:
        new_state = GameState(state.board[1].copy(), state.board[0].copy(),
                              state.stores[1], state.stores[0],
                              state.turn_number + 1, 1)
        new_state.extra_turn = False
        return new_state

    new_p1_pits = state.board[0].copy()
    new_p2_pits = state.board[1].copy()
    new_stores = state.stores.copy()
    
    pits = [new_p1_pits, new_p2_pits]
    player_index = state.current_player - 1
    pit_index = move
    stones = pits[player_index][pit_index]
    pits[player_index][pit_index] = 0
    total_positions = 2 * len(pits[0]) + 2
    pos = pit_index if player_index == 0 else pit_index + len(pits[0]) + 1
    
    while stones > 0:
        pos = (pos + 1) % total_positions
        if (player_index == 0 and pos == total_positions - 1) or (player_index == 1 and pos == len(pits[0])):
            continue
        if pos < len(pits[0]):
            pits[0][pos] += 1
        elif pos == len(pits[0]):
            new_stores[0] += 1
        elif pos < len(pits[0]) * 2 + 1:
            pits[1][pos - (len(pits[0]) + 1)] += 1
        else:
            new_stores[1] += 1
        stones -= 1

    # Capture rule.
    if not ((player_index == 0 and pos == len(pits[0])) or (player_index == 1 and pos == total_positions - 1)):
        if player_index == 0 and pos < len(pits[0]) and pits[0][pos] == 1:
            opp_index = len(pits[0]) - 1 - pos
            if pits[1][opp_index] > 0:
                captured = pits[1][opp_index] + pits[0][pos]
                pits[0][pos] = 0
                pits[1][opp_index] = 0
                new_stores[0] += captured
        elif player_index == 1 and pos > len(pits[0]) and pos < len(pits[0]) * 2 + 1:
            j = pos - (len(pits[0]) + 1)
            opp_index = len(pits[0]) - 1 - j
            if pits[0][opp_index] > 0 and pits[1][j] == 1:
                captured = pits[0][opp_index] + pits[1][j]
                pits[1][j] = 0
                pits[0][opp_index] = 0
                new_stores[1] += captured

    if (player_index == 0 and pos == len(pits[0])) or (player_index == 1 and pos == total_positions - 1):
        next_player = state.current_player
    else:
        next_player = 1 if state.current_player == 2 else 2

    new_state = GameState(new_p1_pits, new_p2_pits, new_stores[0], new_stores[1],
                          state.turn_number + 1, next_player)
    new_state.extra_turn = (next_player == state.current_player)
    return new_state

def is_terminal(state: GameState) -> bool:
    return np.sum(state.board[0]) == 0 or np.sum(state.board[1]) == 0

def evaluate_state(state: GameState) -> float:
    """
    Computes a weighted sum of features.
      • H1: seeds in leftmost pit.
      • H2: total seeds on current player's side.
      • H3: number of nonempty pits (mobility).
      • H4: current player's store.
      • H5: indicator if rightmost pit is nonempty.
      • H6: negative of opponent's store.
      • H7: count of pits with seeds equal to distance to store (extra-turn chances).
      • H8: store difference.
      • H9, H10: conditional adjustments.
      • H11: vulnerability penalty.
      • H12: weighted proximity sum.
      • H13: bonus if the move granted an extra turn.
      • H14: mobility difference.
    The weights below have been tuned to favor a larger store, a larger store difference,
    to value extra-turn opportunities more, and to penalize vulnerabilities more harshly.
    """
    player = state.current_player
    opponent = 1 if player == 2 else 2
    p_idx = player - 1
    o_idx = opponent - 1
    N = len(state.board[p_idx])
    
    H1 = state.board[p_idx][0]
    H2 = np.sum(state.board[p_idx])
    H3 = np.count_nonzero(state.board[p_idx])
    H4 = state.stores[p_idx]
    H5 = 1 if state.board[p_idx][-1] > 0 else 0
    H6 = -state.stores[o_idx]
    H7 = sum(1 for i in range(N) if state.board[p_idx][i] == (N - i))
    H8 = state.stores[p_idx] - state.stores[o_idx]
    H9 = 0
    if state.stores[o_idx] >= 5:
        H9 = -(state.stores[o_idx] * 1.5) - state.stores[p_idx]
    H10 = 0
    if state.stores[p_idx] >= 5:
        H10 = (state.stores[p_idx] * 1.5) - state.stores[o_idx]
    H11 = 0
    for i in range(N):
        if state.board[p_idx][i] == 0:
            opp_i = N - 1 - i
            H11 += state.board[o_idx][opp_i]
    if player == 1:
        H12 = sum((N - i) * state.board[0][i] for i in range(N))
    else:
        H12 = sum((i + 1) * state.board[1][i] for i in range(N))
    H13 = 1 if state.extra_turn else 0
    H14 = np.count_nonzero(state.board[p_idx]) - np.count_nonzero(state.board[o_idx])
    
    W1  = 0.5     
    W2  = 0.6     
    W3  = 1.0     
    W4  = 40.0    # Reward player's store more.
    W5  = 0.5     
    W6  = 10.0    
    W7  = 10.0    # Extra-turn opportunities are more valuable.
    W8  = 60.0    # Larger store difference is prioritized.
    W9  = 5.0     
    W10 = 5.0     
    W11 = -5.0    # Increased penalty for vulnerability.
    W12 = 0.3     
    W13 = 20.0    # Extra-turn bonus increased.
    W14 = 2.0     

    score = (H1 * W1 + H2 * W2 + H3 * W3 + H4 * W4 + H5 * W5 +
             H6 * W6 + H7 * W7 + H8 * W8 + H9 * W9 + H10 * W10 +
             H11 * W11 + H12 * W12 + H13 * W13 + H14 * W14)
    return score

def evaluate_for_root(state: GameState, root_player: int) -> float:
    val = evaluate_state(state)
    return val if state.current_player == root_player else -val

def get_state_key(state: GameState, depth: int, alpha: float, beta: float, root_player: int) -> tuple:
    return (tuple(state.board[0]),
            tuple(state.board[1]),
            state.stores[0],
            state.stores[1],
            state.current_player,
            depth,
            root_player,
            alpha,
            beta)

if __name__ == "__main__":
    input_str = sys.stdin.readline().strip()
    game_state = process_input(input_str)
    move = nextmove(game_state)
    print(move)
    sys.stdout.flush()
