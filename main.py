import numpy as np
import sys

########################################
# GLOBAL TRANSPOSITION TABLE
########################################
transposition_table = {}

class GameState:
    def __init__(self, p1_pits: np.array, p2_pits: np.array, p1_store: int, p2_store: int, turn: int, currentplayer: int):
        self.board = [p1_pits, p2_pits]
        self.stores = [p1_store, p2_store]
        self.current_player = currentplayer
        self.turn_number = turn

    def duplicate(self):
        return GameState(self.board[0].copy(), self.board[1].copy(),
                         self.stores[0], self.stores[1],
                         self.turn_number, self.current_player)

def process_input(input_str: str) -> GameState:
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
    Top-level function. It clears the transposition table,
    uses the PIE rule when appropriate, and calls negamax from the root.
    We determine root_player from the gamestate so that our search is always
    from the perspective of the player whose turn it is.
    """
    transposition_table.clear()
    
    # Handle PIE for player2's first turn.
    if game_state.current_player == 2 and game_state.turn_number == 1:
        return "PIE"
    
    root_player = game_state.current_player
    best_move = None
    best_value = float("-inf")
    alpha, beta = float("-inf"), float("inf")
    
    total_stones = int(np.sum(game_state.board[0]) + np.sum(game_state.board[1]))
    if total_stones > 20:
        depth = 6  # Early game
    elif total_stones > 10:
        depth = 8  # Mid game
    else:
        depth = 10 # End game
    
    moves = get_valid_moves(game_state)
    # Order moves using a shallow evaluation (from the root player's perspective)
    moves.sort(key=lambda m: evaluate_for_root(simulate_move(game_state, m), root_player), reverse=True)
    
    for move in moves:
        child_state = simulate_move(game_state, move)
        # Negamax call: we use the negative of the child evaluation.
        value = -negamax(child_state, depth - 1, -beta, -alpha, root_player)
        if value > best_value:
            best_value = value
            best_move = move
        alpha = max(alpha, value)
        if alpha >= beta:
            break

    return str(best_move + 1) if best_move is not None else "1"

def negamax(state: GameState, depth: int, alpha: float, beta: float, root_player: int) -> float:
    """
    Negamax with alpha-beta pruning and transposition table.
    All evaluations are returned from the perspective of root_player.
    """
    cache_key = get_state_key(state, depth, alpha, beta, root_player)
    if cache_key in transposition_table:
        return transposition_table[cache_key]
    
    if depth == 0 or is_terminal(state):
        val = evaluate_state(state)
        # If the state is from the opponent's perspective relative to root_player, flip sign.
        if state.current_player != root_player:
            val = -val
        transposition_table[cache_key] = val
        return val

    max_val = float("-inf")
    moves = get_valid_moves(state)
    # Order moves at this node
    moves.sort(key=lambda m: evaluate_for_root(simulate_move(state, m), root_player), reverse=True)
    
    for move in moves:
        child = simulate_move(state, move)
        score = -negamax(child, depth - 1, -beta, -alpha, root_player)
        max_val = max(max_val, score)
        alpha = max(alpha, score)
        if alpha >= beta:
            break

    transposition_table[cache_key] = max_val
    return max_val

def get_valid_moves(state: GameState) -> list:
    current_pits = state.board[state.current_player - 1]
    return [i for i in range(len(current_pits)) if current_pits[i] > 0]

def simulate_move(state: GameState, move: int) -> GameState:
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
        # Skip opponent's store
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
    
    # If the last stone lands in the current player's store, that player goes again.
    if (player_index == 0 and pos == len(pits[0])) or (player_index == 1 and pos == total_positions - 1):
        next_player = state.current_player
    else:
        next_player = 1 if state.current_player == 2 else 2
    
    return GameState(new_p1_pits, new_p2_pits, new_stores[0], new_stores[1], state.turn_number + 1, next_player)

def is_terminal(state: GameState) -> bool:
    return np.sum(state.board[0]) == 0 or np.sum(state.board[1]) == 0

def evaluate_state(state: GameState) -> float:
    """
    This evaluation implements the Algorithm 4 style heuristic.
    It computes various features from the current player's perspective.
    """
    player = state.current_player
    opponent = 1 if player == 2 else 2
    p_idx = player - 1
    o_idx = opponent - 1

    # H1: Stones in leftmost pit
    H1 = state.board[p_idx][0]
    # H2: Total seeds on player's side
    H2 = np.sum(state.board[p_idx])
    # H3: Number of nonempty pits for the player
    H3 = np.count_nonzero(state.board[p_idx])
    # H4: Player's store seeds
    H4 = state.stores[p_idx]
    # H5: 1 if the rightmost pit has stones, else 0
    H5 = 1 if state.board[p_idx][-1] > 0 else 0
    # H6: Negative of opponent's store
    H6 = -state.stores[o_idx]
    # H7: 1 if the game is not terminal, else 0
    H7 = 0 if is_terminal(state) else 1
    # H8: Store difference (from current player's perspective)
    H8 = state.stores[p_idx] - state.stores[o_idx]
    # H9: If opponent's store >= 5, penalize:
    H9 = 0
    if state.stores[o_idx] >= 5:
        H9 = -(state.stores[o_idx] * 1.5) - state.stores[p_idx]
    # H10: If current player's store >= 5, reward:
    H10 = 0
    if state.stores[p_idx] >= 5:
        H10 = (state.stores[p_idx] * 1.5) - state.stores[o_idx]

    # Chosen weights (these remain our baseline; you can experiment further):
    W1  = 0.5   # leftmost pit
    W2  = 1.0   # total seeds on player's side
    W3  = 1.0   # number of nonempty pits
    W4  = 8.0   # player's store seeds
    W5  = 0.5   # rightmost pit
    W6  = 4.0   # negative of opponent's store
    W7  = 2.0   # not terminal bonus
    W8  = 10.0  # store difference
    W9  = 3.0   # penalty for opponent's strong store (if >=5)
    W10 = 3.0   # bonus for own strong store (if >=5)

    score = (H1 * W1 + H2 * W2 + H3 * W3 + H4 * W4 + H5 * W5 +
             H6 * W6 + H7 * W7 + H8 * W8 + H9 * W9 + H10 * W10)
    return score

def evaluate_for_root(state: GameState, root_player: int) -> float:
    """
    Helper function that returns the evaluation from the root player's perspective.
    If the state's current player is not the root, the sign is flipped.
    """
    val = evaluate_state(state)
    return val if state.current_player == root_player else -val

def get_state_key(state: GameState, depth: int, alpha: float, beta: float, root_player: int) -> tuple:
    """
    Constructs a key for the transposition table.
    """
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
    print(nextmove(game_state))
