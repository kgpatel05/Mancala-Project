import numpy as np
import sys

class GameState:
    def __init__(self, p1_pits: np.array, p2_pits: np.array, p1_store: int, p2_store: int, turn: int, currentplayer: int):
        self.board = [p1_pits, p2_pits]
        self.stores = [p1_store, p2_store]
        self.current_player = currentplayer
        self.turn_number = turn

    def duplicate(self):
        return GameState(self.board[0].copy(), self.board[1].copy(), self.stores[0], self.stores[1], self.turn_number, self.current_player)


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
    if game_state.current_player == 2 and game_state.turn_number == 1:
        return "PIE"
    
    best_move = None
    best_value = float("-inf") if game_state.current_player == 1 else float("inf")
    alpha, beta = float("-inf"), float("inf")
    
    total_stones = sum(game_state.board[0]) + sum(game_state.board[1])
    if total_stones > 20:
        depth = 6 # Early game
    elif total_stones > 10:
        depth = 8  # Mid game
    else:
        depth = 10  # End game
    
    for move in get_valid_moves(game_state):
        new_state = simulate_move(game_state, move)
        move_value = minimax(new_state, depth, alpha, beta, maximizing_player=(game_state.current_player == 1))
        
        if game_state.current_player == 1:
            if move_value > best_value:
                best_value = move_value
                best_move = move
            alpha = max(alpha, best_value)
        else:
            if move_value < best_value:
                best_value = move_value
                best_move = move
            beta = min(beta, best_value)
    
    return str(best_move + 1) if best_move is not None else "1"


def minimax(state: GameState, depth: int, alpha: float, beta: float, maximizing_player: bool) -> float:
    if depth == 0 or is_terminal(state):
        return evaluate_state(state)
    
    if maximizing_player:
        max_eval = float("-inf")
        for move in get_valid_moves(state):
            new_state = simulate_move(state, move)
            eval = minimax(new_state, depth - 1, alpha, beta, False)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float("inf")
        for move in get_valid_moves(state):
            new_state = simulate_move(state, move)
            eval = minimax(new_state, depth - 1, alpha, beta, True)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


def get_valid_moves(state: GameState) -> list[int]:
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
    index = pit_index if player_index == 0 else pit_index + len(pits[0]) + 1
    
    while stones > 0:
        index = (index + 1) % (2 * len(pits[0]) + 2)
        if (player_index == 0 and index == len(pits[0]) * 2 + 1) or (player_index == 1 and index == len(pits[0])):
            continue
        
        if index < len(pits[0]):
            pits[0][index] += 1
        elif index == len(pits[0]):
            new_stores[0] += 1
        elif index < len(pits[0]) * 2 + 1:
            pits[1][index - (len(pits[0]) + 1)] += 1
        else:
            new_stores[1] += 1
        stones -= 1
    
    next_player = state.current_player if (player_index == 0 and index == len(pits[0])) or (player_index == 1 and index == len(pits[0]) * 2 + 1) else 3 - state.current_player
    
    return GameState(new_p1_pits, new_p2_pits, new_stores[0], new_stores[1], state.turn_number + 1, next_player)


def evaluate_state(state: GameState) -> float:
    score = (state.stores[0] - state.stores[1]) * 5  # Increase weight of store difference
    score += sum(state.board[state.current_player - 1]) - sum(state.board[2 - state.current_player])
    score += sum(1 for pit in state.board[state.current_player - 1] if pit == 1) * 3  # Reward capture opportunities
    score += sum(1 for pit in state.board[state.current_player - 1] if pit > 3) * 2  # Reward maintaining pits with more stones
    score -= sum(1 for pit in state.board[2 - state.current_player] if pit == 1) * 3  # Penalize opponent's capture opportunities
    score -= sum(1 for pit in state.board[2 - state.current_player] if pit > 3) * 2  # Penalize opponent's strong positions
    score += 10 if state.current_player == 1 and state.stores[0] > state.stores[1] else 0  # Encourage leading
    score -= 10 if state.current_player == 2 and state.stores[1] > state.stores[0] else 0  # Discourage trailing
    return score


def is_terminal(state: GameState) -> bool:
    return np.sum(state.board[0]) == 0 or np.sum(state.board[1]) == 0


if __name__ == "__main__":
    input_str = sys.stdin.readline().strip()
    game_state = process_input(input_str)
    print(nextmove(game_state))
