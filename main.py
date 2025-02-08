import numpy as np
import time

class GameState:
    def __init__(self, p1_pits:np.array, p2_pits:np.array, p1_store:int, p2_store:int, turn:int, currentplayer:int):
        self.board = [p1_pits, p2_pits]  # 2d array representation of the game board with provided pit values
        self.stores = [p1_store, p2_store]
        self.current_player = currentplayer
        self.turn_number = turn

    def display_board(self):
        print("  ", end="")
        for pit in self.board[1][::-1]:  # Temporarily reverse the second player's pits for display
            print(f"{pit:2} ", end="")
        print()
        print(f"{self.stores[1]:2} ", end="", flush=True)
        print("   " * len(self.board[0]), end="")
        print(f" {self.stores[0]:2}")
        print("  ", end="")
        for pit in self.board[0]:
            print(f"{pit:2} ", end="")
        print()
        print("Current Player:", self.current_player)
        print("Turn Number:", self.turn_number)


def process_input(input_str: str) -> GameState:
    tokens = input_str.split()
    
    if tokens[0] != "STATE":
        raise ValueError("Invalid input format, expected 'STATE' at the beginning.")
    
    N = int(tokens[1])  # Number of pits per player

    p1_pits = np.array(list(map(int, tokens[2:N+2])))  # Extract p1 pits
    p2_pits = np.array(list(map(int, tokens[N+2:2*N+2])))  # Extract p2 pits

    p1_store = int(tokens[2*N+2])  # Extract p1 store
    p2_store = int(tokens[2*N+3])  # Extract p2 store

    turn = int(tokens[2*N+4])  # Extract turn number
    currentplayer = int(tokens[2*N+5])  # Extract current player

    return GameState(p1_pits, p2_pits, p1_store, p2_store, turn, currentplayer)

def nextmove(game_state: GameState) -> int:
    pass

def minimax(state: GameState, depth: int, alpha: float, beta: float, maximizing_player: bool) -> float:
    '''
        Implements the minimax algorithm with alpha-beta pruning.
        Parameters:
        state (GameState): The current state of the game.
        depth (int): The maximum depth to search in the game tree.
        alpha (float): The best value that the maximizer currently can guarantee at that level or above.
        beta (float): The best value that the minimizer currently can guarantee at that level or above.
        maximizing_player (bool): A boolean indicating whether the current move is by the maximizing player.
        Returns:
        float: The heuristic value of the board state.    
    '''
    if depth == 0 or is_terminal(state):
        return evaluate_state(state)

    if maximizing_player:
        max_eval = float("-inf")
        for move in get_valid_moves(state):
            new_state = simulate_move(state, move)
            eval = minimax(new_state, depth - 1, alpha, beta, False)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:  # Alpha-beta pruning
                break
        return max_eval
    else:
        min_eval = float("inf")
        for move in get_valid_moves(state):
            new_state = simulate_move(state, move)
            eval = minimax(new_state, depth - 1, alpha, beta, True)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:  # Alpha-beta pruning
                break
        return min_eval

def get_valid_moves(state: GameState) -> list[int]:
    pass

def simulate_move(state: GameState, move: int) -> GameState:
    pass

def evaluate_state(state: GameState) -> float:
    #This function is the heuristic function that evaluates the state of the game
    pass

def is_terminal(state: GameState) -> bool:
    #checks if the game is over
    pass


    

if __name__ == "__main__":
    input_str = "STATE 8 3 3 2 1 5 6 2 4 4 5 6 1 5 1 2 2 1 13 24 2"
    
    start_time = time.time()
    gamestate = process_input(input_str)
    gamestate.display_board()
    end_time = time.time()
    
    print(f"Execution time: {end_time - start_time} seconds")