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

    def duplicate(self):
        return GameState(self.board[0].copy(), self.board[1].copy(), self.stores[0], self.stores[1], self.turn_number, self.current_player)


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
    """
    Determines the best move for the current player using Minimax with Alpha-Beta pruning.
    Returns the pit index of the best move.
    """
    best_move = None
    best_value = float("-inf") if game_state.current_player == 1 else float("inf")
    alpha, beta = float("-inf"), float("inf")
    depth = 5 #TODO: ADJUST ON THE BASIS OF TIME FOR COMPUTATION

    for move in get_valid_moves(game_state):
        new_state = simulate_move(game_state, move)
        move_value = minimax(new_state, depth, alpha, beta, maximizing_player=(game_state.current_player == 1))

        if game_state.current_player == 1: # Maximizing player (Player 1)
            if move_value > best_value:
                best_value = move_value
                best_move = move
            alpha = max(alpha, best_value)
        else:  # Minimizing player (Player 2)
            if move_value < best_value:
                best_value = move_value
                best_move = move
            beta = min(beta, best_value)

    return best_move


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
    current_pits = state.board[state.current_player - 1]
    return [i for i in range(len(current_pits)) if current_pits[i] > 0]

def simulate_pie(state: GameState, player: int) -> GameState:
    new_p1_pits = state.board[1].copy()
    new_p2_pits = state.board[0].copy()
    new_p1_store = state.stores[1]
    new_p2_store = state.stores[0]
    new_turn_number = state.turn_number + 1
    new_current_player = 1

    return GameState(new_p1_pits, new_p2_pits, new_p1_store, new_p2_store, new_turn_number, new_current_player)

def simulate_move(state: GameState, move: int) -> GameState:
    if move == len(state.board[0]) + 1:
        return simulate_pie(state, state.current_player)

    new_p1_pits = state.board[0].copy()
    new_p2_pits = state.board[1].copy()
    new_stores = state.stores.copy()

    pits = [new_p1_pits, new_p2_pits]
    player_index = 0 if state.current_player == 1 else 1
    pit_index = move - 1
    stones = pits[player_index][pit_index]
    pits[player_index][pit_index] = 0

    # Increase index by +7 for Player 2 so stones are distributed properly
    index = pit_index if player_index == 0 else pit_index + 7

    while stones > 0:
        index = (index + 1) % 14  # Move counterclockwise through indices 0-13

        # Skip the opponent's store
        if (player_index == 0 and index == 13) or (player_index == 1 and index == 6):
            index = (index + 1) % 14

        # Place stones
        if index < 6:  
            pits[0][index] += 1  # Player 1's pits
        elif index == 6:  
            if player_index == 0:  # Player 1's store
                new_stores[0] += 1
        elif index < 13:  
            pits[1][index - 7] += 1  # Player 2's pits
        elif index == 13:  
            if player_index == 1:  # Player 2's store
                new_stores[1] += 1

        stones -= 1

    # Stone Captures: If last stone landed in an empty pit on current player's side
    if player_index == 0 and index < 6 and pits[0][index] == 1:
        opposite_index = 12 - index
        if pits[1][opposite_index - 7] > 0:
            new_stores[0] += pits[1][opposite_index - 7] + pits[0][index]
            pits[1][opposite_index - 7] = 0
            pits[0][index] = 0
            print("Player 1 has captured stones from Player 2")

    elif player_index == 1 and 7 <= index < 13 and pits[1][index - 7] == 1:
        opposite_index = 12 - index
        if pits[0][opposite_index - 7] > 0:
           new_stores[1] += pits[0][opposite_index - 7] + pits[1][index - 7]
           pits[0][opposite_index - 7] = 0
           pits[1][index - 7] = 0
           print("Player 2 has captured stones from Plkayer 1")

    #Check if last stone landed in current player's store
    if (player_index == 0 and index == 6) or (player_index == 1 and index == 13):
        next_player = state.current_player
        print("Player granted additional turn")
    else:
        next_player = 3 - state.current_player
    

    return GameState(new_p1_pits, new_p2_pits, new_stores[0], new_stores[1], state.turn_number + 1, next_player)

def evaluate_state(state: GameState) -> float:
    #our heuristic function
    if(state.current_player == 1):
        return state.stores[0] - state.stores[1]
    else:
        return state.stores[1] - state.stores[0]

def is_terminal(state: GameState) -> bool:
    #checks if the game is over
    if np.sum(state.board[0]) == 0 or np.sum(state.board[1]) == 0:
        return True
    else:
        return False


    

if __name__ == "__main__":
    input_str = "STATE 8 3 3 2 1 5 6 2 4 4 5 6 1 5 1 2 2 1 13 24 2"
    
    start_time = time.time()
    gamestate = process_input(input_str)
    gamestate.display_board()
    end_time = time.time()
    
    print(f"Execution time: {end_time - start_time} seconds")
    
    
'''

USED FOR TESTING

if __name__ == "__main__":
    # Test Case: Player 1 selects pit 6
    initial_state = GameState(
        np.array([4, 3, 4, 6, 4, 9]),  # Player 1 pits
        np.array([4, 4, 4, 4, 4, 8]),  # Player 2 pits
        0,  # Player 1 store
        0,  # Player 2 store
        1,  # Turn number
        1   # Current player (Player 1)
    )

    print("Before move:")
    initial_state.display_board()

    # Player 1 chooses pit 6 (index 5)
    new_state = simulate_move(initial_state, 3)
    #new_state2 = simulate_move(new_state, 6)


    print("\nAfter move:")
    new_state.display_board()
    new_state2 = simulate_move(new_state, 6)
    new_state2.display_board()
'''
    
