import numpy as np

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
    

if __name__ == "__main__":
    input_str = "STATE 8 3 3 2 1 5 6 2 4 4 5 6 1 5 1 2 2 1 13 24 2"
    gamestate = process_input(input_str)
    gamestate.display_board()