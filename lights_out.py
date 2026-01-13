from shutil import move
import ained
import numpy as np

# The quiet patterns from deterministic lights out theory
quiet_patterns = ([
    [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
    [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
    [0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0]
])

# The pseudo-inverse matrix used to find a solution matrix in the deterministic version of lights out 
a_inv = np.array([
    [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0],
    [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
    [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0],
    [0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0],
    [0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0],
    [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0],
    [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
])

class Board:
    def __init__(self, N):
        self.N = N
        # Creating the AiNed instance
        self.ai = ained.AiNed()
        self.generate_board() # Random board state
    
    # Generating a random board state untill one is found that is solvable in the deterministic version
    def generate_board(self):
        while True:
            print("Generating a board...")
            board = np.random.randint(0, 2, self.N * self.N).tolist()
            if self.is_solvable(board):
                self.ai.reconstruct_board(board, 0, 0, self.N, self.N)
                self.ai.commit()
                return

    # Checking if the board is solvable in the deterministic version using the quiet patterns
    def is_solvable(self, board):
        for qp in quiet_patterns:
            overlap = np.sum((np.array(board) * np.array(qp)))
            if not overlap % 2 == 0:
                return False
        return True
    
    # Finding a solution pattern using the pseudo-inverse matrix
    def basic_solve(self):
        board = self.ai.get_board(0, 0, self.N, self.N)
        solution_vector = np.dot(a_inv, np.array(board).reshape(self.N * self.N)) % 2 # % 2 possible because of modulo 2 arithmetic
        return solution_vector.tolist()

    # Finding the solution pattern with the least number of presses based on the basic solve pattern and the quiet patterns
    def optimal_solve(self):
        basic_pattern = self.basic_solve()
        best_pattern = basic_pattern
        for qp in quiet_patterns:
            # Taking the sum of the solve pattern and the quiet pattern modulo 2
            result = [(a + b) % 2 for a, b in zip(basic_pattern, qp)]
            # Accept if the new solve pattern has less presses
            if sum(result) < sum(best_pattern):
                best_pattern = result
        return np.array(best_pattern).reshape(self.N, self.N)
    
    # Returns the move that maximizes the probability of turing lights off and minizes the probability of turning lights on using the coefficient matrix
    def greedy_move(self):
        values = np.zeros((self.N, self.N))
        board = np.array(self.ai.get_board(0, 0, self.N, self.N)).reshape(self.N, self.N)
        # The negative image of the board: inverting 0s to 1s and 1s to 0s
        negative_board = (board + 1)%2 # % 2 possible because of modulo 2 arithmetic
        # Calculating the value of each possible move
        for row in range(self.N):
            for col in range(self.N):
                coeffs = self.get_coefficients(row, col)
                # Positive effect: probability ofturning lights off
                positive = board * coeffs
                # Negative effect: probability of turning lights on
                negative = negative_board * coeffs
                values[row][col] = np.sum(positive) - np.sum(negative)
        # Selecting the move with the highest value
        move = np.unravel_index(np.argmax(values), values.shape)
        return move

    # Getting the coefficient matrix for a specific move
    def get_coefficients(self, x, y):
        coeffs = self.ai.get_coefficients()
        press_coeffs = np.zeros((self.N, self.N))
        for row in range(self.N):
            for col in range(self.N):
                row_diff = abs(row - x)
                col_diff = abs(col - y)
                press_coeffs[row][col] = coeffs[row_diff * 5 + col_diff]
                if row_diff + col_diff == 1: # Part of the cross
                    press_coeffs[row][col] = 1

        return press_coeffs

    # Printing the current board state and the optimal (deterministic) solve pattern next to each other
    def print_board(self):
        board = np.array(self.ai.get_board(0, 0, self.N, self.N)).reshape(self.N, self.N)
        solve_pattern = self.optimal_solve()
        print("Board:        Optimal solve pattern:")
        for row_b, row_s in zip(board, solve_pattern):
            print(' '.join(str(cell) for cell in row_b), "   ", ' '.join(str(cell) for cell in row_s))

# Mode where the user manually inputs the moves
# Returns the number of moves taken to solve the board
def play_manually(my_board, N):
    counter = 0
    while my_board.ai.game_not_over(0, 0, N, N):
        my_board.print_board()
        move = input("Enter your move as 'row col' (or 'q' to quit): ")
        if move.lower() == 'q':
            print("Exited game.")
            break
        try:
            row, col = map(int, move.split())
            if 1 <= row <= N and 1 <= col <= N:
                # Play users move
                my_board.ai.flip_lights(0, 0, N, N, row-1, col-1)
                counter += 1
            else:
                print("Invalid move. Please enter values between 1 and", N)
        except ValueError:
            print("Invalid input. Please enter two integers.")
    my_board.print_board()
    print("Congratulations! You've solved the puzzle in", counter, "moves.")
    return counter

# mode where it automatically solves the board using a strategy that is efficient for deterministic games
# Returns the number of moves taken to solve the board
def deterministic_strategy(my_board, N):
    counter = 0
    while my_board.ai.game_not_over(0, 0, N, N):
        #my_board.print_board()
        solve_pattern = my_board.optimal_solve()
        pressed = False
        for row in range(N):
            for col in range(N):
                if solve_pattern[row][col] == 1:
                    my_board.ai.flip_lights(0, 0, N, N, row, col)
                    counter += 1
                    pressed = True
                    # Recalculate the solve pattern after each move
                    break
            # Recalculate the solve pattern after each move
            if pressed:
                break
        # If the solve pattern is all zeros, but board is not solved (can happen for deterministically unsolvable board states in stochastic games),
        # make a random fallback move
        if not pressed:
            my_board.ai.flip_lights(0, 0, N, N, np.random.randint(0, N), np.random.randint(0, N))
    my_board.print_board()
    print("Congratulations! You've solved the puzzle in", counter, "moves.")
    return counter

# mode where it automatically solves the board using a greedy strategy
# Returns the number of moves taken to solve the board
def greedy_strategy(my_board, N):
    counter = 0
    while my_board.ai.game_not_over(0, 0, N, N):
        #my_board.print_board()
        # Getting the most valuable move according to the greedy algorithm
        move = my_board.greedy_move()
        my_board.ai.flip_lights(0, 0, N, N, move[0], move[1])
        counter += 1
    my_board.print_board()
    print("Congratulations! You've solved the puzzle in", counter, "moves.")
    return counter

def chase_strategy(my_board, N):
    counter = 0
    while my_board.ai.game_not_over(0, 0, N, N):
        #my_board.print_board()
        pressed = False
        for row in range(N):
            for col in range(N):
                board = my_board.ai.get_board(0, 0, N, N)
                if board[row * N + col] == 1:
                    if row < N - 1:
                        my_board.ai.flip_lights(0, 0, N, N, row + 1, col)
                    elif col < N - 1:
                        my_board.ai.flip_lights(0, 0, N, N, row, col + 1)
                    else:
                        my_board.ai.flip_lights(0, 0, N, N, row, col)
                    counter += 1
                    pressed = True
                    break
            if pressed:
                break
    my_board.print_board()
    print("Congratulations! You've solved the puzzle in", counter, "moves.")
    return counter

# Running a specified number of games based on a manhattan coefficient factor, reach and strategy and returning the results
def run_test(N, manhattan_factor, reach, strategy, runs):
    results = np.array([])
    my_board = Board(N)
    my_board.ai.set_coefficients_manhattan(manhattan_factor, reach)
    my_board.ai.print_coefficients()
    for _ in range(runs):
        # Generating a new starting pattern for each run
        my_board.generate_board()
        # Using the strategy chosen by the user
        count = strategy(my_board, N)
        results = np.append(results, count)
    mean = np.mean(results)
    median = np.median(results)
    p10 = np.percentile(results, 10)
    p25 = np.percentile(results, 25)
    p75 = np.percentile(results, 75)
    p90 = np.percentile(results, 90)
    
    print(f"{runs} runs completed.")
    print(f"Strategy: {strategy.__name__}, Factor: {manhattan_factor}, Reach: {reach}")
    print(f"Mean: {mean}, Median: {median} moves, P10: {p10}, P25: {p25}, P75: {p75}, P90: {p90}")

# Letting the user input the manhattan coefficient factor and reach
def user_interface():
    while True:
        try:
            manhattan_factor = float(input("Enter Manhattan coefficient factor (0 to 1): "))
            if 0 <= manhattan_factor <= 1:
                break
            else:
                print("Factor must be between 0 and 1.")
        except ValueError:
            print("Invalid input. Please enter a number between 0 and 1.")
    
    while True:
        try:
            reach_input = input("Enter maximum reach (non-negative whole number, or 'n' for no limit): ").lower().strip()
            if reach_input == 'n' or reach_input == 'no':
                reach = 999999 # Practically no limit
                break
            else:
                reach = int(reach_input)
                if reach >= 0:
                    break
                else:
                    print("Reach cannot be negative.")
        except ValueError:
            print("Invalid input. Please enter a non-negative whole number or 'n' for no limit.")
    return manhattan_factor, reach

# Running a single game using the strategy chosen by the user
def single_game():
    N = 5
    my_board = Board(N)

    manhattan_factor, reach = user_interface()
    
    my_board.ai.set_coefficients_manhattan(manhattan_factor, reach)
    my_board.ai.print_coefficients()

    mode = input("Choose strattegy: (1) deterministic, (2) greedy, (3) chasing, (4) play manually: ")
    if mode == '1':
        deterministic_strategy(my_board, N)
    elif mode == '2':
        greedy_strategy(my_board, N)
    elif mode == '3':
        chase_strategy(my_board, N)
    elif mode == '4':
        play_manually(my_board, N)

# Main loop for user interaction, where user can choose to play a single game or run a number of games to get results
def main():
    while True:
        choice = input("What do you want to do? - (1) Single game, (2) Run test, (q) Quit: ")
        if choice == '1':
            single_game()
        elif choice == '2':
            N = 5
            manhattan_factor, reach = user_interface()
            runs = int(input("Enter number of test runs: "))
            strategy_choice = input("Choose a strategy: (1) deterministic, (2) greedy, (3) chasing: ")
            if strategy_choice == '1':
                strategy = deterministic_strategy
            elif strategy_choice == '2':
                strategy = greedy_strategy
            elif strategy_choice == '3':
                strategy = chase_strategy
            else:
                print("Invalid strategy choice.")
                continue
            run_test(N, manhattan_factor, reach, strategy, runs)
        elif choice.lower() == 'q':
            print("Exiting the game. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()