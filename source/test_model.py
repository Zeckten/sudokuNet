import torch
import numpy as np
from SudokuSolver import SudokuSolver
import pandas as pd

def load_model(model_path):
    """Load the trained model."""
    model = SudokuSolver()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_sudoku(model, puzzle):
    """Make a prediction for a single Sudoku puzzle."""
    # Convert puzzle to tensor and one-hot encode it
    puzzle_tensor = torch.tensor(puzzle, dtype=torch.long)
    puzzle_one_hot = torch.nn.functional.one_hot(puzzle_tensor, num_classes=10).float()
    puzzle_one_hot = puzzle_one_hot.unsqueeze(0)  # Add batch dimension
    
    # Get prediction
    with torch.no_grad():
        outputs = model(puzzle_one_hot)
        prediction = outputs.argmax(dim=1).squeeze(0)  # Get most likely number (0-8)
        prediction = prediction + 1  # Convert back to 1-9 range
    
    # Convert prediction to numpy array
    prediction = prediction.numpy()
    
    # Keep original puzzle numbers
    mask = puzzle != 0
    prediction[mask] = puzzle[mask]
    
    return prediction

def calculate_accuracy(original, prediction, solution):
    """Calculate accuracy metrics for the prediction."""
    # Calculate accuracy only for cells that were empty in original puzzle
    empty_cells = (original == 0)
    correct_predictions = (prediction[empty_cells] == solution[empty_cells])
    accuracy = correct_predictions.sum() / empty_cells.sum()
    
    # Add complete puzzle accuracy
    complete_correct = (prediction == solution).all()
    
    return accuracy, complete_correct

def main():
    # Load the trained model
    model_path = 'sudoku_solver_final.pth'
    try:
        model = load_model(model_path)
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"Error: Could not find model file at {model_path}")
        return
    
    # Load test puzzles from the dataset
    try:
        # Test 100 puzzles instead of 5
        test_data = pd.read_csv('data/sudoku.csv').sample(n=100, random_state=42)
    except FileNotFoundError:
        print("Error: Could not find test data file")
        return
    
    # Track accuracies
    accuracies = []
    complete_correct = 0
    
    # Test each puzzle
    for idx, row in test_data.iterrows():
        print(f"\nTesting Puzzle {idx + 1}")
        print("-" * 40)
        
        # Convert string puzzle to numpy array
        original_puzzle = np.array(list(row['puzzle']), dtype=np.float32).reshape(9, 9)
        solution = np.array(list(row['solution']), dtype=np.float32).reshape(9, 9)
        
        # Make prediction
        prediction = predict_sudoku(model, original_puzzle)
        
        # Calculate accuracy
        accuracy, is_complete = calculate_accuracy(original_puzzle, prediction, solution)
        accuracies.append(accuracy)
        if is_complete:
            complete_correct += 1
        
        # Print results
        print("Original Puzzle:")
        print_sudoku(original_puzzle.astype(int))
        print("\nModel's Solution:")
        print_sudoku(prediction.astype(int))
        print("\nTrue Solution:")
        print_sudoku(solution.astype(int))
        print(f"\nAccuracy for empty cells: {accuracy:.2%}")
        print(f"Complete puzzle solved correctly: {'Yes' if is_complete else 'No'}")

    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    print(f"Average accuracy for empty cells: {np.mean(accuracies):.2%}")
    print(f"Median accuracy for empty cells: {np.median(accuracies):.2%}")
    print(f"Complete puzzles solved correctly: {complete_correct}/{len(test_data)} ({complete_correct/len(test_data):.2%})")
    print(f"Minimum accuracy: {np.min(accuracies):.2%}")
    print(f"Maximum accuracy: {np.max(accuracies):.2%}")

def print_sudoku(grid):
    """Pretty print a Sudoku grid."""
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("-" * 21)
        for j in range(9):
            if j % 3 == 0 and j != 0:
                print("|", end=" ")
            print(int(grid[i][j]), end=" ")
        print()

if __name__ == "__main__":
    main() 