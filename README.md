# sudokuNet
This project uses PyTorch to create a ResNet-based Convolutional Neural Network with the goal of solving sudoku puzzles. SudokuSolver.py initializes and trains this model with easy modification of parameters through command line arguments. test_model.py allows for easy testing of any pretrained model on a random selection of puzzles, printing useful evaluation statistics and displaying model architecture with Tensorboard.\
\
My best model was trained on an RTX 2080 Ti with 20 epochs, 2000000 example puzzles, and a batch size of 256.\
Recent output for test_model.py with this model:\
\
\==================================================\
SUMMARY STATISTICS\
\==================================================\
Average accuracy for empty cells: 96.78%\
Median accuracy for empty cells: 100.00%\
Complete puzzles solved correctly: 72/100 (72.00%)\
Minimum accuracy: 57.14%\
Maximum accuracy: 100.00%\
\
Loosely inspired by https://www.geeksforgeeks.org/sudoku-solver-using-tensorflow/ \
Training data from https://www.kaggle.com/datasets/rohanrao/sudoku/data \
Due to issues with GitHub storage limits for Git LFS, pretrained models cannot be stored in this repository. I have uploaded some of my models to my personal OneDrive here: https://1drv.ms/u/c/e775e84bf4865dfa/EaFXLLj_c_pNkJkjrcsw8y0BH-msC_9qxptkYgHMg85MxA?e=J5iwLj