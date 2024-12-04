import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import time
from tqdm import tqdm
import logging
import torch.nn.functional as F
import argparse
from azureml.core import Run

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    filename='sudoku_training.log'
)

class SudokuDataset(Dataset):
    def __init__(self, data, sample_size=None, augment=True):
        """
        Load Sudoku dataset with optional sampling
        data can be either a CSV file path or a pandas DataFrame
        """
        if isinstance(data, str):
            self.data = pd.read_csv(data)
        else:
            self.data = data
        
        # Optional sampling for testing/faster iteration
        if sample_size:
            self.data = self.data.sample(n=sample_size)
        
        # Reset index after sampling to ensure continuous indexing
        self.data = self.data.reset_index(drop=True)
        
        self.augment = augment
    
    def augment_puzzle(self, puzzle, solution):
        # Randomly rotate/flip
        k = np.random.randint(4)  # 0-3 rotations
        if np.random.random() > 0.5:
            puzzle = np.flip(puzzle, axis=0)
            solution = np.flip(solution, axis=0)
        if np.random.random() > 0.5:
            puzzle = np.flip(puzzle, axis=1)
            solution = np.flip(solution, axis=1)
        puzzle = np.rot90(puzzle, k).copy()
        solution = np.rot90(solution, k).copy()
        return puzzle, solution
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        puzzle = np.array(list(self.data.loc[idx, 'puzzle']), dtype=np.float32).reshape(9, 9)
        solution = np.array(list(self.data.loc[idx, 'solution']), dtype=np.float32).reshape(9, 9)
        
        if self.augment:
            puzzle, solution = self.augment_puzzle(puzzle, solution)
        
        # Subtract 1 from solution to convert 1-9 range to 0-8 range
        solution = solution - 1
        
        # One-hot encode the input (keep as is since puzzle needs all 10 classes for 0-9)
        puzzle_one_hot = F.one_hot(torch.tensor(puzzle, dtype=torch.long), num_classes=10)
        
        return puzzle_one_hot.float(), torch.tensor(solution, dtype=torch.long)

class SudokuSolver(nn.Module):
    def __init__(self):
        super(SudokuSolver, self).__init__()
        # Initial embedding
        channels = 512;
        self.embedding = nn.Sequential(
            nn.Conv2d(10, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        
        # Residual blocks for better feature extraction
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(8)
        ])
        
        # Global context block
        self.global_context = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=9),  # Global receptive field
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, 9, kernel_size=1)
        )
    
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.embedding(x)
        
        # Apply residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Global context
        context = self.global_context(x)
        x = x + context
        
        x = self.final(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

def sudoku_constraints_loss(outputs, device):
    # outputs shape is [batch_size, 9, 9, 9] (channels last)
    outputs = outputs.permute(0, 3, 1, 2)  # Change to [batch_size, 9, 9, 9]
    
    # Row constraint
    row_loss = -torch.log(torch.sum(torch.softmax(outputs, dim=1), dim=2)).mean()
    
    # Column constraint
    col_loss = -torch.log(torch.sum(torch.softmax(outputs, dim=1), dim=3)).mean()
    
    # 3x3 box constraint
    box_loss = 0
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            box = outputs[:, :, i:i+3, j:j+3]
            box_loss += -torch.log(torch.sum(torch.softmax(box.reshape(outputs.size(0), 9, -1), dim=2), dim=1)).mean()
    
    return row_loss + col_loss + box_loss

def train_batch(model, batch, optimizer, criterion, device):
    puzzles, solutions = batch
    puzzles = puzzles.to(device)
    solutions = solutions.to(device)
    
    optimizer.zero_grad(set_to_none=True)
    
    with torch.amp.autocast('cuda'):
        outputs = model(puzzles)
        # Calculate constraints loss before reshaping for main loss
        constraints_loss = sudoku_constraints_loss(outputs, device)
        
        # Reshape for main loss calculation
        targets = solutions.reshape(-1)
        outputs = outputs.permute(0,2,3,1).reshape(-1,9)
        
        loss = criterion(outputs, targets) + 0.1 * constraints_loss
    
    return loss.item()

def validate_batch(model, batch, criterion, device):
    puzzles, solutions = batch
    puzzles = puzzles.to(device)
    solutions = solutions.to(device)
    
    outputs = model(puzzles)
    targets = solutions.reshape(-1)
    outputs = outputs.permute(0,2,3,1).reshape(-1,9)
    
    loss = criterion(outputs, targets)
    accuracy = calculate_accuracy(outputs, targets)
    return loss.item(), accuracy.item()

def calculate_accuracy(outputs, targets):
    predictions = outputs.argmax(dim=1)
    correct = (predictions == targets).float().sum()
    return correct / targets.size(0)

def train_sudoku_solver(model, train_loader, val_loader, epochs=20, learning_rate=0.001, checkpoint_dir='checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Use a combination of losses for better accuracy
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)  # Changed to AdamW
    
    # Use OneCycleLR instead of cosine annealing
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=25
    )
    
    # Add gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    best_val_loss = float('inf')
    best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
    latest_checkpoint_path = os.path.join(checkpoint_dir, 'latest_model.pth')
    patience = 10
    patience_counter = 0
    
    scaler = torch.amp.GradScaler('cuda')  # For mixed precision training
    start_time = time.time()

    # Get Azure ML run context
    run = Run.get_context()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', unit='batch')
        
        for batch in train_progress:
            puzzles, solutions = batch
            puzzles = puzzles.to(device)
            solutions = solutions.to(device)
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            # Use mixed precision training
            with torch.amp.autocast('cuda'):
                outputs = model(puzzles)
                constraints_loss = sudoku_constraints_loss(outputs, device)
                
                # Reshape for main loss calculation
                outputs_reshaped = outputs.permute(0,2,3,1).reshape(-1,9)
                targets = solutions.reshape(-1)
                
                loss = criterion(outputs_reshaped, targets) + 0.1 * constraints_loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_progress.set_postfix({'Loss': loss.item()})
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation', unit='batch'):
                with torch.cuda.amp.autocast():
                    val_batch_loss, val_batch_accuracy = validate_batch(model, batch, criterion, device)
                val_loss += val_batch_loss
                val_accuracy += val_batch_accuracy
                num_batches += 1
        
        # Average metrics
        train_loss /= len(train_loader)
        val_loss /= num_batches
        val_accuracy /= num_batches
        
        # Log results
        logging.info(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        # Learning rate scheduling
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f'Learning rate: {current_lr:.2e}')
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            }, best_checkpoint_path)
            logging.info(f'Saved new best model with val_loss: {val_loss:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f'Early stopping triggered after {epoch+1} epochs')
                break

        run.log('train_loss', train_loss)
        run.log('val_loss', val_loss)
        run.log('val_accuracy', val_accuracy)
    
    # Total training time
    total_time = time.time() - start_time
    logging.info(f'Total Training Time: {total_time/3600:.2f} hours')
    
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/sudoku.csv')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--sample_size', type=int, default=10000)
    args = parser.parse_args()
    
    
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0).lower()
        print(f"Found GPU: {gpu_name}")
        batch_size = args.batch_size
        epochs = args.epochs
        sample_size = args.sample_size
    else:
        print("No GPU detected - using CPU settings")
        batch_size = max(64, args.batch_size)
        epochs = min(5, args.epochs)
        sample_size = min(16384, args.sample_size)
    print("Batch size: " + str(batch_size) + " | Epochs: " + str(epochs) + " | Training sample: " + str(sample_size))
    
    
    # Load dataset with adjusted size
    dataset = pd.read_csv(args.data_path)
    dataset = dataset.sample(n=sample_size, random_state=42)
    
    # Use more training data
    train_idx = int(len(dataset) * 0.95)
    train_data = dataset.iloc[:train_idx]
    val_data = dataset.iloc[train_idx:]
    
    train_dataset = SudokuDataset(train_data)
    val_dataset = SudokuDataset(val_data)
    
    # Optimize workers
    num_workers = os.cpu_count()
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        prefetch_factor=2 
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # Initialize model
    model = SudokuSolver()
    
    # Train with enhanced parameters
    trained_model = train_sudoku_solver(
        model, 
        train_loader, 
        val_loader,
        epochs,
        learning_rate=0.001,
        checkpoint_dir='checkpoints'
    )
    
    # Save final model
    torch.save(trained_model.state_dict(), 'pretrained/solver_last.pth')
    torch.save(trained_model.state_dict(), 'pretrained/solver_' + str(epochs) + 'x' + str(sample_size) + '.pth')

        

if __name__ == "__main__":
    main()