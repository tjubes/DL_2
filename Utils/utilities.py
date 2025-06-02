# utilities.py
import h5py
import os
from pathlib import Path
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sys
from torch.utils.data import Dataset, DataLoader, Subset
import random
from sklearn.model_selection import train_test_split, StratifiedKFold

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_dataset_name(filename_with_dir):
    filepath = Path(filename_with_dir)
    filename_without_dir = filepath.name
    temp = filename_without_dir.split('_')[:-1]
    dataset_name = "_".join(temp)
    return dataset_name


def get_filepath():
    # Get the full path of this script
    current_path = os.path.abspath(__file__)
    
    # Go up until we find the DL_2 folder and stop there
    parts = current_path.split(os.sep)
    if 'DL_2' in parts:
        dl2_index = parts.index('DL_2')
        modified_path = os.sep.join(parts[:dl2_index])
    else:
        # If DL_2 not in path, assume script is already at or below project root
        modified_path = os.path.dirname(os.path.dirname(current_path))
        
    # Check if the Cross and Intra folders are extracted
    if not os.path.exists(os.path.join(modified_path, 'Cross')):
        modified_path = os.path.join(modified_path, 'Final Project data')
        
    if not os.path.exists(os.path.join(modified_path, 'Cross')):
        modified_path = os.path.join(modified_path, 'Final Project data')
        
    return modified_path



def read_h5py_file(filename_path):
    with h5py.File(filename_path, 'r') as f:
        dataset_name = get_dataset_name(filename_path)
        matrix = f.get(dataset_name)[()]
        # print(f"Loaded: {type(matrix)}, Shape: {matrix.shape}")
        return matrix
    



def load_h5_files(directory_path, max_files=None):
    """
    Load raw MEG data and labels from a directory of h5 files.
    
    Returns:
        List of raw data matrices and labels
    """
    data_list = []
    labels = []

    all_files = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.h5'):
            all_files.append(filename)

    if max_files is not None:
        all_files = all_files[:max_files]

    for file in all_files:
        if file.startswith("task_"):
            parts = file.split('_')
            task_type = '_'.join(parts[:2])  # task_motor, task_working
        else:
            task_type = file.split('_')[0]  # rest

        file_path = os.path.join(directory_path, file)
        matrix = read_h5py_file(file_path)

        data_list.append(matrix)
        labels.append(task_type)

    return data_list, labels




def normalize_meg_data(data_list, downsample_factor=None):
    """
    Normalize a list of MEG matrices using z-score normalization per channel.
    
    Returns:
        List of normalized (and optionally downsampled) data
    """
    normalized_data = []

    for matrix in data_list:
        if downsample_factor is not None:
            matrix = matrix[:, ::downsample_factor]

        norm_matrix = zscore(matrix, axis=1, nan_policy='propagate')
        norm_matrix = np.nan_to_num(norm_matrix, nan=0.0)

        normalized_data.append(norm_matrix)

    return normalized_data

class MEGDataset(Dataset):
    """Custom Dataset for MEG data"""
    def __init__(self, data_list, labels, label_map):
        self.data_list = data_list
        self.labels = labels
        self.label_map = label_map 

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]        
        data = torch.FloatTensor(data).unsqueeze(0)        
        label = torch.LongTensor([self.label_map[self.labels[idx]]]).squeeze()
        return data, label


def plot_meg_sample(sample, label, num_channels=10):
    """
    Plot the first few MEG sensor time series from a single sample.
    
    Parameters:
    - sample (np.array): The MEG data matrix of shape (248, time_steps)
    - label (str): The task label for the sample
    - num_channels (int): Number of sensor channels to plot
    """
    plt.figure(figsize=(12, 6))
    time_steps = sample.shape[1]
    
    for i in range(num_channels):
        plt.plot(np.arange(time_steps), sample[i] + i * 10, label=f'Channel {i}')
    
    plt.title(f'MEG Sample - Label: {label}')
    plt.xlabel('Time Steps')
    plt.ylabel('Amplitude (offset for clarity)')
    plt.grid(True)
    plt.show()

def create_dataloaders(data, labels, label_map, batch_size=4, suffle=True, num_workers=0):  
    """Create PyTorch DataLoaders from your loaded data"""    
    # Create dataset
    dataset = MEGDataset(data, labels, label_map)
    # Create dataloaders
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=suffle, num_workers=num_workers)    
    return data_loader


def train_one_epoch(model, dataloader, criterion, optimizer, device='cpu'):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0  
    correct = 0 
    total = 0 
    
    for inputs, labels in dataloader: 
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate_model(model, dataloader, criterion, device='cpu'):
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = len([1 for i, j in zip(all_predictions, all_labels) if i == j]) / len(all_labels)
    
    return avg_loss, accuracy, all_predictions, all_labels



class EarlyStopper:
    def __init__(self, patience=15):
        self.patience = patience
        self.best_score = None
        self.counter = 0

    def check(self, current_score):
        """
        Returns True if training should stop (i.e., patience exceeded).
        """
        if self.best_score is None or current_score > self.best_score:
            self.best_score = current_score
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
        


def train_val_experiment(
    train_loader,
    val_loader,
    model_fn,
    label_map,
    lr=1e-3, 
    weight_decay=1e-5,
    epochs=50, 
    seed=42
):
    """
    Train/Val experiment using pre-built DataLoaders.
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        model_fn: function that returns a new model instance
        label_map: dict mapping class labels to indices (not used here but kept for consistency)
        optimizer: optional custom optimizer
        criterion: optional custom loss function
        epochs: max number of training epochs
        seed: random seed for reproducibility
    Returns:
        dict with training/validation loss & accuracy history and best val acc
    """

    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_fn().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    early_stopper = EarlyStopper(patience=10)
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    best_val_acc = 0.0

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")

        if early_stopper.check(val_acc):
            print(f"Early stopping at epoch {epoch}")
            break

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc
    }




def analyze_train_val_results(results):
    """
    Analyze and visualize training and validation results.

    Args:
        results: dictionary returned by train_val_experiment()
    """
    import matplotlib.pyplot as plt

    print(f"\n{'='*60}")
    print("  TRAIN/VALIDATION RESULTS SUMMARY")
    print(f"{'='*60}")

    print(f"\n📈 Best Validation Accuracy: {results['best_val_acc']:.4f}")

    # Plot training and validation loss
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(results['train_losses'], label='Train Loss')
    plt.plot(results['val_losses'], label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(results['train_accs'], label='Train Accuracy')
    plt.plot(results['val_accs'], label='Val Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()




def cross_validation_experiment(
    all_data, 
    all_labels, 
    label_map,
    model_fn,  # <- model factory function passed as an argument
    n_splits=4, 
    epochs=50, 
    batch_size=4
):
    """Complete cross-validation experiment using a model factory function"""

    # Convert labels to indices
    label_indices = [label_map[label] for label in all_labels]

    # Initialize cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Store results
    fold_results = []
    all_histories = []

    print(f"\n{'='*60}")
    print(f"  CROSS-VALIDATION EXPERIMENT ({n_splits} folds)")
    print(f"{'='*60}")
    print(f"Total samples: {len(all_data)}")
    print(f"Samples per fold - Train: ~{len(all_data) * (n_splits-1) // n_splits}, Val: ~{len(all_data) // n_splits}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_data, label_indices)):
        print(f"\n--- FOLD {fold + 1}/{n_splits} ---")

        # Create datasets
        full_dataset = MEGDataset(all_data, all_labels, label_map)
        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_idx)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Initialize a fresh model using the factory function
        model = model_fn()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        train_losses, train_accs = [], []
        val_losses, val_accs = [], []

        early_stopper = EarlyStopper(patience=10)
        best_val_acc = 0.0

        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, device)

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:2d}: Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}, Val Loss={val_loss:.3f}")

            if early_stopper.check(val_acc):
                print(f"Early stopping at epoch {epoch}")
                break

            if val_acc > best_val_acc:
                best_val_acc = val_acc

        final_val_loss, final_val_acc, val_preds, val_true = evaluate_model(model, val_loader, criterion, device)

        fold_results.append({
            'fold': fold + 1,
            'best_val_acc': best_val_acc,
            'final_val_acc': final_val_acc,
            'val_predictions': val_preds,
            'val_true': val_true,
            'train_idx': train_idx,
            'val_idx': val_idx
        })

        all_histories.append({
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs
        })

        print(f"Fold {fold + 1} Best Val Accuracy: {best_val_acc:.4f}")

    return fold_results, all_histories



def analyze_cv_results(fold_results, all_histories, label_map):
    """Analyze cross-validation results"""
    
    print(f"\n{'='*60}")
    print(f"  CROSS-VALIDATION RESULTS SUMMARY")
    print(f"{'='*60}")
    
    # Extract accuracies
    best_accs = [result['best_val_acc'] for result in fold_results]
    final_accs = [result['final_val_acc'] for result in fold_results]
    
    # Summary statistics
    print(f"\nBest Validation Accuracies per Fold:")
    for i, acc in enumerate(best_accs):
        print(f"  Fold {i+1}: {acc:.4f} ({acc*100:.2f}%)")
    
    print(f"\nCross-Validation Summary:")
    print(f"  Mean Accuracy: {np.mean(best_accs):.4f} ± {np.std(best_accs):.4f}")
    print(f"  Best Fold: {np.max(best_accs):.4f}")
    print(f"  Worst Fold: {np.min(best_accs):.4f}")

    sys.stdout.flush()
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    # Training loss
    plt.subplot(1, 3, 1)
    for i, history in enumerate(all_histories):
        plt.plot(history['train_losses'], label=f'Fold {i+1}', alpha=0.7)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Training accuracy
    plt.subplot(1, 3, 2)
    for i, history in enumerate(all_histories):
        plt.plot(history['train_accs'], label=f'Fold {i+1}', alpha=0.7)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Validation accuracy
    plt.subplot(1, 3, 3)
    for i, history in enumerate(all_histories):
        plt.plot(history['val_accs'], label=f'Fold {i+1}', alpha=0.7)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Overall confusion matrix (combine all folds)
    all_predictions = []
    all_true_labels = []
    
    for result in fold_results:
        all_predictions.extend(result['val_predictions'])
        all_true_labels.extend(result['val_true'])
    
    # Use your existing analyze_results function
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    reverse_label_map = {v: k for k, v in label_map.items()}
    label_names = [reverse_label_map[i] for i in range(len(label_map))]
    
    cm = confusion_matrix(all_true_labels, all_predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names)
    plt.title('Cross-Validation - Combined Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    return np.mean(best_accs), np.std(best_accs)




##################################
# ---- Final Model Training ---- #
##################################



def train_final_model(
    all_data,
    all_labels,
    label_map,
    model_fn,
    save_path='final_model.pt',
    epochs=50,
    batch_size=4,
    lr=0.001,
    weight_decay=1e-5,
    seed=42
):
    """
    Train a model on the full dataset and save the trained parameters.

    Args:
        all_data: all training input data
        all_labels: all training labels
        label_map: dictionary mapping class labels to indices
        model_fn: factory function to create the model
        save_path: filename for saving model weights
        epochs: number of training epochs
        batch_size: training batch size
        seed: random seed
    Returns:
        trained model
    """
    set_seed(seed)

    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_fn().to(device)

    # Dataset and DataLoader
    train_dataset = MEGDataset(all_data, all_labels, label_map)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = total_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.4f}")

    # Save final model
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Final model trained and saved to: {save_path}")

    return model


#######################
# ---- Test Eval ---- #
#######################

def evaluate_on_test_set(model, test_loader, criterion, label_map, device=None):
    """
    Evaluate a trained model on the test set and show performance metrics.
    
    Args:
        model: trained PyTorch model
        test_loader: DataLoader for the test set
        criterion: loss function
        label_map: dict, maps label names to indices
        device: optional, torch.device
    Returns:
        test_loss, test_accuracy, predictions, true_labels
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_true = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_true.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total

    print(f"\n📊 Test Set Results:")
    print(f"  Loss     : {avg_loss:.4f}")
    print(f"  Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    reverse_label_map = {v: k for k, v in label_map.items()}
    label_names = [reverse_label_map[i] for i in range(len(label_map))]

    cm = confusion_matrix(all_true, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.title('Test Set Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

    return avg_loss, accuracy, all_preds, all_true
