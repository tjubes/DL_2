# utilities.py
import h5py
import os
from pathlib import Path
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
import torch
import sys
from torch.utils.data import Dataset, DataLoader, Subset


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