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
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

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
    


def load_and_normalize_files_batch(directory_path, batch_size=8, downsample_factor=None, verbose=False):
    """
    Load and normalize h5 files in batches to manage memory for Cross dataset
    
    Parameters:
    directory_path (str): Path to directory containing h5 files
    batch_size (int): Number of files to load per batch
    downsample_factor (int, optional): Factor by which to downsample the data
    verbose (bool): Whether to print batch loading progress
    
    Yields:
    tuple: (batch_data, batch_labels) for each batch
    """
    h5_files = [f for f in os.listdir(directory_path) if f.endswith('.h5')]
    # if verbose:
    #     print(f"Found {len(h5_files)} files in {directory_path}")
    
    # Process files in batches
    for i in range(0, len(h5_files), batch_size):
        batch_files = h5_files[i:i + batch_size]
        batch_data = []
        batch_labels = []
        
        # if verbose:
        #     print(f"Loading batch {i//batch_size + 1}/{(len(h5_files) + batch_size - 1)//batch_size} "
        #           f"(files {i+1}-{min(i+batch_size, len(h5_files))})")
        
        for file in batch_files:
            # Extract task type from filename
            if file.startswith("task_"):
                parts = file.split('_')
                task_type = '_'.join(parts[:2])
            else:
                task_type = file.split('_')[0]
            
            # Load and process the data
            file_path = os.path.join(directory_path, file)
            matrix = read_h5py_file(file_path)
            
            # Downsample if specified
            if downsample_factor is not None:
                matrix = matrix[:, ::downsample_factor]
            
            # Normalize using Z-score
            normalized_matrix = zscore(matrix, axis=1, nan_policy='propagate')
            normalized_matrix = np.nan_to_num(normalized_matrix, nan=0.0)
            
            batch_data.append(normalized_matrix)
            batch_labels.append(task_type)
        
        yield batch_data, batch_labels


from sklearn.model_selection import train_test_split

def train_one_epoch_batched(model, 
                            train_path, 
                            label_map, 
                            criterion, 
                            optimizer, 
                            batch_size=8, 
                            downsample_factor=16, 
                            dataloader_batch_size=4, 
                            device='cpu', 
                            verbose=False):
    """
    Train for one epoch using batched file loading with 80/20 validation split
    """
    model.train()
    total_train_loss = 0.0
    total_train_correct = 0
    total_train_samples = 0

    total_val_loss = 0.0
    total_val_correct = 0
    total_val_samples = 0

    batch_count = 0

    for batch_data, batch_labels in load_and_normalize_files_batch(
        train_path, batch_size=batch_size, downsample_factor=downsample_factor, verbose=verbose):


        # === Train ===
        train_dataset = MEGDataset(batch_data, batch_labels, label_map)
        train_loader = DataLoader(train_dataset, batch_size=dataloader_batch_size, shuffle=True)

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train_samples += labels.size(0)
            total_train_correct += (predicted == labels).sum().item()

        # Clear memory
        del train_dataset, train_loader, batch_data, batch_labels

    # === Averages ===
    avg_train_loss = total_train_loss / total_train_samples if total_train_samples > 0 else 0
    train_accuracy = total_train_correct / total_train_samples if total_train_samples > 0 else 0    

    return avg_train_loss, train_accuracy

def evaluate_model_batched(model, 
                           label_map,
                           test_path, 
                           criterion, 
                           batch_size=8, 
                           downsample_factor=16, 
                           dataloader_batch_size=4, 
                           device='cpu', 
                           verbose=False):
    """
    Evaluate model using batched loading
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    batch_count = 0
    
    with torch.no_grad():
        for batch_data, batch_labels in load_and_normalize_files_batch(
            test_path, batch_size=batch_size, downsample_factor=downsample_factor, verbose=verbose):
            
            temp_dataset = MEGDataset(batch_data, batch_labels, label_map)
            temp_dataloader = DataLoader(temp_dataset, batch_size=dataloader_batch_size, 
                                       shuffle=False, num_workers=0)
            
            for inputs, labels in temp_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            batch_count += 1
            del temp_dataset, temp_dataloader, batch_data, batch_labels
    
    avg_loss = total_loss / (batch_count * dataloader_batch_size) if batch_count > 0 else 0
    accuracy = len([1 for i, j in zip(all_predictions, all_labels) if i == j]) / len(all_labels)
    
    return avg_loss, accuracy, all_predictions, all_labels



class EarlyStopper:
    def __init__(self, patience=12, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None

    def check(self, score):
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def train_cross_subject_model(filepath, 
                              model_fn,
                              LABEL_MAP,
                              epochs=50, 
                              device='cpu',
                              file_batch_size=8, 
                              dataloader_batch_size=4, 
                              downsample_factor=4, 
                              lr=0.001, 
                              weight_decay=1e-4):
    """
    Train cross-subject model with validation tracking
    """

    cross_train_path = os.path.join(filepath, "Cross", "train")

    print(f"\n{'='*60}")
    print(f"  CROSS-SUBJECT TRAINING")
    print(f"{'='*60}")
    print(f"Training data path: {cross_train_path}")
    print(f"File batch size: {file_batch_size}")
    print(f"DataLoader batch size: {dataloader_batch_size}")
    print(f"Downsample factor: {downsample_factor}")
    print(f"Learning rate: {lr}")
    print(f"Weight decay: {weight_decay}")

    # Sample input shape
    sample_batch_data, _ = next(load_and_normalize_files_batch(
        cross_train_path, batch_size=1, downsample_factor=downsample_factor))
    input_time_steps = sample_batch_data[0].shape[1]
    print(f"Input shape: (248, {input_time_steps})")

    # Model setup
    model = model_fn().to(device)
    model_save_path = 'best_cross_2dcnn_lstm_model.pth'

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses, train_accs = [], []


    early_stopper = EarlyStopper(patience=20, min_delta=0.001)

    print(f"\nStarting training for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch_batched(
            model, cross_train_path, LABEL_MAP, criterion, optimizer,
            batch_size=file_batch_size, downsample_factor=downsample_factor,
            dataloader_batch_size=dataloader_batch_size, device=device,
            verbose=(epoch % 5 == 0)
        )

        train_losses.append(train_loss)
        train_accs.append(train_acc)
         

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}: Train Acc={train_acc:.4f},"
                  f"Train Loss={train_loss:.4f}")
    
    torch.save(model.state_dict(), model_save_path)
    model.load_state_dict(torch.load(model_save_path, map_location=device))

    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\n🌟 TRAINING COMPLETED")
    print(f"Model saved as '{model_save_path}'")

    return model, train_losses, train_accs




def evaluate_cross_subject_model(filepath, 
                                 model_fn,
                                 label_map,
                                 model_path='best_cnn_lstm_model.pth', 
                                 downsample_factor=4, 
                                 file_batch_size=8, 
                                 dataloader_batch_size=4,
                                 device='cpu'):
    """
    Evaluate the cross-subject model on all three test sets (test1, test2, test3)
    """
    print(f"\n{'='*60}")
    print(f"  CROSS-SUBJECT MODEL EVALUATION")
    print(f"{'='*60}")

    # Get sample data to determine correct input dimensions
    cross_train_path = os.path.join(filepath, "Cross", "train")
    sample_batch_data, _ = next(load_and_normalize_files_batch(
        cross_train_path, batch_size=1, downsample_factor=downsample_factor, verbose=False))
    input_time_steps = sample_batch_data[0].shape[1]

    # Load the trained model with correct dimensions
    print(f"Loading model from {model_path}")
    model = model_fn()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device) 
    model.eval()

    criterion = nn.CrossEntropyLoss()

    # Test on all three test sets
    test_sets = ['test1', 'test2', 'test3']
    all_results = {}

    reverse_label_map = {v: k for k, v in label_map.items()}
    label_names = [reverse_label_map[i] for i in range(len(label_map))]

    for test_set in test_sets:
        test_path = os.path.join(filepath, "Cross", test_set)
        print(f"\n--- Evaluating on {test_set} ---")

        if not os.path.exists(test_path):
            print(f"Warning: {test_path} does not exist, skipping...")
            continue

        # Get number of files
        # h5_files = [f for f in os.listdir(test_path) if f.endswith('.h5')]
        # print(f"Found {len(h5_files)} files in {test_set}")

        # Evaluate
        test_loss, test_acc, predictions, true_labels = evaluate_model_batched(
            model, label_map, test_path, criterion, 
            batch_size=file_batch_size, downsample_factor=downsample_factor,
            dataloader_batch_size=dataloader_batch_size, device=device, verbose=False
        )

        all_results[test_set] = {
            'loss': test_loss,
            'accuracy': test_acc,
            'predictions': predictions,
            'true_labels': true_labels,
            'num_samples': len(predictions)
        }

        print(f"{test_set} - Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%), "
              f"Loss: {test_loss:.4f}, Samples: {len(predictions)}")

    # Overall statistics
    print(f"\n--- OVERALL RESULTS ---")
    all_accuracies = [results['accuracy'] for results in all_results.values()]
    all_samples = [results['num_samples'] for results in all_results.values()]

    if all_accuracies:
        weighted_avg_acc = sum(acc * samples for acc, samples in zip(all_accuracies, all_samples)) / sum(all_samples)
        print(f"Individual test accuracies: {[f'{acc:.3f}' for acc in all_accuracies]}")
        print(f"Mean accuracy: {np.mean(all_accuracies):.4f} ± {np.std(all_accuracies):.4f}")
        print(f"Weighted average accuracy: {weighted_avg_acc:.4f}")
        print(f"Total test samples: {sum(all_samples)}")

    # Create visualizations
    if all_results:
        plot_cross_subject_results(all_results, label_names)

    return all_results



def plot_cross_subject_results(all_results, label_names):
    """
    Plot confusion matrices and accuracy comparison for cross-subject results
    """
    n_tests = len(all_results)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, max(2, n_tests), figsize=(5*n_tests, 10))
    if n_tests == 1:
        axes = axes.reshape(2, 1)
    
    # Plot confusion matrices for each test set
    for i, (test_name, results) in enumerate(all_results.items()):
        if i < n_tests:  # Only plot if we have space
            ax = axes[0, i] if n_tests > 1 else axes[0]
            
            cm = confusion_matrix(results['true_labels'], results['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=label_names, yticklabels=label_names, ax=ax)
            ax.set_title(f'{test_name}\nAccuracy: {results["accuracy"]:.3f}')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
    
    # Hide extra subplots in first row
    for i in range(n_tests, axes.shape[1]):
        axes[0, i].set_visible(False)
    
    # Plot accuracy comparison
    ax_acc = axes[1, 0] if n_tests > 1 else axes[1]
    test_names = list(all_results.keys())
    accuracies = [all_results[test]['accuracy'] for test in test_names]
    
    bars = ax_acc.bar(test_names, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'][:len(test_names)])
    ax_acc.set_title('Cross-Subject Test Accuracies')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_ylim(0, 1)
    
    # Add accuracy labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax_acc.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom')
    
    # Add horizontal line at 0.25 (random chance for 4 classes)
    ax_acc.axhline(y=0.25, color='red', linestyle='--', alpha=0.7, label='Random chance (25%)')
    ax_acc.legend()
    
    # Hide extra subplots in second row
    for i in range(1, axes.shape[1]):
        axes[1, i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed classification reports
    print(f"\n--- DETAILED CLASSIFICATION REPORTS ---")
    for test_name, results in all_results.items():
        print(f"\n{test_name.upper()}:")
        print(classification_report(results['true_labels'], results['predictions'], 
                                  target_names=label_names, digits=3))
        

def analyze_overfitting(train_losses, train_accs, all_results):
    """
    Analyze potential overfitting by comparing train/val/test performance
    """
    print(f"\n{'='*60}")
    print(f"  OVERFITTING ANALYSIS")
    print(f"{'='*60}")

    final_train_acc = train_accs[-1] if train_accs else 0
    test_accuracies = [results['accuracy'] for results in all_results.values()]

    if test_accuracies:
        avg_test_acc = np.mean(test_accuracies)
        train_gap = final_train_acc - avg_test_acc


        print(f"Final training accuracy:    {final_train_acc:.4f}")
        print(f"Average test accuracy:      {avg_test_acc:.4f}")
        print(f"Train-test gap:             {train_gap:.4f}")

        if train_gap > 0.3:
            print("\n🚨 SEVERE OVERFITTING DETECTED")
        elif train_gap > 0.15:
            print("\n⚠️  MODERATE OVERFITTING DETECTED")
        else:
            print("\n✅ GOOD GENERALIZATION")

    # Plot
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_accs, label='Train Acc')
    if test_accuracies:
        plt.axhline(np.mean(test_accuracies), linestyle='--', color='red', label='Test Avg')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Add evaluation to the main function
def run_complete_cross_subject_experiment(filepath, model_fn, label_map,
                                           epochs=50, file_batch_size=8,
                                           dataloader_batch_size=4,
                                           downsample_factor=4,
                                           lr=0.001, weight_decay=1e-4,
                                           device='cpu'):
    """
    Run complete cross-subject experiment: train + evaluate + analyze
    """

    # Train the model
    model, train_losses, train_accs = train_cross_subject_model(
        filepath=filepath,
        model_fn=model_fn,
        LABEL_MAP=label_map,
        epochs=epochs,
        device=device,
        file_batch_size=file_batch_size,
        dataloader_batch_size=dataloader_batch_size,
        downsample_factor=downsample_factor,
        lr=lr,
        weight_decay=weight_decay
    )

    # Evaluate on test sets
    model_path = 'best_cross_2dcnn_lstm_model.pth'
    test_results = evaluate_cross_subject_model(
        filepath=filepath,
        model_fn=model_fn,
        label_map=label_map,
        model_path=model_path,
        downsample_factor=downsample_factor,
        file_batch_size=file_batch_size,
        dataloader_batch_size=dataloader_batch_size,
        device=device
    )

    # Analyze overfitting
    analyze_overfitting(train_losses, train_accs, test_results)

    return model, train_losses, train_accs, test_results

