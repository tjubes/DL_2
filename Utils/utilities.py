# utilities.py
import h5py
import os
from pathlib import Path
import numpy as np
from scipy.stats import zscore


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