# utilities.py
import h5py
import os

def get_dataset_name(filename_with_dir):
    filename_without_dir = filename_with_dir.split('/')[-1]
    temp = filename_without_dir.split('_')[:-1]
    dataset_name = "_".join(temp)
    return dataset_name

def get_filepath():
    # Get the directory where this Python file is located
    localpath = os.path.dirname(os.path.abspath(__file__))
    
    # Remove '/DL_2' or '\DL_2' from the path, depending on platform
    if os.name == 'nt':  # Windows
        modified_path = localpath.replace('\\DL_2', '')
    else:  # Unix/Mac
        modified_path = localpath.replace('/DL_2', '')
        
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
        print(type(matrix))
        print(matrix.shape)