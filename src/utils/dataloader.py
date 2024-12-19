import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import tensorflow as tf

# Define the dataset path relative to the current file location
CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
DATASET_PATH = os.path.join(CURRENT_PATH, '../../data/twitter-datasets/')
NEG_SAMPLES_PATH = os.path.join(DATASET_PATH, 'train_neg.txt')
POS_SAMPLES_PATH = os.path.join(DATASET_PATH, 'train_pos.txt')
FULL_NEG_SAMPLES_PATH = os.path.join(DATASET_PATH, 'train_neg_full.txt')
FULL_POS_SAMPLES_PATH = os.path.join(DATASET_PATH, 'train_pos_full.txt')

NEG_MEAN_SAMPLES_PATH = os.path.join(DATASET_PATH, 'train_neg_embedding.txt')
POS_MEAN_SAMPLES_PATH = os.path.join(DATASET_PATH, 'train_pos_embedding.txt')

FULL_NEG_MEAN_SAMPLES_PATH = os.path.join(DATASET_PATH, 'train_neg_full_embedding.txt')
FULL_POS_MEAN_SAMPLES_PATH = os.path.join(DATASET_PATH, 'train_pos_full_embedding.txt')

TEST_SAMPLES_PATH = os.path.join(DATASET_PATH, 'test_data.txt')

# Define the split ratio for training and testing data
TRAIN_SPLIT = 0.8

def count_lines(filepath):
    """
    Count the number of lines in a file.

    Parameters:
        filepath (str): Path to the file.

    Returns:
        int: Number of lines in the file.
    """
    with open(filepath, 'r', errors='ignore') as f:
        return sum(1 for _ in f)

def load_data_seq(full=False):
    """
    Load the training and test samples from the dataset files.
    
    Parameters:
        full (bool): Whether to load the full dataset.

    Returns:
        tf.data.Dataset: Dataset containing training samples.
        tf.data.Dataset: Dataset containing test samples.
        int: Number of training samples.
        int: Number of test samples.
    """
    # Load the samples
    print("Loading Data...", end="\r")
    neg_samples_path = FULL_NEG_SAMPLES_PATH if full else NEG_SAMPLES_PATH
    pos_samples_path = FULL_POS_SAMPLES_PATH if full else POS_SAMPLES_PATH

    # Count the number of positive and negative samples
    neg_size = count_lines(neg_samples_path)
    pos_size = count_lines(pos_samples_path)

    # Load the samples and map them to a tuple with the label
    neg_samples = tf.data.TextLineDataset(neg_samples_path).map(lambda x: (x, 0))
    pos_samples = tf.data.TextLineDataset(pos_samples_path).map(lambda x: (x, 1))

    # Shuffle the samples
    neg_samples = neg_samples.shuffle(buffer_size=neg_size, reshuffle_each_iteration=False)
    pos_samples = pos_samples.shuffle(buffer_size=pos_size, reshuffle_each_iteration=False)

    # Split the samples into training and test sets
    neg_train_size = int(TRAIN_SPLIT * neg_size)
    pos_train_size = int(TRAIN_SPLIT * pos_size)

    # Take the first n negative samples for training and the rest for testing
    neg_train = neg_samples.take(neg_train_size)
    neg_test = neg_samples.skip(neg_train_size)

    # Take the first n positive samples for training and the rest for testing
    pos_train = pos_samples.take(pos_train_size)
    pos_test = pos_samples.skip(pos_train_size)

    # Concatenate the negative and positive samples for training and testing
    train_data = tf.data.Dataset.concatenate(neg_train, pos_train)
    test_data = tf.data.Dataset.concatenate(neg_test, pos_test)

    # Calculate the number of training and test samples
    train_data_size = neg_train_size + pos_train_size
    test_data_size = neg_size + pos_size - train_data_size

    # Shuffle the training and test sets
    train_data = train_data.shuffle(buffer_size=train_data_size, reshuffle_each_iteration=False)
    test_data = test_data.shuffle(buffer_size=test_data_size, reshuffle_each_iteration=False)

    print(f'Loaded {train_data_size} training samples and {test_data_size} test samples ✔️\n')

    return train_data, test_data, train_data_size, test_data_size

# Assuming train_neg_padded and train_pos_padded are already defined
filename_pos = 'data/twitter-datasets/train_pos_full_embedding.txt'
filename_neg = 'data/twitter-datasets/train_neg_full_embedding.txt'

# Function to load embeddings
def load_embeddings(filename):
    """
    Load the embeddings from a file.
    
    Parameters:
        filename (str): Path to the file containing the embeddings.
        
    Returns:
        list: List of embeddings.
    """

    # Load the embeddings
    embeddings = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                str_values = line.split()
                float_values = [float(val) for val in str_values]
                embedding = np.array(float_values)
                embeddings.append(embedding)

    return embeddings

def load_data_mean(full=False):
    """
    Load the training samples from the mean embeddings files.

    Parameters:
        full (bool): Whether to load the full dataset.
        
    Returns:
        np.ndarray: Padded training samples.
        np.ndarray: Training labels.
    """
    # Load the samples
    print("Loading Data...", end="\r")
    filename_pos = FULL_POS_MEAN_SAMPLES_PATH if full else POS_MEAN_SAMPLES_PATH
    filename_neg = FULL_NEG_MEAN_SAMPLES_PATH if full else NEG_MEAN_SAMPLES_PATH

    # Load the embeddings
    pos_embeddings = load_embeddings(filename_pos)
    neg_embeddings = load_embeddings(filename_neg)

    # Pad the embeddings
    train_neg_padded = np.array(neg_embeddings)
    train_pos_padded = np.array(pos_embeddings)

    # Create labels
    neg_labels = np.zeros(train_neg_padded.shape[0], dtype=int)
    pos_labels = np.ones(train_pos_padded.shape[0], dtype=int)

    # Combine data and labels
    train_padded = np.concatenate((train_neg_padded, train_pos_padded), axis=0)
    train_labels = np.concatenate((neg_labels, pos_labels), axis=0)

    # Shuffle data
    indices = np.arange(train_padded.shape[0])
    np.random.shuffle(indices)
    train_padded = train_padded[indices]
    train_labels = train_labels[indices]

    print("Data Loaded ✔️  \n")

    return train_padded, train_labels

def load_submission_data():
    """
    Load the test samples from the submission data file.

    Returns:
        tf.data.TextLineDataset: Dataset containing test samples.
    """
    test_samples = tf.data.TextLineDataset(TEST_SAMPLES_PATH)
    return test_samples
