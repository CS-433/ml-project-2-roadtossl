import tensorflow as tf
from tqdm import tqdm

DATASET_PATH = '../data/twitter-datasets/'
NEG_SAMPLES_PATH = DATASET_PATH + 'train_neg.txt'
POS_SAMPLES_PATH = DATASET_PATH + 'train_pos.txt'
FULL_NEG_SAMPLES_PATH = DATASET_PATH + 'train_neg_full.txt'
FULL_POS_SAMPLES_PATH = DATASET_PATH + 'train_pos_full.txt'
TEST_SAMPLES_PATH = DATASET_PATH + 'test_data.txt'

TRAIN_SPLIT = 0.8

def count_lines(filepath):
        with open(filepath, 'r', errors='ignore') as f:
            return sum(1 for _ in f)

def load_data(full=False):
    neg_samples_path = FULL_NEG_SAMPLES_PATH if full else NEG_SAMPLES_PATH
    pos_samples_path = FULL_POS_SAMPLES_PATH if full else POS_SAMPLES_PATH

    # Count lines using wc -l
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

    neg_train = neg_samples.take(neg_train_size)
    neg_test = neg_samples.skip(neg_train_size)

    pos_train = pos_samples.take(pos_train_size)
    pos_test = pos_samples.skip(pos_train_size)

    # Concatenate the training and test sets
    train_data = tf.data.Dataset.concatenate(neg_train, pos_train)
    test_data = tf.data.Dataset.concatenate(neg_test, pos_test)

    train_data_size = neg_train_size + pos_train_size
    test_data_size = neg_size + pos_size - train_data_size

    # Shuffle the training and test sets
    train_data = train_data.shuffle(buffer_size=train_data_size, reshuffle_each_iteration=False)
    test_data = test_data.shuffle(buffer_size=test_data_size, reshuffle_each_iteration=False)

    print(f'Loaded {train_data_size} training samples and {test_data_size} test samples')

    return train_data, test_data, train_data_size, test_data_size

def load_submission_data():
    """
    Load the test samples from the submission data file.

    Returns:
        tf.data.TextLineDataset: Dataset containing test samples.
    """
    test_samples = tf.data.TextLineDataset(TEST_SAMPLES_PATH)
    return test_samples