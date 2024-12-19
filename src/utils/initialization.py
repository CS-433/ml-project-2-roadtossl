import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(root_dir)

import subprocess

from src.datasets import pickle_vocab
from src.datasets import cooc
from src.datasets import glove_solution
from src.datasets import tweet_to_vector

def create_init_folder():
    """
    Create the data/init/ folder if it does not exist
    """
    vocab_dir = os.path.join(root_dir, 'data/init')
    if not os.path.exists(vocab_dir):
        os.makedirs(vocab_dir)

def create_submission_folder():
    """
    Create the data/submission/ folder if it does not exist
    """
    submission_dir = os.path.join(root_dir, 'data/submission')
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir)

def unzip_dataset():
    """
    Unzip the twitter dataset
    """
    zip_file_path = os.path.join(root_dir, 'data/twitter-datasets.zip')
    dest_dir = os.path.join(root_dir, 'data/')
    subprocess.run(['unzip', zip_file_path, '-d', dest_dir])

def build_vocab():
    """
    Build the vocab file
    """
    vocab_file_path = os.path.join(root_dir, 'src/datasets/build_vocab.sh')
    subprocess.run(['bash', vocab_file_path])

def cut_vocab():
    """
    Cut the vocab file by runnnig the cut_vocab.sh script
    """
    vocab_file_path = os.path.join(root_dir, 'src/datasets/cut_vocab.sh')
    subprocess.run(['bash', vocab_file_path])

def generate_all_required_files(force_generation=False):
    """
    Generate all required files for the project

    Parameters:
    - force_generation: bool
        Whether to force the generation of all files even if they already exist
    """

    print("Starting initialization...\n")

    if force_generation:
        print("Forcing generation of all files...\n")

    # Create the data/init/ folder if it does not exist
    if not os.path.exists(os.path.join(root_dir, 'data/init')) or force_generation:
        print("1. Creating data/init/ folder...\n")
        create_init_folder()
        print("data/init/ folder created ✔️\n")
    else:
        print("1. data/init/ folder already created ✔️\n")

    # Create the data/submission/ folder if it does not exist
    if not os.path.exists(os.path.join(root_dir, 'data/submission')) or force_generation:
        print("2. Creating data/submission/ folder...\n")
        create_submission_folder()
        print("data/submission/ folder created ✔️\n")
    else:
        print("2. data/submission/ folder already created ✔️\n")

    # Unzip the dataset
    if not os.path.exists(os.path.join(root_dir, 'data/twitter-datasets')) or force_generation:
        print("3. Unzipping dataset...\n")
        unzip_dataset()
        print("Dataset unzipped ✔️\n")
    else:
        print("3. Dataset already unzipped ✔️\n")

    # Build the vocab file
    if not os.path.exists(os.path.join(root_dir, 'data/init/vocab_full.txt')) or force_generation:
        print("4. Building vocab...\n")
        build_vocab()
        print("Vocab built ✔️\n")
    else:
        print("4. Vocab already built ✔️\n")

    # Cut the vocab file
    if not os.path.exists(os.path.join(root_dir, 'data/init/vocab_cut.txt')) or force_generation:
        print("5. Cutting vocab...\n")
        cut_vocab()
        print("Vocab cut ✔️\n")
    else:
        print("5. Vocab already cut ✔️\n")

    # Pickle the vocab
    if not os.path.exists(os.path.join(root_dir, 'data/init/vocab.pkl')) or force_generation:
        print("6. Pickling vocab...\n")
        pickle_vocab.run()
        print("Vocab pickled ✔️\n")
    else:
        print("6. Vocab already pickled ✔️\n")

    # Create the cooc matrix
    if not os.path.exists(os.path.join(root_dir, 'data/init/cooc.pkl')) or force_generation:
        print("7. Creating cooc matrix...\n")
        cooc.run()
        print("Cooc matrix created ✔️\n")
    else:
        print("7. Cooc matrix already created ✔️\n")

    # Train the vectors with GloVe
    if not os.path.exists(os.path.join(root_dir, 'data/init/SGD_embeddings.npy')) or force_generation:
        print("8. Training vectors with GloVe...\n")
        glove_solution.run()
        print("Glove solution created ✔️\n")
    else:
        print("8. Vectors already trained ✔️\n")

    # Create the tweet embeddings vectors
    cond1 = not os.path.exists(os.path.join(root_dir, 'data/twitter-datasets/train_neg_embedding.txt'))
    cond2 = not os.path.exists(os.path.join(root_dir, 'data/twitter-datasets/train_pos_embedding.txt'))
    cond3 = not os.path.exists(os.path.join(root_dir, 'data/twitter-datasets/train_neg_full_embedding.txt'))
    cond4 = not os.path.exists(os.path.join(root_dir, 'data/twitter-datasets/train_pos_full_embedding.txt'))
    if cond1 or cond2 or cond3 or cond4 or force_generation:
        print("9. Creating tweet vectors...\n")
        tweet_to_vector.run()
        print("Tweet vectors created ✔️\n")
    else:
        print("9. Tweet vectors already created ✔️\n")

    print("Initialization complete ✔️\n")



