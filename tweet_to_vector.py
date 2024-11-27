import numpy as np

def tweet_to_vector(tweet: str, vocab, embeddings) -> np.ndarray:
    """
    Convert a tweet to its vector representation by averaging word vectors.
    
    Args:
        tweet (str): Input tweet text
    
    Returns:
        np.ndarray: Mean vector of all word vectors in tweet
    """
    # Load embeddings
    
    # Preprocess tweet: lowercase and split into words
    words = tweet.lower().split()
    
    # Get vectors for words that exist in embeddings
    word_indices = [vocab[word] for word in words if word in vocab]
    vectors = [embeddings[word_index] for word_index in word_indices]

    # Return zero vector if no words found
    if not vectors:
         return np.zeros(embeddings.shape[1])
    
    # Calculate and return mean vector
    return np.mean(vectors, axis=0)

def write_tweet_vectors(input_file: str, output_file: str) -> None:
    if input_file == output_file:
        raise ValueError('Input and output files must be different')
    
    print(f"Creating vectors from strings from '{input_file}' into '{output_file}'...")

    vocab = np.load('data/vocab.pkl', allow_pickle=True)
    embeddings = np.load('data/embeddings.npy')

    nb_lines = sum(1 for _ in open(input_file, 'r', errors='ignore'))

    with open(input_file, 'r', errors='ignore') as file:
        with open(output_file, 'w') as out_file:
            for i, line in enumerate(file):
                if (((i + 1) % 100) == 0):
                    print(f'Processing line {i+1} / {nb_lines}', end='\r')
                vector = tweet_to_vector(line, vocab, embeddings)
                out_file.write(' '.join(map(str, vector)) + '\n')

    print(f"\nDone!\n")

if __name__ == '__main__':
    files_to_process = [
        './data/twitter-datasets/train_pos.txt', 
        './data/twitter-datasets/train_neg.txt', 
        './data/twitter-datasets/train_pos_full.txt', 
        './data/twitter-datasets/train_neg_full.txt'
    ]

    for file_to_process in files_to_process:
        output_file = file_to_process.replace('.txt', '_embedding.txt')
        write_tweet_vectors(file_to_process, output_file)
