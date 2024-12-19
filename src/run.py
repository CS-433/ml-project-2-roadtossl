import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.utils.submission import write_submission
from src.utils.dataloader import load_data_mean, load_data_seq
from src.utils.initialization import generate_all_required_files

# Models
from src.models.averaged_embeddings_models.GradientBoosting import GradientBoosting
from src.models.averaged_embeddings_models.LogisticRegression import LogisticRegression
from src.models.averaged_embeddings_models.SupportVectorMachine import SupportVectorMachine
from src.models.averaged_embeddings_models.NeuralNetwork import NeuralNetwork
from src.models.sequenced_embedding_models.RecurrentNeuralNetwork import RecurrentNeuralNetwork

# Specify the model to train
model_type = RecurrentNeuralNetwork # Set to one of the following: GradientBoosting, LogisticRegression, SupportVectorMachine, NeuralNetwork, RecurrentNeuralNetwork
full_dataset = True # Set to True to use the full dataset for training (may take a long time for the entire dataset)
force_generation = False # Set to True to force the generation of all required files each time the script is run

# Initialize model and submission variables
model = None
submission = None

# Generate all required files
generate_all_required_files(force_generation=force_generation)

# Check if a model type is specified
if model_type is None:
    print("Please specify a model type. To do so, set the 'model_type' variable to one of the following:")
    print("  - GradientBoosting (Averaged-Based Model)")
    print("  - LogisticRegression (Averaged-Based Model)")
    print("  - SupportVectorMachine (Averaged-Based Model)")
    print("  - NeuralNetwork (Averaged-Based Model)")
    print("  - RecurrentNeuralNetwork (Sequenced-Based Model)")
    exit()


# Sequenced-Based Model
if model_type is RecurrentNeuralNetwork:
    # Hyperparameters can be passed here. Default values are used if not specified.
    # Default Values: vocab_size=15000, word_embedding_dim=128, batch_size=64, epochs=10
    model = RecurrentNeuralNetwork()

    # Load the data
    train_data, test_data, train_data_size, test_data_size = load_data_seq(full=full_dataset)

    # Train the model and make predictions
    print(f"Training {model_type.__name__}...\n")
    submission = model.train(train_data, test_data, train_data_size, test_data_size)
    print("Training complete ✔️\n")


# Averaged-Based Model
else:
    # Load the data for the Averaged-Based Models (mean embeddings for each tweet)
    X, y = load_data_mean(full=full_dataset)

    # Initialize the GradientBoosting model
    if model_type is GradientBoosting:
        # Hyperparameters can be passed here. Default values are used if not specified.
        # Default Values: use_label_encoder=False, eval_metric='logloss'
        model = GradientBoosting()
    
    # Initialize the LogisticRegression model
    elif model_type is LogisticRegression:
        # Hyperparameters can be passed here. Default values are used if not specified.
        # Default Values: max_iter=100000
        model = LogisticRegression()
    
    # Initialize the SupportVectorMachine model
    elif model_type is SupportVectorMachine:
        # Hyperparameters can be passed here. Default values are used if not specified.
        # Default Values: kernel='rbf', random_state=42
        model = SupportVectorMachine()
    
    # Initialize the NeuralNetwork model
    elif model_type is NeuralNetwork:
        # Hyperparameters can be passed here. Default values are used if not specified.
        # Default Values: hidden_layer_sizes=(100,), max_iter=200, random_state=42
        model = NeuralNetwork()
    
    # Train the model and make predictions
    print(f"Training {model_type.__name__}...\n")
    submission = model.train(X, y)
    print("Training complete ✔️\n")


# Write the submission to a CSV file (This file can be submitted on aicrowd.com for evaluation)
model_name = model_type.__name__
dataset_type = "FullDataset" if full_dataset else "SmallDataset"
write_submission(submission, output_path=f"../data/submission/{model_name}_{dataset_type}.csv")

