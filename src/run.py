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

model = None
submission = None

generate_all_required_files(force_generation=force_generation)

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
    # -> vocab_size=15000, word_embedding_dim=128, batch_size=64, epochs=10
    model = RecurrentNeuralNetwork()
    train_data, test_data, train_data_size, test_data_size = load_data_seq(full=full_dataset)
    print(f"Training {model_type.__name__}...\n")
    submission = model.train(train_data, test_data, train_data_size, test_data_size)
    print("Training complete ✔️\n")


# Averaged-Based Model
else:
    X, y = load_data_mean(full=full_dataset)

    if model_type is GradientBoosting:
        # Hyperparameters can be passed here. Default values are used if not specified.
        # -> use_label_encoder=False, eval_metric='logloss'
        model = GradientBoosting()
    
    elif model_type is LogisticRegression:
        # Hyperparameters can be passed here. Default values are used if not specified.
        # -> max_iter=100000
        model = LogisticRegression()
    
    elif model_type is SupportVectorMachine:
        # Hyperparameters can be passed here. Default values are used if not specified.
        # -> kernel='rbf', random_state=42
        model = SupportVectorMachine()
    
    elif model_type is NeuralNetwork:
        # Hyperparameters can be passed here. Default values are used if not specified.
        # -> hidden_layer_sizes=(100,), max_iter=200, random_state=42
        model = NeuralNetwork()
    
    print(f"Training {model_type.__name__}...\n")
    submission = model.train(X, y)
    print("Training complete ✔️\n")


# Write the submission to a CSV file (This file can be submitted on aicrowd.com for evaluation)
model_name = model_type.__name__
dataset_type = "FullDataset" if full_dataset else "SmallDataset"
write_submission(submission, output_path=f"../data/submission/{model_name}_{dataset_type}.csv")

