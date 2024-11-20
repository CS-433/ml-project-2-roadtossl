#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle
import random
import torch
import torch.nn as nn
import torch.optim as optim


def main():

    print("loading cooccurrence matrix")
    with open("data/cooc.pkl", "rb") as f:
        cooc = pickle.load(f)
    print("{} nonzero entries".format(cooc.nnz))

    nmax = 100
    print("using nmax =", nmax, ", cooc.max() =", cooc.max())


    # Hyperparameters
    embedding_dim = 20
    alpha = 0.75
    learning_rate = 0.01
    epochs = 10

    # Create the vocabulary size from the co-occurrence data
    vocab_size = max(max(row[0], row[1]) for row in cooc) + 1

    # Weighting function for co-occurrence
    def weighting_function(n, nmax=nmax, alpha=alpha):
        return min(1.0, (n / nmax) ** alpha)

    # Neural network model
    class GloVeNN(nn.Module):
        def __init__(self, vocab_size, embedding_dim):
            super(GloVeNN, self).__init__()
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        def forward(self, word_ix, context_ix):
            word_embed = self.word_embeddings(word_ix)  # (batch_size, embedding_dim)
            context_embed = self.context_embeddings(context_ix)  # (batch_size, embedding_dim)
            dot_product = torch.sum(word_embed * context_embed, dim=1)  # (batch_size,)
            return dot_product

    # Initialize the model, optimizer, and loss function
    model = GloVeNN(vocab_size, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        for word_ix, context_ix, cooc in cooc:
            # Convert to tensors
            word_ix = torch.tensor([word_ix], dtype=torch.long)
            context_ix = torch.tensor([context_ix], dtype=torch.long)
            log_cooc = torch.tensor([torch.log(torch.tensor(cooc, dtype=torch.float32))], dtype=torch.float32)
            weight = torch.tensor([weighting_function(cooc)], dtype=torch.float32)

            # Forward pass
            pred = model(word_ix, context_ix)  # Predicted log(co-occurrence)

            # Loss computation
            loss = weight * (pred - log_cooc) ** 2
            epoch_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    # Save embeddings
    word_embeddings = model.word_embeddings.weight.detach().numpy()
    context_embeddings = model.context_embeddings.weight.detach().numpy()

    np.save("data/NN_embeddings", word_embeddings)

if __name__ == "__main__":
    main()
