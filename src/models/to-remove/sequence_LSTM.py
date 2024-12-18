import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class TweetDataset(Dataset):
    def __init__(self, tweets: List[str], labels: List[int], vocab: Dict, embeddings: np.ndarray, max_len: int = 50):
        self.tweets = tweets
        self.labels = labels
        self.vocab = vocab
        self.embeddings = embeddings
        self.max_len = max_len

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        tweet = self.tweets[idx].lower().split()
        # Get word indices and pad/truncate to max_len
        word_indices = [self.vocab.get(word, 0) for word in tweet[:self.max_len]]
        word_indices = word_indices + [0] * (self.max_len - len(word_indices))
        
        # Convert indices to embeddings
        sequence = torch.tensor([self.embeddings[i] for i in word_indices], dtype=torch.float)
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        
        return sequence, label

class LSTM(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Softmax(dim=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_weights = self.attention(lstm_out)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        return self.classifier(context)

def load_tweets(pos_file: str, neg_file: str) -> Tuple[List[str], List[int]]:
    tweets, labels = [], []
    
    # Load positive tweets
    with open(pos_file, 'r', errors='ignore') as f:
        pos_tweets = f.readlines()
        tweets.extend(pos_tweets)
        labels.extend([1] * len(pos_tweets))
    
    # Load negative tweets
    with open(neg_file, 'r', errors='ignore') as f:
        neg_tweets = f.readlines()
        tweets.extend(neg_tweets)
        labels.extend([-1] * len(neg_tweets))
    
    return tweets, labels

def create_data_loaders(tweets: List[str], labels: List[int], 
                       vocab: Dict, embeddings: np.ndarray,
                       batch_size: int = 32, train_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    # Split data
    train_tweets, val_tweets, train_labels, val_labels = train_test_split(tweets, labels, test_size=0.2)
    
    # Create datasets
    train_dataset = TweetDataset(train_tweets, train_labels, vocab, embeddings)
    val_dataset = TweetDataset(val_tweets, val_labels, vocab, embeddings)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader

def train_epoch(model: nn.Module, 
                train_loader: DataLoader, 
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> float:
    model.train()
    total_loss = 0
    
    for sequences, labels in tqdm(train_loader, desc='Training'):
        sequences, labels = sequences.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs.squeeze(), labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate(model: nn.Module, 
            val_loader: DataLoader,
            criterion: nn.Module,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Tuple[float, float]:
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sequences, labels in tqdm(val_loader, desc='Evaluating'):
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), labels)
            
            predictions = (outputs > 0).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()
    
    accuracy = correct / total
    avg_loss = total_loss / len(val_loader)
    return avg_loss, accuracy

def train_model(model: nn.Module, 
                train_loader: DataLoader,
                val_loader: DataLoader,
                n_epochs: int = 10,
                learning_rate: float = 0.001,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    best_accuracy = 0
    
    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        
        print(f'Epoch {epoch+1}/{n_epochs}:')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}\n')
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pt')
    
    return model

if __name__ == '__main__':
    # Load data and create model
    vocab = np.load('../../../data/vocab/vocab.pkl', allow_pickle=True)
    embeddings = np.load('../../../data/vocab/embeddings.npy')
    
    tweets, labels = load_tweets(
        '../../../data/twitter-datasets/train_pos.txt',
        '../../../data/twitter-datasets/train_neg.txt'
    )
    
    train_loader, val_loader = create_data_loaders(tweets, labels, vocab, embeddings)
    
    model = LSTM(
        embedding_dim=embeddings.shape[1],
        hidden_dim=128,
        num_layers=2
    )
    
    # Example usage with training loop
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    trained_model = train_model(model, train_loader, val_loader)


