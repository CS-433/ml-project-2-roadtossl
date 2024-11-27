from abc import ABC, abstractmethod

class Model(ABC):
    """
    Abstract class for models
    """

    @abstractmethod
    def train(self, X, y):
        """
        Train the model
        
        Parameters:
        X: np.array
            Features
        y: np.array
            Target
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predict using the model
        
        Parameters:
        X: np.array
            Features
        """
        pass

    @abstractmethod
    def save(self):
        """
        Save the model
        """
        pass

    @abstractmethod
    def load(self):
        """
        Load the model
        """
        pass