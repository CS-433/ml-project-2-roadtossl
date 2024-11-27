from abc import ABC, abstractmethod
from model import Model
from joblib import dump, load


class SKLearnModel(Model):
    """
    Abstract class for models using scikit-learn
    """

    def __init__(self, model: object):
        self.model = model

    @abstractmethod
    def train(self, X, y):
        pass

    def predict(self, X):
        return self.model.predict(X)

    def save(self):
        print(f"Saving model to ../../data/models/{self.__class__.__name__}")
        path = f"../../data/models/{self.__class__.__name__}.joblib"
        dump(self.model, path)

    def load(self):
        path=f"../../data/models/{self.__class__.__name__}.joblib"
        return load(path)
        