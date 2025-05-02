from sklearn.metrics import accuracy_score, confusion_matrix
from abc import ABC, abstractmethod

class FaceModel(ABC):
    def __init__(self):
        if type(self) == FaceModel:
            raise Exception("FaceModel is an abstract class and cannot be instantiated")
        self.data_test = None
        self.data_train = None
        self.labels_prediction = None
        self.accuracy = None
        self.confusion_matrix = None
        
        self._distances = None
    
    @abstractmethod
    def train(self):
        pass    
    
    @abstractmethod
    def test(self):
        pass
    
    @abstractmethod
    def predict_euclidean(self):
        pass
        
    def analyze(self, labels_test):
        if type(labels_test) == None:
            raise TypeError()
        self.accuracy = accuracy_score(labels_test, self.labels_prediction)
        self.confusion_matrix = confusion_matrix(labels_test, self.labels_prediction)
    
    @abstractmethod
    def do_pipeline(self, data_train, data_test, labels_train, labels_test):
        pass
    
    def get_data(self):
        return (self.data_train, self.data_test)
    
    def print_accuracy(self, name):
        if type(name) == None:
            name = self.__class__.__name__
        print(f"Accuracy for {name}: {self.accuracy:.2f}")