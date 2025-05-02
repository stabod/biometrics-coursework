from .facemodel import FaceModel
from sklearn.decomposition import PCA
from numpy import argmin
from scipy.spatial.distance import cdist

class PCAModel(FaceModel):
    
    def __init__(self, components=150):
        super().__init__()
        self.PCA = PCA(n_components=components, svd_solver="randomized", whiten=True)
        
    def train(self, data_train):
        self.PCA.fit(data_train)
        self.data_train = self.PCA.transform(data_train)
        
    def test(self, data_test):
        self.data_test = self.PCA.transform(data_test)
        
    def predict_euclidean(self, labels_train):
        if type(labels_train) == None:
            raise TypeError()
        self._distances = cdist(self.data_test, self.data_train, metric="euclidean")
        self.labels_prediction = labels_train[argmin(self._distances, axis=1)]
        
    def do_pipeline(self, data_train, data_test, labels_train, labels_test):
        self.train(data_train)
        self.test(data_test)
        self.predict_euclidean(labels_train)
        self.analyze(labels_test)
        
    def print_accuracy(self, name="PCA"):
        return super().print_accuracy(name)
        
    
    