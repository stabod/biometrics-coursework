from .facemodel import FaceModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from numpy import argmin
from scipy.spatial.distance import euclidean
from numpy import argmin

class PCAModel(FaceModel):
    
    def __init__(self, components=150):
        super().__init__("PCA")
        self.PCA = PCA(n_components=components, svd_solver="randomized", whiten=True)
        self.scaler = StandardScaler()
        self.labels_prediction = list()
        
    def train(self, data_train):
        scaled = self.scaler.fit_transform(data_train)
        self.data_train = self.PCA.fit_transform(scaled)
        
    def test(self, data_test):
        scaled = self.scaler.transform(data_test)
        self.data_test = self.PCA.transform(scaled)
        
    def predict_euclidean(self, labels_train):
        for test_item in self.data_test:
            distances = [euclidean(test_item, train_item) for train_item in self.data_train]
            nearest = argmin(distances)
            self.labels_prediction.append(labels_train[nearest])
        
    def do_pipeline(self, data_train, data_test, labels_train, labels_test, image_height, image_width):
        self.train(data_train)
        self.test(data_test)
        self.predict_euclidean(labels_train)
        self.analyze(labels_test)
        
    def clear(self):
        super().clear()
        self.labels_prediction = list()
