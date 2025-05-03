from .facemodel import FaceModel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from numpy import argmin
from scipy.spatial.distance import euclidean

class LDAModel(FaceModel):
    def __init__(self):
        super().__init__("LDA")
        self.LDA = LDA()
        self.labels_prediction = list()
        
    def train(self, data_train_PCA, labels_train):
        self.data_train = self.LDA.fit_transform(data_train_PCA, labels_train)
    
    def test(self, data_test_PCA):
        self.data_test = self.LDA.transform(data_test_PCA)
        
    def predict_euclidean(self, labels_train):
        for test_item in self.data_test:
            distances = [euclidean(test_item, train_item) for train_item in self.data_train]
            nearest = argmin(distances)
            self.labels_prediction.append(labels_train[nearest])
        
    def do_pipeline(self, data_train, data_test, labels_train, labels_test, image_height, image_width):
        self.train(data_train, labels_train)
        self.test(data_test)
        self.predict_euclidean(labels_train)
        self.analyze(labels_test)
        
    def clear(self):
        super().clear()
        self.labels_prediction = list()
        self.templates = list()
        self.template_labels = list()
        

   