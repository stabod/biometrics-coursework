from .facemodel import FaceModel
from skimage.feature import hog
from numpy import argmin
from scipy.spatial.distance import euclidean

class HOGModel(FaceModel):
    
    def __init__(self):
        super().__init__("HOG")
        self.data_train = list()
        self.data_test = list()
        self.labels_prediction = list()
        
    def train(self, data_train, height, width):
        for img in data_train:
            img = img.reshape((height, width))
            result = hog(img, feature_vector=True)
            self.data_train.append(result)
    
    def test(self, data_test, height, width):
        for img in data_test:
            img = img.reshape((height, width))
            result = hog(img, feature_vector=True)
            self.data_test.append(result)
            
    def predict_euclidean(self, labels_train):
        for test_item in self.data_test:
            distances = [euclidean(test_item, train_item) for train_item in self.data_train]
            nearest = argmin(distances)
            self.labels_prediction.append(labels_train[nearest])
            
    def do_pipeline(self, data_train, data_test, labels_train, labels_test, image_height, image_width):
        self.train(data_train, image_height, image_width)
        self.test(data_test, image_height, image_width)
        self.predict_euclidean(labels_train)
        self.analyze(labels_test)

    def clear(self):
        super().clear()
        self.data_train = list()
        self.data_test = list()
        self.labels_prediction = list()