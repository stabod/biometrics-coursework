from facemodel import FaceModel
from skimage.feature import hog
from numpy import argmin
from scipy.spatial.distance import cdist

class HOGModel(FaceModel):
    
    def __init__(self):
        super().__init__()
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
        self._distances = cdist(self.data_test, self.data_train, metric="euclidean")
        self.labels_prediction = labels_train[argmin(self._distances, axis=1)]
            
    def do_pipeline(self, data_train, data_test, labels_train, labels_test, image_height, image_width):
        self.train(data_train, image_height, image_width)
        self.test(data_test, image_height, image_width)
        self.predict_euclidean(labels_train)
        self.analyze(labels_test)
        
    def print_accuracy(self, name="HOG"):
        return super().print_accuracy(name)