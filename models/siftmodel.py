from .facemodel import FaceModel
from skimage.feature import SIFT, match_descriptors
from collections import defaultdict

class SIFTModel(FaceModel):
    
    def __init__(self, min_matches_number=3):
        super().__init__("SIFT")
        self.SIFT = SIFT()
        self.data_train = list()
        self.data_test = list()
        self.labels_prediction = list()
        self.min_matches = min_matches_number
        
    def train(self, data_train, height, width):
        for img in data_train:
            img = img.reshape((height, width))
            self.SIFT.detect_and_extract(img)
            self.data_train.append(self.SIFT.descriptors)
    
    def test(self, data_test, height, width):
        for img in data_test:
            img = img.reshape((height, width))
            self.SIFT.detect_and_extract(img)
            self.data_test.append(self.SIFT.descriptors)
    
    def predict_euclidean(self, labels_train):
        votes = defaultdict(int)
        for item_test in self.data_test:
            for num_train, item_train in enumerate(self.data_train):
                result = match_descriptors(item_test, item_train, metric="euclidean", max_ratio=0.8)
                if len(result) > self.min_matches:
                    votes[labels_train[num_train]] += 1
            self.labels_prediction.append(max(votes, key=votes.get))
            votes.clear()
    
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
