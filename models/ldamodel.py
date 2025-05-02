from .facemodel import FaceModel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from numpy import mean, array, argmin
from scipy.spatial.distance import cdist

class LDAModel(FaceModel):
    def __init__(self, classes):
        if type(classes) == None:
            raise TypeError("No classes provided")
        super().__init__()
        self.LDA = LDA(n_components=(len(classes) - 1))
        self.classes = classes
        self.templates = []
        self.template_labels = []
        
    def train(self, data_train_PCA, labels_train):
        self.data_train = self.LDA.fit_transform(data_train_PCA, labels_train)
        
        for label in self.classes:
            class_samples = self.data_train[labels_train == label]
            class_mean = mean(class_samples, axis=0)
            self.templates.append(class_mean)
            self.template_labels.append(label)
            
        self.templates = array(self.templates)
        self.template_labels = array(self.template_labels)
        
    
    def test(self, data_test_PCA):
        self.data_test = self.LDA.transform(data_test_PCA)
        
    def predict_euclidean(self):
        self._distances = cdist(self.data_test, self.templates, metric='euclidean')
        self.labels_prediction = self.template_labels[argmin(self._distances, axis=1)]
        
    def do_pipeline(self, data_train, data_test, labels_train, labels_test):
        self.train(data_train, labels_train)
        self.test(data_test)
        self.predict_euclidean()
        self.analyze(labels_test)
        
    def print_accuracy(self, name="LDA"):
        return super().print_accuracy(name)
        
        