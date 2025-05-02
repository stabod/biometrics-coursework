from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from numpy import unique

class FaceHolder:
    def __init__(self):
        self.n_samples = None
        self.image_height = None
        self.image_width = None
        
        self.people_data = None
        self.people_labels = None
        self.target_names = None
        self.classes = None
    
        self.data_train = None
        self.data_test = None
        self.labels_train = None
        self.labels_test = None
        
    def get_lfw(self, min_faces=60):
        people = fetch_lfw_people(min_faces_per_person=min_faces, resize=0.5)
        self.n_samples, self.image_height, self.image_width = people.images.shape
        self.people_data = people.data
        self.people_labels = people.target
        self.target_names = people.target_names
        self.classes = unique(self.people_labels)
        
    def split_data(self, test_size=0.25, random=None):
        self.data_train, self.data_test, self.labels_train, self.labels_test = train_test_split(self.people_data, self.people_labels, test_size=test_size, random_state=random)
    
    def get_image_size(self):
        return (self.image_height, self.image_width)
       
    def get_data(self):
        return (self.data_train, self.data_test)
    
    def get_labels(self):
        return (self.labels_train, self.labels_test)        
