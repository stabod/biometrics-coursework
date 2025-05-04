from .visualizer import Visualizer
from .faceholder import FaceHolder
from models import *

class ModelManager():
    def __init__(self, face_holder: FaceHolder):
        self.models = list()
        self.holder = face_holder
        if type(self.holder.target_names) == None:
            self.holder.get_lfw()
        self.visualizer = Visualizer(self.holder.target_names)
        
    def add_model(self, model: FaceModel):
        if not isinstance(model, FaceModel):
            raise TypeError("Model is not of correct type.") 
        self.models.append(model)
        
    def run(self, loops=3, visualize=False):
        for i in range(loops):
            print(f"Running loop {i+1}")
            save_data_train, save_data_test = None, None
            for model in self.models:
                self.holder.split_data()
                print(f"Running model {model.get_name()}")
                if model.get_name() == "LDA" and (type(save_data_train) != None and type(save_data_test) != None):
                    model.do_pipeline(save_data_train, save_data_test, *self.holder.get_labels(), *self.holder.get_image_size())
                else:
                    model.do_pipeline(*self.holder.get_data(), *self.holder.get_labels(), *self.holder.get_image_size())
                model.print_accuracy()
                if model.get_name() == "PCA":
                    save_data_train = model.data_train
                    save_data_test = model.data_test
                if visualize == True:
                    self.visualizer.add_matrix(model.get_matrix())
                model.clear()
        if visualize == True:
            self.visualizer.show()
            self.visualizer.clear()
        print()
        
        