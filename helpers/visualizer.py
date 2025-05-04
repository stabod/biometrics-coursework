import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from math import ceil, sqrt

class Visualizer():
    def __init__(self, labels):
        self.matrices = list()
        self._vis = list()
        self.labels = labels
        
        self.fig = None
        self.axes = None
        
    def add_matrix(self, matrix):
        self.matrices.append(matrix)
        
    def _generate_visual(self):
        for item in self.matrices:
            result = ConfusionMatrixDisplay(confusion_matrix=item, display_labels=self.labels)
            self._vis.append(result)
        
        vis_num = len(self._vis)
        if vis_num > 1:
            rows = ceil(sqrt(vis_num))
            columns = ceil(vis_num/ rows)
            self.fig, self.axes = plt.subplots(rows, columns, constrained_layout=True)
            self.fig.suptitle("Confusion matrices")
            index = 0
            for y in range(rows):
                for x in range(columns):
                    if index < vis_num:
                        self._vis[index].plot(ax=self.axes[y, x], xticks_rotation="vertical")
                        index += 1
                    else:
                        self.fig.delaxes(self.axes[y,x])
            return    
        else:
            self.fig, self.axes = plt.subplots()
            self._vis[0].plot(ax=self.axes, xticks_rotation="vertical")
            return
            
    def show(self):
        self._generate_visual()
        plt.show()
    
    def clear(self):
        self.fig = None
        self.axes = None
        self._vis.clear()
        self.matrices.clear()
        
        