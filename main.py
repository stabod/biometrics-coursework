from faceholder import FaceHolder
from models import *
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

holder = FaceHolder(min_faces=60, resize=1)

holder.get_lfw()

models = list()

models.append(PCAModel())
models.append(LDAModel())
models.append(SIFTModel())
models.append(HOGModel())

for i in range(3):
    print(f"Loop {i+1}")
    save_data_train, save_data_test = None, None
    for model in models:
        holder.split_data()
        print(f"Running model {model.get_name()}")
        if model.get_name() == "LDA":
            model.do_pipeline(save_data_train, save_data_test, *holder.get_labels(), *holder.get_image_size())
        else:
            model.do_pipeline(*holder.get_data(), *holder.get_labels(), *holder.get_image_size())
        model.print_accuracy()
        if model.get_name() == "PCA":
            save_data_train = model.data_train
            save_data_test = model.data_test
        model.clear()
    print()
    
#pcaMatrix = ConfusionMatrixDisplay(confusion_matrix=pcamodel.confusion_matrix, display_labels=holder.target_names)
#ldaMatrix = ConfusionMatrixDisplay(confusion_matrix=ldamodel.confusion_matrix, display_labels=holder.target_names)
#siftMatrix = ConfusionMatrixDisplay(confusion_matrix=siftmodel.confusion_matrix, display_labels=holder.target_names)
#hogMatrix = ConfusionMatrixDisplay(confusion_matrix=hogmodel.confusion_matrix, display_labels=holder.target_names)

#fig, axes = plt.subplots(2, 2)
#fig.suptitle("Confusion matrices")
#pcaMatrix.plot(ax=axes[0, 0], xticks_rotation="vertical")
#ldaMatrix.plot(ax=axes[0, 1], xticks_rotation="vertical")
#siftMatrix.plot(ax=axes[1, 0], xticks_rotation="vertical")
#hogMatrix.plot(ax=axes[1, 1], xticks_rotation="vertical")

#fig.tight_layout()
#plt.show()
