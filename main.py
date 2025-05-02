from faceholder import FaceHolder
from pcamodel import PCAModel
from ldamodel import LDAModel
from siftmodel import SIFTModel
from hogmodel import HOGModel
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

holder = FaceHolder()

holder.get_lfw()
holder.split_data()

pcamodel = PCAModel()
ldamodel = LDAModel(holder.classes)
siftmodel = SIFTModel()
hogmodel = HOGModel()

print("PCA model working...")
pcamodel.do_pipeline(*holder.get_data(), *holder.get_labels())
print("PCA done.")
print("LDA model working...")
ldamodel.do_pipeline(*pcamodel.get_data(), *holder.get_labels())
print("LDA done.")
print("SIFT model working... This might take a while...")
siftmodel.do_pipeline(*holder.get_data(), *holder.get_labels(), *holder.get_image_size())
print("SIFT done.")
print("HOG model working... Also quite slow...")
hogmodel.do_pipeline(*holder.get_data(), *holder.get_labels(), *holder.get_image_size())
print("HOG done.")

pcamodel.print_accuracy()
ldamodel.print_accuracy()
siftmodel.print_accuracy()
hogmodel.print_accuracy()
pcaMatrix = ConfusionMatrixDisplay(confusion_matrix=pcamodel.confusion_matrix, display_labels=holder.target_names)
ldaMatrix = ConfusionMatrixDisplay(confusion_matrix=ldamodel.confusion_matrix, display_labels=holder.target_names)
siftMatrix = ConfusionMatrixDisplay(confusion_matrix=siftmodel.confusion_matrix, display_labels=holder.target_names)
hogMatrix = ConfusionMatrixDisplay(confusion_matrix=hogmodel.confusion_matrix, display_labels=holder.target_names)

fig, axes = plt.subplots(2, 2)
fig.suptitle("Confusion matrices")
fig.set_size_inches(4, 4)
pcaMatrix.plot(ax=axes[0, 0], xticks_rotation="vertical")
ldaMatrix.plot(ax=axes[0, 1], xticks_rotation="vertical")
siftMatrix.plot(ax=axes[1, 0], xticks_rotation="vertical")
hogMatrix.plot(ax=axes[1, 1], xticks_rotation="vertical")

plt.show()
