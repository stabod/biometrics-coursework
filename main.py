from faceholder import FaceHolder
from pcamodel import PCAModel
from ldamodel import LDAModel
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

holder = FaceHolder()

holder.get_lfw()
holder.split_data()

pcamodel = PCAModel()
ldamodel = LDAModel(holder.classes)

pcamodel.do_pipeline(*holder.get_data(), *holder.get_labels())
ldamodel.do_pipeline(*pcamodel.get_data(), *holder.get_labels())

pcamodel.print_accuracy()
ldamodel.print_accuracy()
pcaMatrix = ConfusionMatrixDisplay(confusion_matrix=pcamodel.confusion_matrix, display_labels=holder.target_names)
ldaMatrix = ConfusionMatrixDisplay(confusion_matrix=ldamodel.confusion_matrix, display_labels=holder.target_names)

fig, axes = plt.subplots(1, 2)
fig.suptitle("Confusion matrices")
pcaMatrix.plot(ax=axes[0], xticks_rotation="vertical")
ldaMatrix.plot(ax=axes[1], xticks_rotation="vertical")

plt.show()
