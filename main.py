from models import *
from helpers import *

holder = FaceHolder(min_faces=60, resize=1)
holder.get_lfw()

manager = ModelManager(holder)
manager.add_model(PCAModel())
manager.add_model(LDAModel())
manager.add_model(SIFTModel())
manager.add_model(HOGModel())

manager.run()