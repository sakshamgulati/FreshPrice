from Preprocessor.ImagePreprocessor import *
from Model.Base import *

# preprocessing the image data
preprocessor_class = preprocessor()
X, y = preprocessor_class.create_dataset()

# TODO:apply transformative function to y before model training

# training the base model
base_model = model()
trained_model = base_model.model_trainer(X, y, 5)
