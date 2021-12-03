from Preprocessor.ImagePreprocessor import *
from Model.Base import *
import time
from Evaluator.model_evaluator import *

# preprocessing the image data
preprocessor_class = preprocessor()
img_folder = (
    "C:/Users/saksh/PycharmProjects/FreshPrice/FreshPrice/Data/Image_data/train"
)
img_folder_test = (
    "C:/Users/saksh/PycharmProjects/FreshPrice/FreshPrice/Data/Image_data/test"
)

X, y = preprocessor_class.create_dataset(img_folder)

# Converting string labels to encodings
y = np.array(list(map(lambda p: preprocessor_class.product_mapping(p), y)))
print("the datatype for labels has been changed from string to: ", type(y))
# training the base model
base_model = model()
start = time.time()

trained_model = base_model.model_trainer(X, y, 1)
end = time.time()

print("Model Training completed in(seconds): ", end - start)
evaluate = model_evaluator()
print("the accuracy of the model on training set is: ", evaluate.model_predict())
