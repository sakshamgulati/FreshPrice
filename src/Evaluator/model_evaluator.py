from Model.Base import model
from Preprocessor.ImagePreprocessor import *
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report
import confuse


class model_evaluator:
    """
    This class is used to evaluate the performance of the model
    :param- test data file

    :return- Accuracy score for the model

    # """

    def __init__(self, config_file="./FreshPrice/conf/ML/preprocessor.yaml"):
        logging.basicConfig(
            filename="./FreshPrice/Output/model_evaluator.log", level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)
        self.IMG_WIDTH = 200
        self.IMG_HEIGHT = 200
        self.config = confuse.Configuration("FreshPrice", __name__)
        self.config.set_file(config_file)
        self.img_folder_test = self.config["img_folder_test"].get(str)
        p = preprocessor()
        self.img_data_test, self.class_name_test_raw = p.create_dataset(
            self.img_folder_test
        )
        self.class_name_test = np.array(
            list(map(lambda p: p.product_mapping(p), self.class_name_test_raw))
        )

    def model_predict(self):
        preds = model.predict(self.img_data_test).round().astype(int)
        flat_pred = [item for sublist in preds for item in sublist]

        accuracy = accuracy_score(self.class_name_test, flat_pred)
        print("The Accuracy is: %2f" % accuracy)
        self.logger.info("Accuracy of the model", accuracy)
        print(
            "classification report is:",
            classification_report(self.class_name_test, flat_pred),
        )
        self.logger.info(
            "Classification report:",
            classification_report(self.class_name_test, flat_pred),
        )

        return accuracy

    def model_predict_proba(self):
        return model.predict(self.img_data_test)

