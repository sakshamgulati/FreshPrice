from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from Model.pruning import *
import tempfile


class model:
    """
    This class is used to train a baseline model
    It utilizes basic conv and max pooling years with 0.2 dropout
    output: Predicts whether the banana is riped or not
    input: images (unriped and overriped)

    """

    @staticmethod
    def model_trainer(img_data, class_name, epochs):
        def model_func():
            model_object = Sequential()
            model_object.add(
                Conv2D(
                    28, kernel_size=(3, 3), input_shape=(200, 200, 3), activation="relu"
                )
            )

            model_object.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
            model_object.add(MaxPooling2D(pool_size=(2, 2)))
            # model_object.add(Conv2D(128, kernel_size=(2, 2), activation="relu"))
            # model_object.add(Conv2D(128, kernel_size=(2, 2), activation="relu"))
            model_object.add(Flatten())
            model_object.add(Dense(64, activation="relu"))
            model_object.add(Dropout(0.2))
            model_object.add(Dense(1, activation="sigmoid"))
            return model_object

        model_object = model_func()
        model_for_pruning = prune_low_magnitude(model_object, **pruning_params)

        model_for_pruning.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        logdir = tempfile.mkdtemp()

        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
            tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
        ]

        print("model fitting underway...Please expect some delay")
        model_for_pruning.fit(
            x=img_data, y=class_name, epochs=epochs, callbacks=callbacks
        )
        print("model training completed!")

        return model_object

    @staticmethod
    def model_saver(model_obj):
        try:
            model_obj.save("./FreshPrice/Output/my_model")
        except:
            print("Model not saved successfully!")
