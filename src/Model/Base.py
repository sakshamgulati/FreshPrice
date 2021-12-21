from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential


class model:
    """
    This class is used to train a baseline model
    It utilizes basic conv and max pooling years with 0.2 dropout
    output: Predicts whether the banana is riped or not
    input: images (unriped and overriped)

    """

    @staticmethod
    def model_trainer(img_data, class_name, epochs):

        model_object = Sequential()
        model_object.add(
            Conv2D(28, kernel_size=(3, 3), input_shape=(200, 200, 3), activation="relu")
        )

        model_object.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
        model_object.add(MaxPooling2D(pool_size=(2, 2)))
        model_object.add(Conv2D(128, kernel_size=(2, 2), activation="relu"))
        model_object.add(Conv2D(128, kernel_size=(2, 2), activation="relu"))
        model_object.add(Flatten())
        model_object.add(Dense(256, activation="relu"))
        model_object.add(Dropout(0.2))
        model_object.add(Dense(1, activation="sigmoid"))

        model_object.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        print("model fitting underway...Please expect some delay")
        model_object.fit(x=img_data, y=class_name, epochs=epochs)
        print("model training completed!")

        return model_object

    @staticmethod
    def model_saver(model_obj):
        try:
            model_obj.save("./FreshPrice/Output/my_model.h5")
        except:
            print("Model not saved successfully!")
