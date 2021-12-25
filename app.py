import streamlit as st
from PIL import Image, ImageOps
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm


st.title("FreshPrice")
st.write(
    """
# This app is used to predict the price of an item based on its freshness

Please input the image to gauge its freshness

Created by Saksham Gulati

"""
)

salvage_price = st.number_input(
    "Please input the price of an over riped avocado (could be salvage price) :",
    min_value=0.00,
    max_value=10.0,
    value=0.0,
    step=0.01,
)

underriped_price = st.number_input(
    "Please input the price of an under riped avocado",
    min_value=0.01,
    max_value=10.0,
    value=2.0,
    step=0.01,
)


optimal_price = st.number_input(
    "Please input the price of an optimal avocado :",
    min_value=0.01,
    max_value=10.0,
    value=5.0,
    step=0.01,
)


@st.cache(allow_output_mutation=True)
def classifier(img, weights_file):
    # Load the model
    model = keras.models.load_model(weights_file)

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 200, 200, 3), dtype=np.float32)
    image = img
    # image sizing
    size = (200, 200)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = image_array.astype(np.float32) / 255

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction_percentage = model.predict(data)
    prediction = prediction_percentage.round()

    return prediction, prediction_percentage


def model_training():
    """
    This method is used to ingest,transform and fit an OLS to get the elasticity
    :param input- dataframe containing prices and volumes
    :return: elasticity of the product
    """
    data = pd.read_csv("./Data/Price-Sales Volume/avocado.csv")
    print("data loaded with: ", data.shape)
    data_ref = data.copy()
    data_ref = data_ref[["AveragePrice", "Total Volume"]]
    grp_data_ref = (
        data_ref.groupby(["Total Volume", "AveragePrice"]).count().reset_index()
    )
    del data_ref
    # Bifurcating into input and output
    X = grp_data_ref.drop(["Total Volume"], axis=1)
    y = grp_data_ref["Total Volume"]

    # Adding constant for the OLS
    X = sm.add_constant(X)
    # Split into train and test
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )

    # Fitting the OLS model
    olsmodel = sm.OLS(y_train, x_train).fit()
    print(olsmodel.summary())

    # Calculating the mean price and quantity
    mean_price = np.mean(x_train["AveragePrice"])
    mean_quantity = np.mean(y_train)
    print("mean price: ", mean_price)
    print("mean quantity: ", mean_quantity)

    intercept, slope = olsmodel.params
    print("slope:", slope)

    price_elasticity = (slope) * (mean_price / mean_quantity)
    return price_elasticity


uploaded_file = st.file_uploader(
    "Please upload a JPG image to be evaluated:", type="jpg"
)


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded file", use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label, perc = classifier(image, "my_model.h5")
    if label == 1:
        st.write("Its a over-riped Avocado")
    else:
        st.write("Its an under-riped Avocado!")

    x = [0, 0.5, 1]
    y = [underriped_price, optimal_price, salvage_price]

    z = np.polyfit(x, y, 3)
    f = np.poly1d(z)
    pred_optimal_price = f(perc[0][0]).round(2)

    st.text("Optimal Price(Based on Freshness) : ${}".format(pred_optimal_price))

    elastic_affect = model_training()
    # st.text("Elasticity:{:.2f}".format(elastic_affect))
    perc_change_price = ((pred_optimal_price - optimal_price) / optimal_price) * 100
    perc_quantity_change = perc_change_price * elastic_affect
    st.text(
        "with this {:.2f} % change in price, your quantities sold will change by {:.2f}%".format(
            perc_change_price, perc_quantity_change
        )
    )

    st.write(
        """
    ## Price-Quantity Calculator
    ### Please use this slider to determine prices and view corresponding change in quantity sold

    """
    )
    chosen_price = st.slider(
        "Price-Variability Calculator",
        min_value=salvage_price,
        max_value=optimal_price,
        value=float(pred_optimal_price),
        step=0.05,
        help="Choose a price to see variability in quantity",
    )

    perc_change_price_chosen = ((chosen_price - optimal_price) / optimal_price) * 100

    perc_quantity_change_chosen = perc_change_price_chosen * elastic_affect

    st.text(
        "with this {:.2f} % change in price, your quantities sold will change by {:.2f}%".format(
            perc_change_price_chosen, perc_quantity_change_chosen
        )
    )
