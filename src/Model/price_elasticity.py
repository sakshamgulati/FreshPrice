import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import confuse

class pe_model:
    '''
    This class is used to infer the price elasticity for any product

    :param Input- dataset with price and volume sales
    :param Output- price elasticity of the product as computed from OLS
    '''
    def __init__(self, config_file="./FreshPrice/conf/DataOps/imagecrawler.yaml"):
        self.config = confuse.Configuration("FreshPrice", __name__)
        self.config.set_file(config_file)
        self.df_location = self.config["elasticity_data"].get(str)

    def model_training(self):
        '''
        This method is used to ingest,transform and fit an OLS to get the elasticity
        :param input- dataframe containing prices and volumes
        :return: elasticity of the product
        '''
        data=pd.read_csv(self.df_location)
        print("data loaded with: ",data.shape)
        data_ref = data.copy()
        data_ref = data_ref[['AveragePrice', 'Total Volume']]
        grp_data_ref = data_ref.groupby(['Total Volume', 'AveragePrice']).count().reset_index()
        del data_ref
        #Bifurcating into input and output
        X = grp_data_ref.drop(["Total Volume"], axis=1)
        y = grp_data_ref["Total Volume"]

        #Adding constant for the OLS
        X = sm.add_constant(X)
        #Split into train and test
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

        #Fitting the OLS model
        olsmodel = sm.OLS(y_train, x_train).fit()
        print(olsmodel.summary())

        #Calculating the mean price and quantity
        mean_price = np.mean(x_train['AveragePrice'])
        mean_quantity = np.mean(y_train)
        print("mean price: ", mean_price)
        print("mean quantity: ", mean_quantity)

        intercept, slope = olsmodel.params
        print("slope:",slope)

        price_elasticity = (slope) * (mean_price / mean_quantity)
        return price_elasticity





