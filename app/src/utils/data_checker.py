from typing import Any
import numpy as np
import pandas as pd

columns = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley',
           'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
           'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
           'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',
           'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',
           'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
           'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
           'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC',
           'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
           'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
           'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
           'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt',
           'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',
           'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
           'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal',
           'MoSold', 'YrSold', 'SaleType', 'SaleCondition']

columns_to_log = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallCond', 'MasVnrArea',
                  'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
                  '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtHalfBath', 'HalfBath',
                  'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageArea',
                  'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
                  'ScreenPorch', 'PoolArea', 'MiscVal']

SINGLE_PREDICTION_SCHEMA = {
    '$schema': 'http://json-schema.org/draft-07/schema#',
    '$id': 'http://json-schema.org/draft-07/schema#',
    'type': "object",
    'properties': {
        'MSZoning': {'type': 'string'},
        'Street': {'type': 'string'},
        'Alley': {'type': 'string'},
        'LotShape': {'type': 'string'},
        'LandContour': {'type': 'string'},
        'Utilities': {'type': 'string'},
        'LotConfig': {'type': 'string'},
        'LandSlope': {'type': 'string'},
        'Neighborhood': {'type': 'string'},
        'Condition1': {'type': 'string'},
        'Condition2': {'type': 'string'},
        'BldgType': {'type': 'string'},
        'HouseStyle': {'type': 'string'},
        'RoofStyle': {'type': 'string'},
        'RoofMatl': {'type': 'string'},
        'Exterior1st': {'type': 'string'},
        'Exterior2nd': {'type': 'string'},
        'MasVnrType': {'type': 'string'},
        'ExterQual': {'type': 'string'},
        'ExterCond': {'type': 'string'},
        'Foundation': {'type': 'string'},
        'BsmtQual': {'type': 'string'},
        'BsmtCond': {'type': 'string'},
        'BsmtExposure': {'type': 'string'},
        'BsmtFinType1': {'type': 'string'},
        'BsmtFinType2': {'type': 'string'},
        'Heating': {'type': 'string'},
        'HeatingQC': {'type': 'string'},
        'CentralAir': {'type': 'string'},
        'Electrical': {'type': 'string'},
        'KitchenQual': {'type': 'string'},
        'Functional': {'type': 'string'},
        'FireplaceQu': {'type': 'string'},
        'GarageType': {'type': 'string'},
        'GarageFinish': {'type': 'string'},
        'GarageQual': {'type': 'string'},
        'GarageCond': {'type': 'string'},
        'PavedDrive': {'type': 'string'},
        'PoolQC': {'type': 'string'},
        'Fence': {'type': 'string'},
        'MiscFeature': {'type': 'string'},
        'SaleType': {'type': 'string'},
        'SaleCondition': {'type': 'string'},
        'MSSubClass': {'type': 'number'},
        'LotFrontage': {'type': 'number'},
        'LotArea': {'type': 'number'},
        'OverallQual': {'type': 'number'},
        'OverallCond': {'type': 'number'},
        'YearBuilt': {'type': 'number'},
        'YearRemodAdd': {'type': 'number'},
        'MasVnrArea': {'type': 'number'},
        'BsmtFinSF1': {'type': 'number'},
        'BsmtFinSF2': {'type': 'number'},
        'BsmtUnfSF': {'type': 'number'},
        'TotalBsmtSF': {'type': 'number'},
        '1stFlrSF': {'type': 'number'},
        '2ndFlrSF': {'type': 'number'},
        'LowQualFinSF': {'type': 'number'},
        'GrLivArea': {'type': 'number'},
        'BsmtFullBath': {'type': 'number'},
        'BsmtHalfBath': {'type': 'number'},
        'FullBath': {'type': 'number'},
        'HalfBath': {'type': 'number'},
        'BedroomAbvGr': {'type': 'number'},
        'KitchenAbvGr': {'type': 'number'},
        'TotRmsAbvGrd': {'type': 'number'},
        'Fireplaces': {'type': 'number'},
        'GarageYrBlt': {'type': 'number'},
        'GarageCars': {'type': 'number'},
        'GarageArea': {'type': 'number'},
        'WoodDeckSF': {'type': 'number'},
        'OpenPorchSF': {'type': 'number'},
        'EnclosedPorch': {'type': 'number'},
        '3SsnPorch': {'type': 'number'},
        'ScreenPorch': {'type': 'number'},
        'PoolArea': {'type': 'number'},
        'MiscVal': {'type': 'number'},
        'MoSold': {'type': 'number'},
        'YrSold': {'type': 'number'}
    },
    'required': columns
}


def json_to_df(json: Any) -> pd.DataFrame:
    """converts json object to pandas DataFrame, 
    and does appropriate actions on the df

    Args:
        json (Any): The json object to convert

    Returns:
        DataFrame: the result DataFrame
    """
    df = pd.json_normalize(json)
    df[columns_to_log] = np.log1p(df[columns_to_log])
    return df
