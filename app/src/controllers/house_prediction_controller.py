import pickle
from typing import Any
from flask import jsonify, Blueprint, request
from flask_expects_json import expects_json
from sklearn.pipeline import Pipeline
from utils.data_checker import json_to_df, SINGLE_PREDICTION_SCHEMA

house_prediction_api: Blueprint = Blueprint(
    'house_prediction_api', __name__, url_prefix='/api/v1')

pipeline: Pipeline

with open("./model/pipeline.model", 'rb') as file:
    pipeline = pickle.load(file)


@house_prediction_api.route('/predict', methods=['POST'])
@expects_json(SINGLE_PREDICTION_SCHEMA)
def model_predict():
    """ Predicts one sample and returns the prediction
    """
    json: Any = request.json
    response = {
        "predicted_value": pipeline.predict(json_to_df(json))[0]
    }
    return jsonify(response), 200
