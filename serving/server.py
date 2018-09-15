#!/usr/bin/python
import os
import ast
import atexit
from flask import Flask, jsonify, request

from process_data import predict_data
from project.spark import get_spark, get_logger

app = Flask(__name__)

abspath = os.path.abspath(__file__)
proj_dir = '/'.join(abspath.split(os.sep)[:-2])
model_path = proj_dir + '/model'

spark = get_spark(app_name="pred")
logger = get_logger(spark, "pred_logger")


@app.route('/')
def index():
    return 'index page'


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.json.get('data')
        if data:
            pred = predict_data(spark, logger, model_path,
                                ast.literal_eval(data))
            return jsonify(result=pred)
        else:
            return jsonify(result='empty input')


def stop_spark():
    '''
        sever shutdown handler
    '''
    global spark
    spark.stop()
    logger.info("stop spark")


if __name__ == '__main__':
    atexit.register(stop_spark)
    app.run(host='127.0.0.1', port='8889', debug=True)
