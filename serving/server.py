#!/usr/bin/python
import os
import ast
from flask import Flask, jsonify, request

from process_data import predict_data

app = Flask(__name__)

abspath = os.path.abspath(__file__)
proj_dir = '/'.join(abspath.split(os.sep)[:-2])
model_path = proj_dir + '/model'


@app.route('/')
def index():
    return 'index page'


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.json.get('data')
        if data:
            pred = predict_data(model_path, ast.literal_eval(data))
            return jsonify(result=pred)
        else:
            return jsonify(result='empty input')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port='8889', debug=True)
