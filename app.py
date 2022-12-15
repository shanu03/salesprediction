from flask import Flask, request, jsonify, session
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from flask_cors import CORS
from datetime import datetime
import pandas.api.types
import os
import shutil
import joblib
import warnings

warnings.warn('ignore', category=FutureWarning)

app = Flask(__name__)
cors = CORS(app)
ALLOWED_EXTENSIONS = (['csv'])
app.secret_key = "abcdef"


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extraTreesRegressor():
    clf = ExtraTreesRegressor(n_estimators=100, max_features='auto', verbose=1, n_jobs=1)
    return clf


def predict_(m, test_x):
    return pd.Series(m.predict(test_x))


def model_():
    return extraTreesRegressor()


def train_(train_x, train_y):
    m = model_()
    m.fit(train_x, train_y)
    return m


def train_and_predict(train_x, train_y, test_x):
    m = train_(train_x, train_y)
    return predict_(m, test_x), m


def calculate_error(test_y, predicted):
    return mean_absolute_error(test_y, predicted)


@app.route('/load_data', methods=["GET", "POST"])
def data_load():
    if request.method == 'POST':
        UPLOAD_FOLDER_1 = 'data/'
        UPLOAD_FOLDER_2 = 'models/'
        if os.path.isdir(UPLOAD_FOLDER_1):
            shutil.rmtree('data')
        if os.path.isdir(UPLOAD_FOLDER_2):
            shutil.rmtree('models')
        if not os.path.isdir(UPLOAD_FOLDER_1):
            os.mkdir(UPLOAD_FOLDER_1)
        if not os.path.isdir(UPLOAD_FOLDER_2):
            os.mkdir(UPLOAD_FOLDER_2)

        app.config['UPLOAD_FOLDER_1'] = UPLOAD_FOLDER_1
        if 'file' not in request.files:
            return jsonify('No file part')
        file = request.files['file']
        if file.filename == '':
            return jsonify('No selected file')
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER_1'], file.filename)
            file.save(filepath)
        else:
            return jsonify('Unsupported File Format')
        df = pd.read_csv(filepath)
        session['filepath'] = filepath
        cols = list(df.columns)
        session['columns'] = cols
        cols_response = {'columns': session['columns']}
        return cols_response


@app.route('/train', methods=["GET", "POST"])
def model_training():
    if request.method == 'POST':
        file = os.listdir('data/')
        dataset = pd.read_csv('data/' + file[0])
        target = request.form['target']

        if pandas.api.types.is_numeric_dtype(dataset[target]):
            dataset['Month'] = pd.to_datetime(dataset['date']).dt.month
            dataset = dataset.select_dtypes(exclude=['object'])
            best_model = None
            X = dataset.drop(target, axis=1)
            y = dataset[target]
            train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.1, random_state=42)
            predicted, model = train_and_predict(train_x, train_y, test_x)
            error = calculate_error(test_y, predicted)
            r2 = r2_score(test_y, predicted)
            n = len(train_x)
            p = len(X.columns)
            Adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
            modelDetails = {'MAE': error, 'r2 score': r2, 'Adjusted r2 score': Adj_r2}
            joblib.dump(model, 'models/' + "model.pkl")
        else:
            return jsonify('Target Feature name should be Numeric')
        return modelDetails


@app.route('/save_model', methods=["GET", "POST"])
def savemodel():
    model_name = request.form['model_name']
    if not os.path.isdir('weights'):
        os.mkdir('weights')
    path = os.listdir('weights/')
    if "model.pkl" not in os.listdir('models/'):
        return jsonify("No weights found to save! Train model before saving!")
    if model_name + ".pkl" not in path:
        shutil.move('models/model.pkl', "weights/")
        new_name = model_name + '.pkl'
        os.rename('weights/model.pkl', 'weights/' + new_name)
    else:
        return jsonify('Model Name already exists!')
    return jsonify('Model Saved Successfully')


@app.route('/model_list', methods=["GET", "POST"])
def model_list():
    path = os.listdir('weights/')
    lis = []
    for i in range(len(path)):
        lis.append(path[i][:-4])
    available_models = {"Models": lis}
    return available_models


@app.route('/test', methods=['GET', 'POST'])
def model_test():
    if request.method == 'POST':
        UPLOAD_FOLDER = 'test_data/'
        file = request.files['test_file']
        if os.path.isdir(UPLOAD_FOLDER):
            shutil.rmtree('test_data')
        if not os.path.isdir(UPLOAD_FOLDER):
            os.mkdir(UPLOAD_FOLDER)
        if 'test_file' not in request.files:
            return jsonify('No file part')
        if file.filename == '':
            return jsonify('No selected file')
        if file and allowed_file(file.filename):
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
        else:
            return jsonify('Unsupported File Format')
        model_name = request.form['model_name']
        data = pd.read_csv(filepath)
        data['Month'] = pd.to_datetime(data['date']).dt.month
        data = data.select_dtypes(exclude=['object'])
        path = os.listdir('weights/')
        lis = []
        for i in range(len(path)):
            lis.append(path[i][:-4])
        available_models = {"Models": lis}
        print(data.columns)
        if model_name in available_models['Models']:
            model = joblib.load('weights/' + model_name + '.pkl')
        predicted_test = model.predict(data)
        data['result'] = predicted_test
        data.to_csv("Output.csv")
        return data.to_dict()

    else:
        return jsonify('GET method is not supported')


if __name__ == '__main__':
    app.run(debug=True)
