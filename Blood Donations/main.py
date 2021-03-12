from flask import Flask, render_template, url_for, request
import os
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open('logistic_regression_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final = [np.array(int_features)]
    #print(int_features)
    #print(final)
    predictions = model.predict(final)
    #print(predictions)
    output = predictions[0]
    #output = "{:f}".format(predictions[0][1])
    #print(output)
    #return render_template('predict.html', prediction_text="Prediction is {}".format(output))

    if predictions == 0:
        statement = "The Donor will not donate blood in given time"
    else:
        statement = "The Donor will donate blood in given time"
    return render_template('predict.html', statement=statement, result=output)


if __name__ == '__main__':
    app.run(debug=True)
