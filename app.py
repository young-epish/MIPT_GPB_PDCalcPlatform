from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle


app = Flask(__name__)

# Загрузка моделей и объекта масштабирования
model_sp = pickle.load(open('models/model1.pkl', 'rb'))
model_pr = pickle.load(open('models/model2.pkl', 'rb'))

scaler_sp = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')