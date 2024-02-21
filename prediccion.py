from flask import Flask, render_template
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)

import matplotlib
matplotlib.use('Agg')

@app.route('/')
def home():

    ventas_que_realizamos = 20

    #Usamos este modelo con dos variables ... de esta forma entreamos al modelo y cuando le pasamos en número de stock vendido nos predice cuánto stock restante nos quedará
    data = {'ventas': [10, 20, 15, 25, 30],
            'stock': [50, 40, 35, 30, 25]}

    df = pd.DataFrame(data)

    X = df[['ventas']]
    Y = df['stock']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = LinearRegression()

    model.fit(X_train, Y_train)

    predictions = model.predict(X_test)

    nuevas_ventas = np.array([ventas_que_realizamos]).reshape(-1, 1)  

    stock_restante_que_quiero_predecir = model.predict(nuevas_ventas)

    return str(stock_restante_que_quiero_predecir[0])

if __name__ == '__main__':
    app.run(debug=True)
