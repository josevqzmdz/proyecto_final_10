import numpy as np
import pandas as pd
import hvplot.pandas

from numpy.random import seed
seed(1)
from tensorflow import random
random.set_seed(2)

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

class BTC_predict:
    def __init__(self):
        df = pd.read_csv('info/BTC-sentiment.csv', index_col="date", infer_datetime_format=True, parse_dates=True)
        df = df.drop(columns="fng_classification")
        df.head()

        df2 = pd.read_csv('info/btc_historic.csv', index_col="Date", infer_datetime_format=True, parse_dates=True)['Close']
        df2 = df2.sort_index()
        df2.tail()

        df = df.join(df2, how="inner")
        df.tail()

        window_size = 10

        feature_column = 0
        target_column = 1
        X, y = self.window_data(df, window_size, feature_column, target_column)

        split = int(0.7 * len(X))

        X_train = X[: split]
        X_test = X[split:]

        y_train = y[: split]
        y_test = y[split:]

        scaler = MinMaxScaler()

        scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        scaler.fit(y_train)
        y_train = scaler.transform(y_train)
        y_test = scaler.transform(y_test)

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        model = Sequential()

        number_units = 30
        dropout_fraction = 0.2
        # capa 1
        model.add(LSTM(
            units=number_units,
            return_sequences=True,
            input_shape=(X_train.shape[1], 1))
        )
        model.add(Dropout(dropout_fraction))

        # capa 2
        model.add(LSTM(units=number_units, return_sequences=True))
        model.add(Dropout(dropout_fraction))

        # capa 3
        model.add(LSTM(units=number_units))
        model.add(Dropout(dropout_fraction))

        # capa exterior
        model.add(Dense(1))

        # compilacion
        model.compile(optimizer="adam", loss="mean_squared_error")

        # resumen
        model.summary()

        # entrena el modelo
        model.fit(X_train, y_train, epochs=10, shuffle=False, batch_size=90, verbose=1)

        # evalua el modelo
        model.evaluate(X_test, y_test, verbose=0)

        # hace predicciones
        predicted = model.predict(X_test)

        # recupera los precios originales en lugar de la version escalada
        predicted_prices = scaler.inverse_transform(predicted)
        real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

        # dataframe
        stocks = pd.DataFrame({
            "Real": real_prices.ravel(),
            "Predicted": predicted_prices.ravel()
        }, index=df.index[-len(real_prices):])
        stocks.head()

        # pequeño diagrama
        stocks.plot(title="Actual Vs. Predicted BTC Prices - Using Fear v Greed Index")

    def window_data(df, window, feature_col_number, target_col_number):
        X = []
        y = []
        for i in range(len(df) - window - 1):
            features = df.iloc[i:(i + window), feature_col_number]
            target = df.iloc[(i + window), target_col_number]
            X.append(features)
            y.append(target)
        return np.array(X), np.array(y).reshape(-1, 1)

