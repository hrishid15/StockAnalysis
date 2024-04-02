company = "AAPL"
n = 30

#IMPORTS
import math
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def StockPredictor(TICKER, days):
    errorVals = []
    run = 0
    
    while run < 5:
        print(f"Run {run+1}:")
        df = yf.Ticker(TICKER)
        df = df.history(period="max")

        del df["Dividends"]
        del df["Stock Splits"]

        fiveYearDF = df.loc["2017-01-01":].copy()

        closingData = fiveYearDF.filter(['Close'])
        closingVals = closingData.values

        trainDataLength = math.ceil(len(closingVals) * 0.8)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaledData = scaler.fit_transform(closingVals)

        train = scaledData[0:trainDataLength, :]
        x_train = []
        y_train = []

        for i in range(days, len(train)):
            x_train.append(train[i-days:i, 0])
            y_train.append(train[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        model.fit(x_train, y_train, batch_size=1, epochs=20)

        test = scaledData[trainDataLength-days: , :]

        x_test, y_test = [], closingVals[trainDataLength: , :]
        for i in range(days, len(test)):
            x_test.append(test[i-days:i, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        rmse = np.sqrt(np.mean((predictions - y_test)**2))
        print(rmse)
        errorVals.append(rmse)
        run += 1

    error = math.ceil(min(errorVals))

    while rmse > error:
        df = yf.Ticker(TICKER)
        df = df.history(period="max")

        del df["Dividends"]
        del df["Stock Splits"]

        fiveYearDF = df.loc["2017-01-01":].copy()

        closingData = fiveYearDF.filter(['Close'])
        closingVals = closingData.values

        trainDataLength = math.ceil(len(closingVals) * 0.8)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaledData = scaler.fit_transform(closingVals)

        train = scaledData[0:trainDataLength, :]
        x_train = []
        y_train = []

        for i in range(days, len(train)):
            x_train.append(train[i-days:i, 0])
            y_train.append(train[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        model.fit(x_train, y_train, batch_size=1, epochs=20)

        test = scaledData[trainDataLength-days: , :]

        x_test, y_test = [], closingVals[trainDataLength: , :]
        for i in range(days, len(test)):
            x_test.append(test[i-days:i, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        rmse = np.sqrt(np.mean((predictions - y_test)**2))
        print(rmse)
    
    pred_df = yf.Ticker(TICKER)
    pred_df = pred_df.history(period="max")
    pred_df = pred_df.filter(['Close'])

    last_n_days = pred_df[-days:].values
    last_n_days_scaled = scaler.transform(last_n_days)
    X = []
    X.append(last_n_days_scaled)
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    pred_price = model.predict(X)
    pred_price = scaler.inverse_transform(pred_price)
    print(f"Tomorrow's Closing Price: {float(pred_price)}")

    pred_df = yf.Ticker(TICKER)
    pred_df = pred_df.history(period="max")
    pred_df = pred_df.filter(['Close'])
    day = 0
    while day < 7:
        last_n_days = pred_df[-days:].values
        last_n_days_scaled = scaler.transform(last_n_days)
        X = []
        X.append(last_n_days_scaled)
        X = np.array(X)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        pred_price = model.predict(X)
        pred_price = scaler.inverse_transform(pred_price)
        pred_df.loc[len(pred_df)] = [float(pred_price)]
        pred_df.index = pred_df.index[:-1].to_list() + [(pred_df.index[-2] +  pd.Timedelta(days=1))]
        day +=1
    
    plt.figure(figsize=(12, 6))
    plt.title("Next N Days")
    plt.xlabel("Date")
    plt.ylabel("Price")

    plt.plot(pred_df["Close"][-90:])
    plt.show()


StockPredictor(company, n)