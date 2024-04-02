# StockAnalysis
 Stock Market Analysis with Python

# Description

This Python script  uses machine learning techniques to predict stock prices for a given stock ticker symbol and a specified number of days. It employs the Long Short-Term Memory (LSTM) neural network model from the Keras library to make predictions based on historical stock data.

The program contains a function called StockPredictor that takes two arguments: the stock ticker symbol (TICKER) and the number of days (days) for which the predictions will be made.

Inside the StockPredictor function, the script performs the following steps:

- Fetches the historical stock data from Yahoo Finance using the yfinance library.
- Preprocesses the data by removing unnecessary columns and scaling the closing prices.
- Splits the data into training and testing sets.
- Defines and trains an LSTM model on the training data.
- Evaluates the model's performance by calculating the root mean squared error (RMSE) on the testing data.
- Iterates the training process multiple times to find the best-performing model with the lowest RMSE.
- Uses the best-performing model to predict the closing price for the next day and the subsequent 6 days.
- Plots the predicted prices for the next 7 days along with the historical data.
- The script also includes some error handling and prints out the RMSE values during the training process.

To use the script, simply call the StockPredictor function with the desired stock ticker symbol and the number of days for which you want to make predictions.

Note: This code requires the installation of several Python libraries, including yfinance, numpy, pandas, scikit-learn, keras, and matplotlib.
