# crypto_analysis.py

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import streamlit as st
import plotly.graph_objs as go
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense


# Streamlit UI

st.title("Cryptocurrency Price Analysis & Forecasting")
st.sidebar.header("Settings")

crypto_symbol = st.sidebar.text_input("Enter Crypto Symbol (e.g., BTC-USD, ETH-USD)", "BTC-USD")
start_date = st.sidebar.date_input("Start Date", datetime(2021,1,1))
end_date = st.sidebar.date_input("End Date", datetime.today())


# Data Collection
@st.cache_data
def load_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    data.reset_index(inplace=True)
    return data

data = load_data(crypto_symbol, start_date, end_date)

st.subheader(f"Historical Data for {crypto_symbol}")
st.dataframe(data.tail())


# Visualization
st.subheader("Price Trend")
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price'))
fig.update_layout(title=f'{crypto_symbol} Closing Price', xaxis_title='Date', yaxis_title='Price (USD)')
st.plotly_chart(fig)


# Data Preprocessing
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)


# Forecasting Models
st.subheader("Forecasting")
forecast_period = st.sidebar.number_input("Forecast Days", min_value=1, max_value=365, value=30)

# ARIMA Forecast
st.markdown(" ARIMA Forecast")
arima_model = ARIMA(data['Close'], order=(5,1,0))
arima_result = arima_model.fit()
arima_forecast = arima_result.forecast(steps=forecast_period)
st.line_chart(pd.DataFrame({'ARIMA Forecast': arima_forecast}))

# Prophet Forecast
st.markdown(" Prophet Forecast")
prophet_df = data['Close'].reset_index().rename(columns={'Date':'ds','Close':'y'})
prophet_model = Prophet(daily_seasonality=True)
prophet_model.fit(prophet_df)
future = prophet_model.make_future_dataframe(periods=forecast_period)
forecast = prophet_model.predict(future)
st.line_chart(forecast.set_index('ds')[['yhat','yhat_lower','yhat_upper']].tail(forecast_period))

#  LSTM Forecast
st.markdown(" LSTM Forecast")
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

train_len = len(scaled_data) - forecast_period
train_data = scaled_data[:train_len]
test_data = scaled_data[train_len:]

def create_dataset(dataset, time_step=60):
    X, Y = [], []
    for i in range(len(dataset)-time_step-1):
        X.append(dataset[i:(i+time_step),0])
        Y.append(dataset[i+time_step,0])
    return np.array(X), np.array(Y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1],1)))
lstm_model.add(LSTM(50))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error', optimizer='adam')
lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

# Prepare test data
X_test, y_test = create_dataset(scaled_data, time_step)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
lstm_forecast = lstm_model.predict(X_test)
lstm_forecast = scaler.inverse_transform(lstm_forecast)

st.line_chart(pd.DataFrame({'LSTM Forecast': lstm_forecast[-forecast_period:]}))


# Volatility Analysis

st.subheader("Volatility Analysis")
data['Returns'] = data['Close'].pct_change()
st.line_chart(data['Returns'].rolling(7).std()*100)


# Sentiment Analysis Placeholder

st.subheader("Sentiment Analysis")
st.info("Sentiment analysis from news and social media can be added here using NLP models.")

st.success("Crypto Analysis & Forecasting Completed ")
