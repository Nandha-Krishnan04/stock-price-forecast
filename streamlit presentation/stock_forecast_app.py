import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

def main():
    st.title("Stock Price Forecasting App")

    symbol = st.text_input("Enter Stock Symbol (e.g., RELIANCE.NS):", "RELIANCE.NS")
    model_option = st.selectbox("Select Model for Forecasting:", ["LSTM", "PROPHET", "ARIMA", "SARIMA"])
    selected_days = st.selectbox("Select number of days to predict:", [30, 60, 90, 120])

    if st.button("Predict"):
        with st.spinner("Fetching data and making prediction..."):
            df = yf.download(symbol, start="2016-01-01")
            df = df.reset_index()
            df['Date'] = pd.to_datetime(df['Date'])

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)

            data = df[['Date', 'Close']].dropna()
            data_recent = data[data['Date'] >= data['Date'].max() - pd.DateOffset(years=1)]

            if model_option == "LSTM":
                data_lstm = data[['Close']].values
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(data_lstm)

                def create_dataset(data, time_step=60):
                    X, y = [], []
                    for i in range(time_step, len(data)):
                        X.append(data[i - time_step:i, 0])
                        y.append(data[i, 0])
                    return np.array(X), np.array(y)

                time_step = 60
                X_all, y_all = create_dataset(scaled_data, time_step)
                X_all = X_all.reshape((X_all.shape[0], X_all.shape[1], 1))

                model = Sequential()
                model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
                model.add(LSTM(50))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(X_all, y_all, epochs=10, batch_size=32, verbose=0)

                # Future forecasting
                future_input = scaled_data[-time_step:].flatten()
                future_preds_scaled = []
                for _ in range(selected_days):
                    input_seq = future_input[-time_step:].reshape(1, time_step, 1)
                    pred_scaled = model.predict(input_seq, verbose=0)[0][0]
                    future_preds_scaled.append(pred_scaled)
                    future_input = np.append(future_input, pred_scaled)

                future_preds = scaler.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1))
                last_date = data['Date'].iloc[-1]
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=selected_days)

                plt.figure(figsize=(12, 6))
                plt.plot(data_recent['Date'], data_recent['Close'], label="Historical Data (1Y)", color='blue')
                plt.plot(future_dates, future_preds, label="Future Forecast", color='green')
                plt.title("Stock Price Forecast - LSTM")
                plt.xlabel("Date")
                plt.ylabel("Price")
                plt.grid(True)
                plt.legend()
                st.pyplot(plt.gcf())
                plt.clf()

            elif model_option == "PROPHET":
                prophet_df = data.rename(columns={'Date': 'ds', 'Close': 'y'})
                model = Prophet()
                model.fit(prophet_df)

                future = model.make_future_dataframe(periods=selected_days)
                forecast = model.predict(future)

                fig1 = model.plot(forecast)
                st.pyplot(fig1)
                plt.clf()

            elif model_option == "ARIMA":
                arima_data = data.set_index('Date')['Close']
                arima_model = auto_arima(arima_data, seasonal=False, trace=False, suppress_warnings=True)
                best_order = arima_model.order

                model = ARIMA(arima_data, order=best_order)
                model_fit = model.fit()

                fitted_values = model_fit.fittedvalues
                future_forecast = model_fit.forecast(steps=selected_days)
                future_dates = pd.date_range(start=arima_data.index[-1] + pd.Timedelta(days=1), periods=selected_days)

                plt.figure(figsize=(12, 6))
                plt.plot(arima_data[-365:].index, arima_data[-365:].values, label="Historical Data (1Y)", color='blue')
                plt.plot(future_dates, future_forecast, label="Future Forecast", color='orange')
                plt.title("Stock Price Forecast - ARIMA")
                plt.xlabel("Date")
                plt.ylabel("Price")
                plt.grid(True)
                plt.legend()
                st.pyplot(plt.gcf())
                plt.clf()

            elif model_option == "SARIMA":
                sarima_data = data.set_index('Date')['Close']
                auto_model = auto_arima(sarima_data, seasonal=True, m=12, trace=False, stepwise=True, suppress_warnings=True)

                sarima_model = SARIMAX(sarima_data,
                                       order=auto_model.order,
                                       seasonal_order=auto_model.seasonal_order,
                                       enforce_stationarity=False,
                                       enforce_invertibility=False)
                model_fit = sarima_model.fit()

                forecast = model_fit.forecast(steps=selected_days)
                future_dates = pd.date_range(start=sarima_data.index[-1] + pd.Timedelta(days=1), periods=selected_days)

                plt.figure(figsize=(12, 6))
                plt.plot(sarima_data[-365:].index, sarima_data[-365:].values, label="Historical Data (1Y)", color='blue')
                plt.plot(future_dates, forecast, label="Future Forecast", color='purple')
                plt.title("Stock Price Forecast - SARIMA")
                plt.xlabel("Date")
                plt.ylabel("Price")
                plt.grid(True)
                plt.legend()
                st.pyplot(plt.gcf())
                plt.clf()

if __name__ == "__main__":
    main()
