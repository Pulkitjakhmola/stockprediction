import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

st.title("Stock Price Predictor App")

st.markdown("### Developed by: Pulkit Jakhmola")  

stock = st.text_input("Enter the stock ID", "GOOG")
st.caption(" Add `.NS` after the stock ID for Indian stocks. For example: State Bank of India = `SBIN.NS` (where `SBIN` is the stock ID).")

end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)
google = yf.download(stock, start, end)

if google.empty:
    st.error("No data found for the entered stock ID. Please try a different ID.")
else:
    model = load_model("latest_stock_price.keras")

    st.subheader("Stock Data")
    st.write(google)

    splitting_len = int(len(google) * 0.7)
    x_test = google[['Close']].iloc[splitting_len:]

    def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
        fig = plt.figure(figsize=figsize)
        plt.plot(values, 'orange', label='Moving Average')
        plt.plot(full_data.Close, 'b', label='Actual Close Price')
        if extra_data:
            plt.plot(extra_dataset, 'green', label='Extra Data')
        plt.legend()
        return fig

    st.subheader('Actual Close Price and MA for 250 days')
    google['MA_for_250_days'] = google.Close.rolling(250).mean()
    st.pyplot(plot_graph((15, 5), google['MA_for_250_days'], google, 0))

    st.subheader('Actual Close Price and MA for 200 days')
    google['MA_for_200_days'] = google.Close.rolling(200).mean()
    st.pyplot(plot_graph((15, 5), google['MA_for_200_days'], google, 0))

    st.subheader('Actual Close Price and MA for 100 days')
    google['MA_for_100_days'] = google.Close.rolling(100).mean()
    st.pyplot(plot_graph((15, 5), google['MA_for_100_days'], google, 0))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(x_test[['Close']])

    x_data = []
    y_data = []
    for i in range(100, len(scaled_data)):
        x_data.append(scaled_data[i-100:i])
        y_data.append(scaled_data[i])

    x_data, y_data = np.array(x_data), np.array(y_data)

    predictions = model.predict(x_data)
    inv_pre = scaler.inverse_transform(predictions)
    inv_y_test = scaler.inverse_transform(y_data)

    plotting_data = pd.DataFrame(
        {
            'original_test_data': inv_y_test.reshape(-1),
            'predictions': inv_pre.reshape(-1)
        },
        index=google.index[splitting_len + 100:]
    )

    st.subheader("Original values vs. predicted values")
    st.write(plotting_data)

    st.subheader('Original close price vs. predicted close price')
    fig = plt.figure(figsize=(15, 6))
    plt.plot(google.Close[:splitting_len + 100], 'blue', label='Data - not used')
    plt.plot(plotting_data['original_test_data'], 'orange', label='Original Test Data')
    plt.plot(plotting_data['predictions'], 'green', label='Predicted Test Data')
    plt.legend()
    st.pyplot(fig)

    last_100_days = scaled_data[-100:] 
    last_100_days = last_100_days.reshape((1, last_100_days.shape[0], last_100_days.shape[1]))  

    tomorrow_prediction = model.predict(last_100_days)
    tomorrow_prediction_inv = scaler.inverse_transform(tomorrow_prediction) 

    st.subheader("Tomorrow's Predicted Price")
    st.write(f"The predicted price for {stock} tomorrow is: {tomorrow_prediction_inv[0][0]:.2f}")
