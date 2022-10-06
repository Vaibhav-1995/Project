import streamlit as st
import pandas_datareader as data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import array
from datetime import datetime, timedelta
from keras.models import load_model

if __name__ == "__main__":
    st.set_page_config(page_title="ML Project", page_icon=":beginner:")
    st.title('Stock Market Price Predictor')
    st.write(
        """ The idea is to build a predictive solution :
        - To predict thes closing prices of for number of days stock based on machine learning.
        - Using web scrapping from YAHOO FINANCE historical data.
        - To be able to visualize data in some form of graphs, where x axis defines time and y axis defines the price.
        - To analyze stock trends using some technical indicators """
        )
    st.write('[YAHOO FINANCE](https://finance.yahoo.com)')

    start =  '1996-01-01'
    now=datetime.now()
    end=now.strftime("%Y-%m-%d")

    user_input=st.text_input('Enter Stock Ticker Symbol','SBIN.NS')
    df=data.DataReader(user_input,'yahoo',start,end)

    #Descirbing data
    st.subheader('Fetch some data till today')
    st.write(df.tail())

    def Plot(x):
        fig=plt.figure(figsize=(12,6))
        plt.plot(x,'b')
        st.pyplot(fig)

    #Visualization
    st.subheader('Close Price vs Year Chart')
    Plot(df['Close'])
    st.subheader('Volume vs Year Chart')
    Plot(df['Volume'])

    st.subheader('Close Price vs Year with 20 & 50 MA Chart')
    ma20 = df.Close.rolling(20).mean()
    ma50 = df.Close.rolling(60).mean()
    fig = plt.figure(figsize=(12,6))
    plt.plot(ma20,'g')
    plt.plot(ma50,'r')
    plt.plot(df['Close'],'gray')
    st.pyplot(fig)

    st.subheader('Close Price vs Year with 100 & 200 MA Chart')
    ma100=df.Close.rolling(100).mean()
    ma200=df.Close.rolling(200).mean()
    fig=plt.figure(figsize=(12,6))
    plt.plot(ma200,'r')
    plt.plot(ma100,'g')
    plt.plot(df['Close'],'gray')
    st.pyplot(fig)

    st.subheader('Close Price vs Year with 50 & 200 EMA Chart')
    ema50=df.Close.ewm(span=50,adjust=False).mean()
    ema200=df.Close.ewm(span=200,adjust=False).mean()
    fig=plt.figure(figsize=(12,6))
    plt.plot(ema50,'g')
    plt.plot(ema200,'r')
    plt.plot(df['Close'],'gray')
    st.pyplot(fig)

    #Splitting Data into Training and Testing
    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.80)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.80):int(len(df))])

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))

    data_training_array = scaler.fit_transform(data_training)

    #Loading Model
    model=load_model('model.h5')

    past_100_days = data_training.tail(100)
    final_df= past_100_days.append(data_testing,ignore_index=True)

    input_data= scaler.fit_transform(final_df)

    x_test=[]
    y_test=[]

    for i in range(100,input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i,0])
        
    x_test , y_test = np.array(x_test) , np.array(y_test)

    #Making Predictions
    y_predicted = model.predict(x_test)

    y_predicted=scaler.inverse_transform(y_predicted)
    y_test=y_test.reshape(-1,1)
    y_test=scaler.inverse_transform(y_test)

    #final Graph
    st.subheader('Close Price Prediction vs Original Chart')
    fig2 = plt.figure(figsize=(12,8))
    plt.plot(y_test , 'g', label='Current price')
    plt.plot(y_predicted , 'r', label='Predicted price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)

    #Predicted Price for tomorrow close
    last_100_days=df.Close[-100:]
    last_100_days=pd.DataFrame(last_100_days)
    last_100_days_scaled=scaler.transform(last_100_days)

    x_test=[]
    x_test.append(last_100_days_scaled)
    x_test=np.array(x_test)
    x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))


    pred_price=model.predict(x_test)
    model.reset_states()

    pred_price=scaler.inverse_transform(pred_price)
    st.write('Predicted price is',pred_price)

    #Predicting for N days
    n_steps=60
    pred_days=st.number_input('Enter No. of Days to be predicted',10)
    x_input=input_data[len(input_data)-n_steps:].reshape(1,-1)
    x_input.shape
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()

    # demonstrate prediction for next N days
    lst_output=[]
    i=0
    while(i<pred_days):
        
        if(len(temp_input)>60):
            #print(temp_input)
            x_input=np.array(temp_input[1:])
            print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1

    print(lst_output)

    predictions=scaler.inverse_transform(lst_output)
    st.write('Prediction for',pred_days,' days is',predictions)

    #Visualizing
    day_new=np.arange(1,61)
    day_pred=np.arange(61,61+pred_days)

    st.subheader('Future Prediction Close Price Chart')
    fig2 = plt.figure(figsize=(12,8))
    #plt.plot(ff,'b',label='CLosing')
    plt.plot(day_new,(df.Close[len(df)-60:]) , 'g', label='Current price')
    plt.plot(day_pred,scaler.inverse_transform(lst_output) , 'r', label='Predicted price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)