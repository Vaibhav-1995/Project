# Project
## Stock Market Price Predictor Using Stacked LSTM

Stock market prediction is prediction of prices based on historical market data. Every company listed on the market provides market data related to the transactions that happen on the stock.This data mainly comprises OpenPrice, HighPrice, LowPrice and ClosePrice. There are various technical indicators which are derived from these prices. The idea is to build a predictive solution to predict these prices based on machine learning of historical data and to be able to visualize data in some form .(Eg Graphs) where x axis defines time and y axis defines the price.

## Project Architecture
### 1. Preprocessing of Data
![image](https://user-images.githubusercontent.com/103194544/194383306-32a91bad-d294-43b7-ad64-27880f480598.png)
### 2. Overall Architecture
![image](https://user-images.githubusercontent.com/103194544/194383479-6c74dbfe-8771-42f3-bd50-55aedf92639a.png)
### 3.  Structure Chart
![image](https://user-images.githubusercontent.com/103194544/194384760-89d8c23e-0f06-406f-a7ec-95fb37271cb0.png)

## LSTM Model Working
### LSTM Architecture
![image](https://user-images.githubusercontent.com/103194544/194385775-f552d212-929b-4fa1-9b27-b5889f6d500c.png)

#### Forget Gate: 
A forget gate is responsible for removing information from the cell state. 
The information that is no longer required for the LSTM to understand things or the information that is of less importance is removed via multiplication of a filter. 
This is required for optimizing the performance of the LSTM network.
This gate takes in two inputs; h_t-1 and x_t. h_t-1 is the hidden state from the previous cell or the output of the previous cell and x_t is the input at that    particular time step. 
#### Input Gate: 
Regulating what values need to be added to the cell state by involving a sigmoid function. This is basically very similar to the forget gate and acts as a filter for all the information from hi-1 and x_t. 
Creating a vector containing all possible values that can be added (as perceived from h_t-1 and x_t) to the cell state. This is done using the tanh function, which outputs values from -1 to +1.
Multiplying the value of the regulatory filter (the sigmoid gate) to the created vector (the tanh function) and then adding this useful information to the cell state via addition operation. 
#### Output Gate:
The functioning of an output gate can again be broken down to three steps:
Creating a vector after applying tanh function to the cell state, thereby scaling the values to the range -1 to +1.
Making a filter using the values of h_t-1 and x_t, such that it can regulate the values that need to be output from the vector created above. This filter again employs a sigmoid function.
Multiplying the value of this regulatory filter to the vector created in step 1, and sending it out as a output and also to the hidden state of the next cell.
