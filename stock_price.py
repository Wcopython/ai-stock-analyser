import numpy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas
import math

# Keras is used to model the network
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import TensorBoard
# from keras.utils import normalize

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error

from model import build_lstm_model, prediction_sequence
from data_processing import build_stock_input, merge_data

def main():
    # Change to get new results
    numpy.random.seed(123456)

    # Reading stock data
    # Edit these strings to change which stocks to create predictions from.
    raw_1 = pandas.read_csv('data/PEAB.csv', usecols=[2], engine='python').values.astype('float32')
    raw_2 = pandas.read_csv('data/GOLD.csv', usecols=[2], engine='python').values.astype('float32')
    
    # plt.plot(stock_1)
    # plt.show()

    scaler_1 = preprocessing.StandardScaler().fit(raw_1)
    scaler_2 = preprocessing.StandardScaler().fit(raw_2)
    
    stock_1 = scaler_1.transform(raw_1)
    stock_2 = scaler_2.transform(raw_2)

    # Creating training and testing data
    stock_1 = build_stock_input(data_list = stock_1, input_size = 100, test_ratio = 0.1, step_size = 5)
    stock_2 = build_stock_input(data_list = stock_2, input_size = 100, test_ratio = 0.1, step_size = 5)

    train_x1 = stock_1[0]
    train_x2 = stock_2[0]
    train_y1 = stock_1[1]
    train_y2 = stock_2[1]

    test_x1 = stock_1[2]
    test_x2 = stock_2[2]
    test_y = stock_1[3]

    print("Loaded and processed data")
    
    # print(train_y)

    # # Creating a scaler object and apply it to the data set
    # Does not work in 2D
    # stocks_test = scaler.fit_transform(stocks_test)
    loadModelFromFile = False
    if not loadModelFromFile:

        model = build_lstm_model()

        # Training model
        tensorboard = TensorBoard(log_dir="logs/{}".format("lstm_test"))
        model.fit([train_x1, train_x2], [train_y1, train_y2], verbose=1, epochs = 15, callbacks=[tensorboard])
        
        model.save('save/model.h5')
        print("Result saved!")

    else:
        model = load_model('save/model.h5')

    # Creating predictions
    # predictions = prediction_invalid(model, [test_x1, test_x2], len(test_x1))[0]
    predictions = model.predict([test_x1, test_x2])[0]

    #Creating difference graph
    difference = []
    buy_sell = []
    threshold = 0.0035
    for i in range(1, len(predictions), 1):
        difference.append(predictions[i] - predictions[i-1])
        if (abs(difference[-1]) > threshold):
            buy_sell.append(round(abs(difference[-1][0])/difference[-1][0]))
        else:
            buy_sell.append(0)
    
    money_result = [0]
    nr_of_stocks = 0
    money = 100
    actual_price = build_stock_input(data_list = raw_1, input_size = 100, test_ratio = 0.1, step_size = 5)[3]
    
    # You get the result one day in advance!
    for i in range(1, len(buy_sell), 1):
        if buy_sell[i] > 0:
            nr_of_stocks += money / actual_price[i-1]
            money = 0
        elif buy_sell[i] < 0:
            money += nr_of_stocks * actual_price[i-1]
            nr_of_stocks = 0
        
        money_result.append(money + nr_of_stocks * actual_price[i])

    subPlots = True
    if subPlots:
        fig, ax = plt.subplots(nrows=4)

        ax[0].set_title("Predicted price (for tomorrow)")
        ax[0].plot(predictions)

        ax[1].set_title("Buy (1) /Sell (-1)")
        ax[1].plot(buy_sell)

        ax[2].set_title("Growth (initially 100)")
        ax[2].plot(money_result)

        ax[3].set_title("Actual data")
        ax[3].plot(actual_price)

        plt.tight_layout()

    else:
        plt.plot(predictions)
        plt.plot(test_y)

    plt.show()

if __name__ == '__main__':
    main()
