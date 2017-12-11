import numpy
import matplotlib.pyplot as plt
import pandas
import math

# Keras is used to model the network
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import TensorBoard

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

from model import build_lstm_model, predict_range, test # TODO: Remove test
from data_processing import build_stock_input, merge_data

def main():
    # Change to get new results
    numpy.random.seed(123456)

    # Reading stock data
    stock_1 = pandas.read_csv('data/AAPL.csv', usecols=[2], engine='python').values.astype('float32')
    stock_2 = pandas.read_csv('data/GOLD.csv', usecols=[2], engine='python').values.astype('float32')

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
    
    loadModelFromFile = True
    if not loadModelFromFile:

        model = build_lstm_model()

        # Training model
        tensorboard = TensorBoard(log_dir="logs/{}".format("lstm_test"))
        model.fit([train_x1, train_x2], [train_y1, train_y2], verbose=1, epochs = 10, callbacks=[tensorboard])
        
        model.save('save/model.h5')
        print("Result saved!")

    else:
        model = load_model('save/model.h5')

    # print(test_x1)

    # Creating predictions
    # print(test_x1)
    predictions = model.predict([test_x1, test_x2])[0]
    # predictions = numpy.array(test(model, [test_x1, test_x2], 100))
    # print(predictions)
    # predictions = predictions[:, 0]

    # print(len(test_x))

    subPlots = True
    if subPlots:
        fig, ax = plt.subplots(nrows=2)

        ax[0].plot(predictions)
        ax[1].plot(test_y)


    else:
        plt.plot(predictions)
        plt.plot(test_y)

    plt.show()

if __name__ == '__main__':
    main()
