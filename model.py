from keras.layers import Input, concatenate
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Model, Sequential

import numpy as np

def build_lstm_model():
    # Defining the input vectors
    input_a = Input(
        shape=(100,1)
    )

    input_b = Input(
        shape=(100,1)
    )

    a_x = LSTM(
        input_shape=(100, 1),
        output_dim=100,
        return_sequences=True
    )(input_a)

    b_x = LSTM(
        input_shape=(100, 1),
        output_dim=100,
        return_sequences=True
    )(input_b)

    x = concatenate([a_x, b_x], axis=-1)

    x = Dropout(0.2)(x)
    
    # Another layer with memory
    x = LSTM(100)(x)
    x = Dropout(0.2)(x)

    # Linear??
    output_a = Dense(1, activation='linear')(x)
    output_b = Dense(1, activation='linear')(x)

    model = Model(
        inputs=[input_a, input_b],
        outputs=[output_a, output_b]
    )

    # Look in to other functions here...
    model.compile(loss='mse', optimizer='rmsprop')

    print("Compiled model! ")

    return model

def predict_range(trained_model, block_data, length):
    data_window = []
    result = []
    for i in range(len(block_data)):
        # Get the first table from each src
        data_window.append(np.array(block_data[i][0:2]))
    
    # # data_window = np.array(data_window)
    print(data_window[0])

    for i in range(length):
        predictions = trained_model.predict(data_window, batch_size=1)
        result.append(predictions)

        # Add this to datawindow
        for j in range(len(data_window)):
            data_window[j] = np.append(data_window[j], predictions[j])[1:]
        

    return result

def test(trained_model, block_data, length):
    data_window = [block_data[0][:], block_data[1][:]]
    result = []
    
    # # data_window = np.array(data_window)
    # print(data_window)

    return trained_model.predict(data_window)

    # for i in range(length):
    #     predictions = trained_model.predict(data_window, batch_size=1)
    #     result.append(predictions)

    #     # Add this to datawindow
    #     for j in range(len(data_window)):
    #         data_window[j] = np.append(data_window[j], predictions[j])[1:]
        

    # return result


if __name__ == '__main__':
    build_lstm_model()
    # test()




