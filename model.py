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

    x = Dense(50, activation='linear')(x)
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

def prediction_real(trained_model, block_data, length):
    data_window = [block_data[0][:1], block_data[1][:1]]
    result = [[],[]]
    window_size = len(block_data[0][0])
    print('Creating predictions')

    for i in range(length):
        predictions = trained_model.predict(data_window)
        # First result list, first output, first batch, first value
        result[0].append(predictions[0][0][0])
        result[1].append(predictions[1][0][0])

        # Adding new last and removing first
        data_window[0] = [data_window[0][0][1:]]
        data_window[1] = [data_window[1][0][1:]]

        data_window[0] = np.insert(data_window[0], window_size - 1, predictions[0], axis=1)
        data_window[1] = np.insert(data_window[1], window_size - 1, predictions[1], axis=1)
 
    return result


if __name__ == '__main__':
    build_lstm_model()
    # test()




