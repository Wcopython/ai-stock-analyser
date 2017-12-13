import numpy as np
from sklearn.preprocessing import MinMaxScaler

# from keras.utils import normalize

def build_stock_input(
    data_list,
    input_size,
    test_ratio = 0.25,
    step_size = 25
    ):

    # Normalizing data
    # TODO: Fix this! MinMaxScaler only works on 2D vectors
    # scaler = MinMaxScaler(feature_range=(1)) 
    # data_list = scaler.fit_transform(data_list)
    # normalize(np.array(data_list))

    # Extracting training data
    training_size = int(len(data_list) * (1.0 - test_ratio))

    training_data = data_list[: training_size]
    testing_data = data_list[training_size:]

    # Dividing content in equal batch sizes (blocks) 
    # Elements will be repeated!

    # Training data
    training_blocks = []
    index = 0
    while index + input_size < len(training_data):
        training_blocks.append(training_data[index: index + input_size + 1])
        index += step_size

    training_blocks = np.array(training_blocks)

    # Shuffling training data
    # It should be the same shuffle every time you run this function if the input size is the same!
    np.random.seed(123456)
    np.random.shuffle(training_blocks)

    # Testing (this is a continuous region)
    testing_blocks = []
    index = 0
    while index + input_size < len(testing_data):
        testing_blocks.append(testing_data[index: index + input_size + 1])
        index += 1

    testing_blocks = np.array(testing_blocks)

    # Generate output
    training_x = training_blocks[:, :-1]
    training_y = training_blocks[:, -1]
    testing_x = testing_blocks[:, :-1]
    testing_y = testing_blocks[:, -1]

    training_x = np.reshape(training_x, (training_x.shape[0], training_x.shape[1], 1))
    testing_x = np.reshape(testing_x, (testing_x.shape[0], testing_x.shape[1], 1)) 

    return [training_x, training_y, testing_x, testing_y]


## Deprecated!
def merge_data(data_sources):
    len0 = len(data_sources[0])
    for i in range(len(data_sources)):
        if len0 != len(data_sources[i]):
            raise ValueError('Arrays of different lengths')

    result = []

    for i in range(len(data_sources[0])):
        for j in range(len(data_sources)):
            if (j == 0):
                # print("new array: ", i)
                result.append([data_sources[0][i]])
            else:
                result[i].append(data_sources[j][i])
    
    return result


def main():
    data1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    data2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    data3 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    # data2 = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
    
    data1 = build_stock_input(data1, 10, step_size = 5)
    data2 = build_stock_input(data2, 10, step_size = 5)
    data3 = build_stock_input(data3, 10, step_size = 5)

    mergedData = merge_data([data1[0], data2[0], data3[0]])

    for line in mergedData:
        print(line)

if __name__ == '__main__':
    main()