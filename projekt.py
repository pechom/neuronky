import numpy as np
import tensorflow as tf
import pylab as pl
from mpl_toolkits import mplot3d
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math
import pandas as pd
import keras as ks

LABEL_PATH = './omni2_24739.fmt.txt'
TRAINING_PATH = './c95,07.txt'
TEST_PATH = './c11.txt'

n_hours = 12
n_features = 5


def add_line_number(filename):
    infile = open(filename, 'r')
    lines = infile.readlines()
    infile.close()
    outtext = ['%d %s' % (i+1, line) for i, line in enumerate(lines)]
    outfile = open(filename, "w")
    outfile.writelines(str("".join(outtext)))
    outfile.close()


def read_signals(filename):
    with open(filename, 'r') as fp:
        data = fp.read().splitlines()
        data = map(lambda x: x.rstrip().lstrip().split(), data)
        data = [list(map(float, line)) for line in data]
        data = np.array(data, dtype=np.float32)
    return data


def repair_training_data(filename):
    data = read_signals(filename)
    for j in (4, 5, 6, 7, 8, 9):
        i = 0
        while i < len(data):
            if str(data[i, j]) == "999.9":
                start = i-1
                while str(data[i, j]) == "999.9":
                    i = i+1
                end = i
                fraction = end - start
                if (fraction <= 24) or ("11" in filename):  # v testovacom nemam ohranicenie
                    d = (data[end, j] - data[start, j])/fraction
                    for k in range(fraction - 1):
                        data[start+k+1, j] = data[start, j] + (k+1)*d
            i = i + 1
    np.savetxt(filename, data, fmt='%1.1f')


def find_storm_max(data):
    for j in range(6):
        max = np.amax(data[:, :, j])


def find_year_max(data):
    for j in range(6):
        max = np.amax(data[:, j])


def find_storm_min(data):
    for j in range(6):
        min = np.amin(data[:, :, j])


def find_year_min(data):
    for j in range(6):
        min = np.amin(data[:, j])

# maxima su [33.3, 22.5, 25.2, 81.5, 54]
# minima su [0.7, -21.4, -21.3, 0.2, -149]
# moje hodnoty pre interval su [34, 23, 26, 82, 150], dosiahnem väcsie rozdiely


def normalise(filename):  # len raz pocas projektu, pre zalohu dat
    normalizator = [0, 0, 0, 0, 34, 23, 26, 0, 82, 150]
    data = read_signals(filename)
    for j in (4, 5, 6, 8, 9):
        i = 0
        while i < len(data):
            data[i, j] = np.true_divide(data[i, j], normalizator[j])
            i = i+1
    np.savetxt(filename, data, fmt='%i ' '%i ' '%i ' '%i ' '%1.3f ' '%1.3f ' '%1.3f ' '%1.1f ' '%1.3f ' '%1.3f ')


def normalize(data):
    normalizator = [34, 23, 26, 82, 150]
    j = 0
    while j < 5:
        data[:, j] = data[:, j]/normalizator[j]
        j = j+1
    return data


def invert_dst_normalization(data):
    normalizator = 150
    data = data*normalizator
    return data


def normalize_storms(storms):
    normalizator = [34, 23, 26, 82, 150]
    for storm in storms:
        j = 0
        while j < 5:
            storm[:, j] = storm[:, j]/normalizator[j]
            j = j+1
    return storms


def get_training_output(storms):
    output = []
    for storm in storms:
        output.extend(storm[:, 4])
    return np.array(output)


def find_storms(data):
    storms = []
    dimension = data.shape[1] - 1
    i = 0
    while i < (len(data) - 2):
        if data[i, dimension] - data[i+2, dimension] >= 40:
            if i < 36:
                start = 0
            else:
                start = i-36
            if i > (len(data) - 1) - 109:
                end = len(data)  # posledna sa neberie
            else:
                end = i + 109
            storm = data[start:end, :]
            storms.append(storm)
            i = i + 108
        i = i + 1
    return np.array(storms)


def check_storms(storms):
    k = 0
    dimension = storms.shape[2]
    while k < len(storms):
        storm = storms[k]
        i = 0
        while i < (len(storm) - 24):
            j = 0
            while j < dimension:
                if str(set(storm[i:i+24, j])) == "{999.9}":  # nejde == ani is
                    storms = np.delete(storms, k, axis=0)
                    i = len(storm - 24)
                    break
                else:
                    j = j + 1
            if i != len(storm - 24):
                i = i + 1
        k = k + 1
    return storms


def graph_storms(data, storms):
    pl.figure()
    pl.plot(data[:, 0], data[:, 5])
    pl.show()
    plt.figure()
    ax = plt.axes(projection='3d')
    x = list(range(145))
    y = [0]*145
    for i in range(len(storms)):
        storm = storms[i]
        ax.scatter3D(x, [i]*145, storm[:, 5], cmap='Greens')
    plt.show()


def shift_array(arr, num):
    fill_value = np.nan
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result = arr
    return result


def series_to_supervised(input, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(input) is list else input.shape[1]
    df = pd.DataFrame(input)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def create_windows(input):
        input = pd.DataFrame(input)
        windows = series_to_supervised(input, 12, 1)  # pocet hodin
        windows = windows.values
        final_windows = windows[:, :n_hours*n_features]
        expected_output = windows[:, -1]
        final_windows = final_windows.reshape((windows.shape[0], n_hours, n_features))
        return final_windows, expected_output


def create_storms_windows(storms):
    final_windows = []
    final_output = []
    final_windows = np.array(final_windows)
    final_output = np.array(final_output)
    for storm in storms:
        storm = pd.DataFrame(storm)
        windows = series_to_supervised(storm, 12, 1)  # pocet hodin
        windows = windows.values
        expected_output = windows[:, -1]
        windows = windows[:, :n_hours*n_features]
        if len(final_windows) == 0:
            final_windows = np.array(windows)
            final_output = np.array(expected_output)
        else:
            final_windows = np.concatenate((final_windows, windows))
            final_output = np.concatenate((final_output, expected_output))
    final_windows = final_windows.reshape((final_windows.shape[0], n_hours, n_features))
    return final_windows, final_output


def create_windows_sequence_output(input):
    sequence = 3
    input = pd.DataFrame(input)
    windows = series_to_supervised(input, 12, sequence)  # pocet hodin
    windows = windows.values
    final_windows = windows[:, :n_hours * n_features]
    expected_output = windows[:, [-11, -6, -1]]  # -(2*5+1)...  hardcoded - zmen na argument!!!
    final_windows = final_windows.reshape((windows.shape[0], n_hours, n_features))
    return final_windows, expected_output


def create_storms_windows_sequence_output(storms):
    sequence = 3
    final_windows = []
    final_output = []
    final_windows = np.array(final_windows)
    final_output = np.array(final_output)
    for storm in storms:
        storm = pd.DataFrame(storm)
        windows = series_to_supervised(storm, 12, sequence)  # pocet hodin
        windows = windows.values
        expected_output = windows[:, [-11, -6, -1]]  # ...  hardcoded - zmen na argument!!!
        windows = windows[:, :n_hours*n_features]
        if len(final_windows) == 0:
            final_windows = np.array(windows)
            final_output = np.array(expected_output)
        else:
            final_windows = np.concatenate((final_windows, windows))
            final_output = np.concatenate((final_output, expected_output))
    final_windows = final_windows.reshape((final_windows.shape[0], n_hours, n_features))
    return final_windows, final_output


def find_drops(data):
    drops, i = 0, 0
    while i < (len(data) - 2):
        if data[i] - data[i + 2] >= 40:
            drops = drops + 1
        i = i+1
    return drops


def fit(model, training_input, training_output, test_input, test_output):
    history = model.fit(training_input, training_output, epochs=50, batch_size=50,
                        validation_data=(test_input, test_output), verbose=2, shuffle=False)
    pl.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()  # chyba pocas trenovania


def predict_evaluate(model, test_input, test_output, normal, accuracy, future):
    predicted = model.predict(test_input)  # postupne robi predpovede, nakonci su pre cely rok
    if future > 1:
        predicted = np.roll(predicted, -(future-1))  # predikujem 3 hodiny dopredu, preto aj rok potrebujem posunut
        predicted = predicted[:-(future-1)]
        test_output = test_output[:-(future-1)]
    predicted = predicted[:, 0]
    if normal == "normal":
        predicted = invert_dst_normalization(predicted)  # dst v povodnej skale
        test_output = invert_dst_normalization(test_output)
    rmse = math.sqrt(mean_squared_error(test_output, predicted))
    print('predict RMSE: %.3f' % rmse)  # chyba pocas predikcie
    pl.figure()
    plt.plot(predicted, label="predicted", color="orange")
    plt.plot(test_output, label="real values", color="navy", alpha=0.5)
    plt.legend()
    plt.show()
    i, start, end, real_storms, predicted_storms, same_storms = 0, 0, 0, 0, 0, 0
    while i < (len(predicted) - 2):
        if test_output[i] - test_output[i + 2] >= 40:
            real_storms = real_storms + 1
            # print("real", i)
        if predicted[i] - predicted[i + 2] >= 40:
            predicted_storms = predicted_storms+1
            # print("predict", i)
            start = i - accuracy
            end = i + accuracy
            if start < 0:
                diff = -start
                start = 0
                end = end+diff
            if end > len(predicted):
                end = len(predicted)-2
            while start <= end:
                if test_output[start] - test_output[start + 2] >= 40:
                    same_storms = same_storms + 1
                start = start+1
        i = i+1

    print("real storms ", real_storms)
    print("predicted storms ", predicted_storms)
    print("same storms ", same_storms)
    print("false storms ", abs(same_storms-predicted_storms))


def predict_evaluate_sequence(model, test_input, test_output, normal, accuracy, dump):
    predicted = model.predict(test_input, batch_size=50)  # postupne robi predpovede, nakonci su pre cely rok
    if normal == "normal":
        predicted = invert_dst_normalization(predicted)  # dst v povodnej skale
        test_output = invert_dst_normalization(test_output)
    predicted[:, 1] = np.roll(predicted[:, 1], -1)
    predicted[:, 2] = np.roll(predicted[:, 2], -2)
    rmse = math.sqrt(mean_squared_error(test_output[:, 0], predicted[:, 0]))
    print('predict RMSE for 1st hour: %.3f' % rmse)  # chyba pocas predikcie
    rmse = math.sqrt(mean_squared_error(test_output[:, 1], predicted[:, 1]))
    print('predict RMSE for 2nd hour: %.3f' % rmse)  # chyba pocas predikcie
    rmse = math.sqrt(mean_squared_error(test_output[:, 2], predicted[:, 2]))
    print('predict RMSE for 3rd hour: %.3f' % rmse)  # chyba pocas predikcie
    pl.figure()
    plt.plot(predicted[:, 0], label="predicted hour 1", color="orange")
    plt.plot(test_output[:, 0], label="real values", color="navy", alpha=0.5)
    plt.legend()
    pl.figure()
    plt.plot(predicted[:, 1], label="predicted hour 2", color="orange")
    plt.plot(test_output[:, 1], label="real values", color="navy", alpha=0.5)
    plt.legend()
    pl.figure()
    plt.plot(predicted[:, 2], label="predicted hour 3", color="orange")
    plt.plot(test_output[:, 2], label="real values", color="navy", alpha=0.5)
    plt.legend()
    plt.show()
    predicted_hour = 0
    while predicted_hour < test_output.shape[1]:
        i, start, end, real_storms, predicted_storms, same_storms = 0, 0, 0, 0, 0, 0
        while i < (len(test_output[:, predicted_hour]) - 2):
            if test_output[i, predicted_hour] - test_output[i + 2, predicted_hour] >= 40:
                real_storms = real_storms + 1
                # print("real", i)
            if predicted[i, predicted_hour] - predicted[i+2, predicted_hour] >= 40:
                predicted_storms = predicted_storms+1
                # print("predict", i)
                start = i - accuracy
                end = i + accuracy
                if start < 0:
                    diff = -start
                    start = 0
                    end = end+diff
                if end > len(test_output[:, predicted_hour]):
                    end = len(test_output[:, predicted_hour])-2
                while start <= end:
                    if test_output[start, predicted_hour] - test_output[start+2, predicted_hour] >= 40:
                        same_storms = same_storms+1
                    start = start+1
            i = i+1
        print("real storms ", real_storms)
        predicted_hour = predicted_hour + 1
        print("predicted storms hour", predicted_hour, ":", " ", predicted_storms)
        print("same storms hour", predicted_hour, ":", " ", same_storms)
        print("false storms ", abs(same_storms - predicted_storms))


def hours_of_storms(storms):
    for storm in storms:
        print(int(storm[0,0]))

# repair_training_data(YEAR_PATH)  # vola sa len raz pocas celeho projektu
data = read_signals(TRAINING_PATH)
test_data = read_signals(TEST_PATH)
# normalize(YEAR_PATH) # vola sa len raz pocas celeho projektu
data_hour = data[:, [0, 4, 5, 6, 8, 9]]  # pre tabulku burok
test_data_hour = test_data[:, [0, 4, 5, 6, 8, 9]]
data = data[:, [4, 5, 6, 8, 9]]
test_data = test_data[:, [4, 5, 6, 8, 9]]
# find_year_min(data) # vola sa len raz pocas celeho projektu

# storms_hour = find_storms(data_hour)
# test_storms_hour = find_storms(test_data_hour)
# hours_of_storms(storms_hour)  # hodiny burok
# graph_storms(data_hour, test_storms_hour)  # pre grafy, raz pocas celeho projektu

storms = find_storms(data)
test_storms = find_storms(test_data)
storms = check_storms(storms)
# find_storm_min(storms) # vola sa len raz pocas celeho projektu
normal_storms = normalize_storms(storms.copy())  # training inputs
normal_test_data = normalize(test_data.copy())

training_input, training_output = create_storms_windows(storms.copy())
training_input_sequence, training_output_sequence = create_storms_windows_sequence_output(storms.copy())
normal_training_input, normal_training_output = create_storms_windows(normal_storms.copy())
normal_training_input_sequence, normal_training_output_sequence = create_storms_windows_sequence_output(normal_storms.copy())
test_input, test_output = create_windows(test_data.copy())
test_input_sequence, test_output_sequence = create_windows_sequence_output(test_data.copy())
normal_test_input, normal_test_output = create_windows(normal_test_data.copy())
normal_test_input_sequence, normal_test_output_sequence = create_windows_sequence_output(normal_test_data.copy())
print("data spracovane")

# pri predikcii sekvencie maju mat vystupne vrstvy pocet neuronov: training_output_shape[1]

# shallow lstm
def model_1(training_input, training_output, test_input, test_output):
    model = ks.Sequential()
    model.add(ks.layers.LSTM(50, input_shape=(training_input.shape[1], training_input.shape[2])))
    model.add(ks.layers.Dense(training_output.shape[1], activation="linear"))
    # model.add(ks.layers.Dense(1, activation="linear"))
    # model.add(ks.layers.Dropout(0.2))
    model.compile(loss='mae', optimizer='adam')
    fit(model, training_input, training_output, test_input, test_output)
    predict_evaluate_sequence(model, test_input, test_output, "normal", 0, 1)  # presnost a hodiny dopredu
    # print(model.get_weights())

# 2 layers lstm
def model_2(training_input, training_output, test_input, test_output):
    model = ks.Sequential()
    model.add(ks.layers.LSTM(42, return_sequences=True, input_shape=(training_input.shape[1],
                             training_input.shape[2])))
    model.add(ks.layers.LSTM(42, activation="tanh"))
    # model.add(ks.layers.Dropout(0.1))
    model.add(ks.layers.Dense(1, activation="linear"))
    model.compile(loss='mae', optimizer='adam')
    fit(model, training_input, training_output, test_input, test_output)
    predict_evaluate(model, test_input, test_output, "normal", 0, 1)
    # model.get_weights()
    # print(model.get_weights())

# 2 layers rnn
def model_3(training_input, training_output, test_input, test_output):
    model = ks.Sequential()
    model.add(ks.layers.SimpleRNN(50, return_sequences=True, input_shape=(training_input.shape[1],
                            training_input.shape[2])))
    model.add(ks.layers.SimpleRNN(50, activation="tanh"))
    # model.add(ks.layers.Dropout(0.1))
    model.add(ks.layers.Dense(1, activation="linear"))
    model.compile(loss='mae', optimizer='adam')
    fit(model, training_input, training_output, test_input, test_output)
    predict_evaluate(model, test_input, test_output, "normal", 0, 1)


# 2 layers rnn s relu
def model_4(training_input, training_output, test_input, test_output):
    model = ks.Sequential()
    model.add(ks.layers.SimpleRNN(100, return_sequences=True, input_shape=(training_input.shape[1],
                            training_input.shape[2])))
    model.add(ks.layers.LeakyReLU(alpha=0.1))
    model.add(ks.layers.SimpleRNN(100))
    model.add(ks.layers.LeakyReLU(alpha=0.1))
    model.add(ks.layers.Dense(1, activation="linear"))
    model.compile(loss='mae', optimizer='adam')
    fit(model, training_input, training_output, test_input, test_output)
    predict_evaluate(model, test_input, test_output, "not", 0, 1)


#deep lstm
def model_5(training_input, training_output, test_input, test_output):
    model = ks.Sequential()
    model.add(ks.layers.LSTM(90, return_sequences=True, input_shape=(training_input.shape[1],
                            training_input.shape[2])))
    # model.add(ks.layers.Dropout(0.1))
    model.add(ks.layers.LSTM(80, return_sequences=True, activation="tanh"))
    # model.add(ks.layers.Dropout(0.1))
    model.add(ks.layers.LSTM(70, return_sequences=True, activation="tanh"))
    # model.add(ks.layers.Dropout(0.1))
    model.add(ks.layers.LSTM(60, return_sequences=True, activation="tanh"))
    # model.add(ks.layers.Dropout(0.1))
    model.add(ks.layers.LSTM(50, activation="tanh"))
    # model.add(ks.layers.Dropout(0.1))
    model.add(ks.layers.Dense(1, activation="linear"))
    model.compile(loss='mae', optimizer='adam')
    fit(model, training_input, training_output, test_input, test_output)
    predict_evaluate(model, test_input, test_output, "normal", 0, 1)


# deep rnn with relu
def model_6(training_input, training_output, test_input, test_output):
    model = ks.Sequential()
    model.add(ks.layers.SimpleRNN(180, return_sequences=True, input_shape=(training_input.shape[1],
                            training_input.shape[2])))
    model.add(ks.layers.LeakyReLU(alpha=0.1))
    # model.add(ks.layers.Dropout(0.1))
    model.add(ks.layers.SimpleRNN(160, return_sequences=True))
    model.add(ks.layers.LeakyReLU(alpha=0.1))
    # model.add(ks.layers.Dropout(0.1))
    model.add(ks.layers.SimpleRNN(140, return_sequences=True))
    model.add(ks.layers.LeakyReLU(alpha=0.1))
    # model.add(ks.layers.Dropout(0.1))
    model.add(ks.layers.SimpleRNN(120, return_sequences=True))
    model.add(ks.layers.LeakyReLU(alpha=0.1))
    # model.add(ks.layers.Dropout(0.1))
    model.add(ks.layers.SimpleRNN(100))
    # model.add(ks.layers.Dropout(0.1))
    model.add(ks.layers.LeakyReLU(alpha=0.1))
    model.add(ks.layers.Dense(1, activation="linear"))
    model.compile(loss='mae', optimizer='adam')
    fit(model, training_input, training_output, test_input, test_output)
    predict_evaluate(model, test_input, test_output, "not", 0, 1)


# deep rnn
def model_7(training_input, training_output, test_input, test_output):
    model = ks.Sequential()
    model.add(ks.layers.SimpleRNN(90, return_sequences=True, input_shape=(training_input.shape[1],
                            training_input.shape[2])))
    # model.add(ks.layers.BatchNormalization())
    # model.add(ks.layers.Activation('tanh'))
    model.add(ks.layers.Dropout(0.3))
    model.add(ks.layers.SimpleRNN(80, return_sequences=True))
    # model.add(ks.layers.BatchNormalization())
    # model.add(ks.layers.Activation('tanh'))
    model.add(ks.layers.Dropout(0.3))
    model.add(ks.layers.SimpleRNN(70, return_sequences=True))
    # model.add(ks.layers.BatchNormalization())
    # model.add(ks.layers.Activation('tanh'))
    model.add(ks.layers.Dropout(0.3))
    model.add(ks.layers.SimpleRNN(60, return_sequences=True))
    # model.add(ks.layers.BatchNormalization())
    # model.add(ks.layers.Activation('tanh'))
    # model.add(ks.layers.Dropout(0.3))
    model.add(ks.layers.SimpleRNN(50))
    # model.add(ks.layers.BatchNormalization())
    # model.add(ks.layers.Activation('tanh'))
    model.add(ks.layers.Dropout(0.3))
    model.add(ks.layers.Dense(1, activation="linear"))
    model.compile(loss='mae', optimizer='adam')
    fit(model, training_input, training_output, test_input, test_output)
    predict_evaluate(model, test_input, test_output, "normal", 0, 1)


# deep dense with lstm input
def model_8(training_input, training_output, test_input, test_output):
    model = ks.Sequential()
    model.add(ks.layers.LSTM(100, input_shape=(training_input.shape[1], training_input.shape[2])))
    model.add(ks.layers.Dense(80, activation="tanh"))
    # model.add(ks.layers.BatchNormalization())
    # model.add(ks.layers.Activation('tanh'))
    # model.add(ks.layers.Dropout(0.1))
    model.add(ks.layers.Dense(70, activation="tanh"))
    # model.add(ks.layers.BatchNormalization())
    # model.add(ks.layers.Activation('tanh'))
    # model.add(ks.layers.Dropout(0.1))
    model.add(ks.layers.Dense(60, activation="tanh"))
    # model.add(ks.layers.BatchNormalization())
    # model.add(ks.layers.Activation('tanh'))
    # model.add(ks.layers.Dropout(0.1))
    model.add(ks.layers.Dense(50, activation="tanh"))
    # model.add(ks.layers.BatchNormalization())
    # model.add(ks.layers.Activation('tanh'))
    # model.add(ks.layers.Dropout(0.1))
    model.add(ks.layers.Dense(1, activation="linear"))
    model.compile(loss='mae', optimizer='adam')
    fit(model, training_input, training_output, test_input, test_output)
    predict_evaluate(model, test_input, test_output, "normal", 0, 1)


# deep dense with relu and lstm input
def model_9(training_input, training_output, test_input, test_output):
    model = ks.Sequential()
    model.add(ks.layers.LSTM(180, input_shape=(training_input.shape[1], training_input.shape[2])))
    model.add(ks.layers.Dropout(0.2))
    model.add(ks.layers.LeakyReLU(alpha=0.1))
    model.add(ks.layers.Dense(160))
    model.add(ks.layers.Dropout(0.2))
    model.add(ks.layers.LeakyReLU(alpha=0.1))
    model.add(ks.layers.Dense(140))
    model.add(ks.layers.Dropout(0.2))
    model.add(ks.layers.LeakyReLU(alpha=0.1))
    model.add(ks.layers.Dense(120))
    model.add(ks.layers.Dropout(0.2))
    model.add(ks.layers.LeakyReLU(alpha=0.1))
    model.add(ks.layers.Dense(100))
    model.add(ks.layers.Dropout(0.2))
    model.add(ks.layers.LeakyReLU(alpha=0.1))
    model.add(ks.layers.Dense(1))
    model.compile(loss='mae', optimizer='adam')
    fit(model, training_input, training_output, test_input, test_output)
    predict_evaluate(model, test_input, test_output, "not", 0, 1)


# siet ktora sa pouziva na sequence to sequence, dokaze fungovat aj ako enkoder-dekoder (aj druha vracia sequence)
def model_10(training_input, training_output, test_input, test_output):
    model = ks.Sequential()
    model.add(ks.layers.LSTM(50, return_sequences=True, input_shape=(training_input.shape[1], training_input.shape[2])))
    # model.add(ks.layers.BatchNormalization())
    # model.add(ks.layers.Activation('tanh'))
    model.add(ks.layers.LSTM(1))  # pocet features
    model.compile(loss='mae', optimizer='adam')
    fit(model, training_input, training_output, test_input, test_output)
    predict_evaluate(model, test_input, test_output, "normal", 0, 1)


# dense 3 layers
def model_11(training_input, training_output, test_input, test_output):
    training_input = training_input.reshape(training_input.shape[0], training_input.shape[1] * training_input.shape[2])
    test_input = test_input.reshape(test_input.shape[0], test_input.shape[1] * test_input.shape[2])
    model = ks.Sequential()
    model.add(ks.layers.Dense(50))
    # model.add(ks.layers.Dropout(0.2))
    model.add(ks.layers.Dense(50))
    # model.add(ks.layers.Dropout(0.2))
    model.add(ks.layers.Dense(1, activation="linear"))
    model.compile(loss='mae', optimizer='adam')
    fit(model, training_input, training_output, test_input, test_output)
    predict_evaluate(model, test_input, test_output, "normal", 0, 1)


# deep dense with relu
def model_12(training_input, training_output, test_input, test_output):
    training_input = training_input.reshape(training_input.shape[0], training_input.shape[1] * training_input.shape[2])
    test_input = test_input.reshape(test_input.shape[0], test_input.shape[1] * test_input.shape[2])
    model = ks.Sequential()
    model.add(ks.layers.Dense(180))
    model.add(ks.layers.Dropout(0.2))
    model.add(ks.layers.LeakyReLU(alpha=0.1))
    model.add(ks.layers.Dense(160))
    model.add(ks.layers.Dropout(0.2))
    model.add(ks.layers.LeakyReLU(alpha=0.1))
    model.add(ks.layers.Dense(140))
    model.add(ks.layers.Dropout(0.2))
    model.add(ks.layers.LeakyReLU(alpha=0.1))
    model.add(ks.layers.Dense(120))
    model.add(ks.layers.Dropout(0.2))
    model.add(ks.layers.LeakyReLU(alpha=0.1))
    model.add(ks.layers.Dense(100))
    model.add(ks.layers.Dropout(0.2))
    model.add(ks.layers.LeakyReLU(alpha=0.1))
    # model.add(ks.layers.Dense(1))
    model.add(ks.layers.Dense(training_output.shape[1]))
    model.compile(loss='mae', optimizer='adam')
    fit(model, training_input, training_output, test_input, test_output)
    predict_evaluate_sequence(model, test_input, test_output, "not", 0, 1)


def lstm_autoencoder(layers, training_input, test_input):  #  pre sekvencie
    weighted_layer = []
    for i in range(len(layers)):
        shape = training_input.shape
        encoded = ks.layers.LSTM(layers[i], input_shape=(shape[1], shape[2]))  # namiesto middle staci dat return_sequence=True
        middle = ks.layers.RepeatVector(12)  # prvy nevracia sekvenciu, ale namiesto toho opakuje posledny vystup - lepsie vysledky
        decoded = ks.layers.LSTM(shape[2], return_sequences=True)  # pre kazdu feature jeden neuron
        sequence_autoencoder = ks.Sequential()
        sequence_autoencoder.add(encoded)
        sequence_autoencoder.add(middle)
        sequence_autoencoder.add(decoded)
        sequence_autoencoder.compile(loss='mae', optimizer='adam')
        sequence_autoencoder.fit(training_input, training_input, epochs=3, batch_size=50, verbose=2, shuffle=False)
        predicted = sequence_autoencoder.predict(test_input)
        training_input = predicted
        weighted_layer.append(encoded.get_weights())
    return weighted_layer


# podla lstm autoencoder
def pretraining_lstm(layers, training_input):
    pretrained = []
    weights = []
    for i in range(len(layers)-1):
        shape = training_input.shape
        encoded = ks.layers.LSTM(layers[i], return_sequences=True, input_shape=(shape[1], shape[2]))
        middle = ks.layers.LSTM(layers[i+1], return_sequences=True)
        decoded = ks.layers.LSTM(shape[2], return_sequences=True)
        sequence_autoencoder = ks.Sequential()
        sequence_autoencoder.add(encoded)
        sequence_autoencoder.add(middle)
        sequence_autoencoder.add(decoded)
        sequence_autoencoder.compile(loss='mae', optimizer='adam')
        sequence_autoencoder.fit(training_input, training_input, epochs=3, batch_size=50, verbose=2, shuffle=False)
        predicted = sequence_autoencoder.predict(training_input)
        training_input = predicted
        pretrained.append(encoded)
        layers.append(encoded)
        weights.append(encoded.get_weights())
    return weights


# pretrained deep lstm - nejde
def model_pretrained_1(training_input, training_output, test_input, test_output):
    # weights = lstm_autoencoder([100, 80, 60, 40], training_input)
    model = ks.Sequential()
    layer1 = ks.layers.LSTM(100, return_sequences=True, input_shape=(training_input.shape[1], training_input.shape[2]))
    layer2 = ks.layers.LSTM(80, return_sequences=True, input_shape=(training_input.shape[1], training_input.shape[2]))
    layer3 = ks.layers.LSTM(60, return_sequences=True, input_shape=(training_input.shape[1], training_input.shape[2]))
    layer4 = ks.layers.LSTM(40, return_sequences=True, input_shape=(training_input.shape[1], training_input.shape[2]))
    layer5 = ks.layers.LSTM(20)
    layers = pretraining_lstm([layer1, layer2, layer3, layer4], training_input)
    model.add(layers[0])
    model.add(layers[1])
    model.add(layers[2])
    model.add(layers[3])
    model.add(layer5)
    model.add(ks.layers.Dense(1, activation="linear"))
    model.compile(loss='mae', optimizer='adam')
    fit(model, training_input, training_output, test_input, test_output)
    predict_evaluate(model, test_input, test_output, "normal", 1, 1)

# pretrained deep lstm - nejde
def model_pretrained_2(training_input, training_output, test_input, test_output):
    model = ks.Sequential()
    weights = pretraining_lstm([100, 80, 60, 40], training_input)
    layer0 = ks.layers.LSTM(100, return_sequences=True, input_shape=(training_input.shape[1], training_input.shape[2]), activation="tanh", weights=weights[1])
    layer1 = ks.layers.LSTM(80, return_sequences=True, activation="tanh", weights=weights[1])
    layer2 = ks.layers.LSTM(60, return_sequences=True, activation="tanh", weights=weights[2])
    layer3 = ks.layers.LSTM(40, return_sequences=True,  activation="tanh")
    layer4 = ks.layers.LSTM(1, activation="tanh")
    model.add(layer0)
    model.add(layer1)
    model.add(layer2)
    model.add(layer3)
    model.add(layer4)
    model.compile(loss='mae', optimizer='adam')
    fit(model, training_input, training_output, test_input, test_output)
    predict_evaluate(model, test_input, test_output, "normal", 1, 1)


# model_1(normal_training_input, normal_training_output, normal_test_input, normal_test_output)
model_1(normal_training_input_sequence, normal_training_output_sequence, normal_test_input_sequence, normal_test_output_sequence)  # vstupy pre sekvencie
# model_12(training_input, training_output, test_input, test_output)  # vstupy pre relu
# model_12(training_input_sequence, training_output_sequence, test_input_sequence, test_output_sequence)  # vstupy pre sekvencie