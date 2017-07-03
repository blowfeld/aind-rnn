import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series,window_size):
    pairs = len(series) - window_size
    # containers for input/output pairs
    X = [ series[i:i + window_size] for i in range(0, pairs) ]
    y = series[window_size:]

    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(step_size, window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))


    # build model using keras documentation recommended optimizer initialization
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    model.summary()


### TODO: list all unique characters in the text and remove any non-english ones
def clean_text(text):
    import re

    # find all unique characters in the text
    char_orig = set(text)

    # remove as many non-english characters and character sequences as you can
    text = re.compile('[àâ]').sub('a', text)
    text = re.compile('[èé]').sub('e', text)
    text = re.compile('[^a-zA-Z!,.:;?]').sub(' ', text)

    # shorten any extra dead space created above
    text = re.compile('\s+').sub(' ', text)

    # find all unique characters in the clean text
    char_clean = set(text)

    # print("Kept: ")
    # print(sorted(char_clean))
    # print("Removed: ")
    # print(sorted(char_orig - char_clean))


### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs
    inputs = [ text[i - window_size:i] for i in range(window_size, len(text), step_size) ]
    outputs = list(text[window_size::step_size])

    return inputs,outputs

### TODO Create a simple RNN model using keras to perform multiclass classification
def build_part2_RNN(window_size, chars):
    from keras.models import Sequential
    from keras.layers import Dense, Activation, LSTM
    from keras.optimizers import RMSprop

    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation("softmax"))


    # initialize optimizer
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile model --> make sure initialized optimizer and callbacks - as defined above - are used
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    model.summary()

# TODO: choose an input sequence and use the prediction function in the previous Python cell to predict 100 characters following it
# get an appropriately sized chunk of characters from the text
def choose_start_index():
    start_inds = [2501]

    # save output
    f = open('text_gen_output/RNN_large_textdata_output.txt', 'w')  # create an output file to write too

    # load weights
    model.load_weights('model_weights/best_RNN_large_textdata_weights.hdf5.bkp')
    for s in start_inds:
        start_index = s
        input_chars = text[start_index: start_index + window_size]

        # use the prediction function
        predict_input = predict_next_chars(model,input_chars,num_to_predict = 100)

        # print out input characters
        line = '-------------------' + '\n'
        print(line)
        f.write(line)

        input_line = 'input chars = ' + '\n' +  input_chars + '"' + '\n'
        print(input_line)
        f.write(input_line)

        # print out predicted characters
        predict_line = 'predicted chars = ' + '\n' +  predict_input + '"' + '\n'
        print(predict_line)
        f.write(predict_line)
    f.close()
