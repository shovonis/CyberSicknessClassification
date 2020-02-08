from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import ConvLSTM2D
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from numpy import mean
from numpy import std
from pandas import read_csv
from scipy import interp
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold


def load_data(file_name):
    # Test the DL classification
    dataset = read_csv(file_name, header=0, index_col=0)
    values = dataset.values
    values = values.astype('float32')
    return values


def load_train_test(file_name):
    # Test the DL classification
    dataset = read_csv(file_name, header=0, index_col=0)
    values = dataset.values
    values = values.astype('float32')

    train, test = train_test_split(values, test_size=0.45)

    return train, test


# Normalize the data
def normalize_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)
    return scaled


def standardize_data(data):
    scaler = StandardScaler()
    standardized = scaler.fit_transform(data)
    return standardized


def one_hot_encoding(data):
    # one hot encoding
    data = data - 1
    # To categorical conversion for the y.
    data = to_categorical(data)

    return data


def prepare_k_fold_data(data_file, time_step=120, features=9):
    data = load_data(data_file)
    total_observation = time_step * features

    data_x, data_y = data[:, :total_observation], data[:, -(features + 1)]

    # Standardize X input
    data_x = normalize_data(data_x)

    # reshape input to be 3D [samples, time_steps, features]
    data_x = data_x.reshape((data_x.shape[0], time_step, features))
    data_y = data_y.reshape((data_y.shape[0], 1))

    # Print the data shape
    print("Data: ")
    print("X:", data_x.shape, "Y:", data_y.shape)

    return data_x, data_y


def prepare_data(data_file, time_step=120, features=9):
    train_data, test_data = load_train_test(data_file)

    # Distribution of test and train data
    total_observation = time_step * features
    train_x, train_y = train_data[:, :total_observation], train_data[:, -(features + 1)]
    test_x, test_y = test_data[:, :total_observation], test_data[:, -(features + 1)]

    # Standardize X input
    train_x = normalize_data(train_x)
    test_x = normalize_data(test_x)

    # reshape input to be 3D [samples, time_steps, features]
    train_x = train_x.reshape((train_x.shape[0], time_step, features))
    test_x = test_x.reshape((test_x.shape[0], time_step, features))
    train_y = train_y.reshape((train_y.shape[0], 1))
    test_y = test_y.reshape((test_y.shape[0], 1))

    # One hot encoding of the y
    train_y = train_y - 1
    test_y = test_y - 1

    # To categorical conversion for the y.
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)

    # Print the data shape
    print("Training Data: ")
    print("X:", train_x, "Y:", train_y.shape)

    print("Test Data: ")
    print("X: ", test_x, "Y:", test_y.shape)

    return train_x, train_y, test_x, test_y


def roc_auc_for_each_class(n_classes, y_test, y_score, name):
    line_width = 2
    severity_class = ["LS", "MS", "AS"]

    false_positive_rate = dict()
    true_positive_rate = dict()
    roc_auc = dict()

    # Calculate for each individual class
    for i in range(n_classes):
        false_positive_rate[i], true_positive_rate[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(false_positive_rate[i], true_positive_rate[i])

    # Compute and print class summary:
    y_pred_bool = np.argmax(y_score, axis=1)
    y_test_bool = np.argmax(y_test, axis=1)
    print(classification_report(y_test_bool, y_pred_bool, target_names=severity_class))

    # Compute micro-average ROC curve and ROC area
    false_positive_rate["micro"], true_positive_rate["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(false_positive_rate["micro"], true_positive_rate["micro"])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([false_positive_rate[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, false_positive_rate[i], true_positive_rate[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    false_positive_rate["macro"] = all_fpr
    true_positive_rate["macro"] = mean_tpr
    roc_auc["macro"] = auc(false_positive_rate["macro"], true_positive_rate["macro"])

    # Plot all ROC curves
    plt.figure(1)
    # plt.plot(false_positive_rate["micro"], true_positive_rate["micro"],
    #          label='micro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["micro"]),
    #          color='deeppink', linestyle=':', linewidth=4)
    #
    # plt.plot(false_positive_rate["macro"], true_positive_rate["macro"],
    #          label='macro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["macro"]),
    #          color='navy', linestyle=':', linewidth=4)

    colors = cycle(['green', 'darkorange', 'red'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(false_positive_rate[i], true_positive_rate[i], color=color, lw=line_width,
                 label='ROC curve of class {0} (AUC = {1:0.2f})'
                       ''.format(severity_class[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=line_width)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC for severity classification')
    plt.legend(loc="lower right")
    plt.savefig('results/roc_severity' + name + ".pdf")
    plt.show()

    # Zoom in view of the upper left corner.
    plt.figure(2)
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)
    # plt.plot(false_positive_rate["micro"], true_positive_rate["micro"],
    #          label='micro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["micro"]),
    #          color='deeppink', linestyle=':', linewidth=4)
    #
    # plt.plot(false_positive_rate["macro"], true_positive_rate["macro"],
    #          label='macro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["macro"]),
    #          color='navy', linestyle=':', linewidth=4)

    colors = cycle(['green', 'darkorange', 'red'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(false_positive_rate[i], true_positive_rate[i], color=color, lw=line_width,
                 label='ROC curve of class {0} (AUC = {1:0.2f})'
                       ''.format(severity_class[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=line_width)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC for severity classification')
    plt.legend(loc="lower right")
    plt.savefig('results/roc_severity_zoomed' + name + ".pdf")
    plt.show()


def evaluate_conv_2d_lstm_model(train_x, train_y, test_x, test_y):
    # define model
    verbose, epochs, batch_size = 1, 100, 512
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

    # reshape into subsequences (samples, time steps, rows, cols, channels)
    n_steps, n_length = 4, 30
    train_x = train_x.reshape((train_x.shape[0], n_steps, 1, n_length, n_features))
    test_x = test_x.reshape((test_x.shape[0], n_steps, 1, n_length, n_features))

    # define model
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(1, 4), activation='relu',
                         input_shape=(n_steps, 1, n_length, n_features), recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.005), metrics=['accuracy'])

    # fit network
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose,
                        validation_data=(test_x, test_y))

    print("Model Summary: ")
    print_model_summary(model, "conv2d")

    # evaluate model
    loss, accuracy = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=0)
    y_predict = model.predict(test_x)

    return loss, accuracy, history, y_predict


def ecg_cnn_lstm(train_x, train_y, test_x, test_y):
    # define model
    verbose, epochs, batch_size = 1, 100, 512
    time_steps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # define model
    model = Sequential()
    model.add(Conv1D(filters=24, kernel_size=4, activation='relu', input_shape=(time_steps, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=40, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.4))
    model.add(LSTM(32, recurrent_dropout=0.20, return_sequences=True))
    model.add(LSTM(32, recurrent_dropout=0.20, return_sequences=True))
    model.add(LSTM(16, recurrent_dropout=0.20))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

    # fit network
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose,
                        validation_data=(test_x, test_y))

    print("Model Summary: ")
    print_model_summary(model, "ecg")

    # evaluate model
    loss, accuracy = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=0)
    y_predict = model.predict(test_x)

    return loss, accuracy, history, y_predict


def print_model_summary(model, name):
    from contextlib import redirect_stdout
    with open(name + '.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()
            for layer in model.layers:
                print(layer.name, " : ", layer.input_shape)


def stacked_cnn_lstm(train_x, train_y, test_x, test_y):
    verbose, epochs, batch_size = 1, 100, 512
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

    # reshape data into time steps of sub-sequences
    n_steps, n_length = 4, 30
    train_x = train_x.reshape((train_x.shape[0], n_steps, n_length, n_features))
    test_x = test_x.reshape((test_x.shape[0], n_steps, n_length, n_features))

    # define model
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=4, activation='relu'),
                              input_shape=(None, n_length, n_features)))
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=4, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(64, recurrent_dropout=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

    # print("Model Summary: ")
    # print_model_summary(model, "cnn-lstm")

    # fit network
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose,
                        validation_data=(test_x, test_y))

    # evaluate model
    loss, accuracy = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=1)
    y_predict = model.predict(test_x)

    return loss, accuracy, history, y_predict


# fit and evaluate a simple LSTM model
def lstm_model(train_x, train_y, test_x, test_y):
    # Define the hyper parameter
    verbose, epochs, batch_size = 1, 100, 512
    time_steps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

    # Define the model
    model = Sequential()
    model.add(LSTM(128, input_shape=(time_steps, n_features)))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

    # fit network
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose,
                        validation_data=(test_x, test_y))

    # print("Model Summary: ")
    # print_model_summary(model, "lstm")

    # evaluate model
    loss, accuracy = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=0)
    y_predict = model.predict(test_x)

    return loss, accuracy, history, y_predict


def plot_model(history_sum):
    line_labels = ["Val. LSTM", "Val. CAD CNN-LSTM", "Val. Proposed CNN-LSTM", "Train LSTM",
                   "Train CAD CNN-LSTM", "Train Proposed CNN-LSTM"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 11))
    ax1.set_title('Loss on each epoch on validation data')
    l1 = ax1.plot(history_sum['lstm'].history['val_loss'], 'k-*')[0]
    l4 = ax1.plot(history_sum['lstm'].history['loss'], 'b-*')[0]

    l2 = ax1.plot(history_sum['ecg'].history['val_loss'], 'g-*')[0]
    l5 = ax1.plot(history_sum['ecg'].history['loss'], 'm-*')[0]

    l3 = ax1.plot(history_sum['cnn_lstm'].history['val_loss'], 'r-*')[0]
    l6 = ax1.plot(history_sum['cnn_lstm'].history['loss'], 'c-*')[0]
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    # plot accuracy during training
    ax2.set_title('Accuracy on each epoch on validation data')
    ax2.plot(history_sum['lstm'].history['val_acc'], 'k-*')
    ax2.plot(history_sum['lstm'].history['acc'], 'b-*')

    ax2.plot(history_sum['ecg'].history['val_acc'], 'g-*')
    ax2.plot(history_sum['ecg'].history['acc'], 'm-*')

    ax2.plot(history_sum['cnn_lstm'].history['val_acc'], 'r-*')
    ax2.plot(history_sum['cnn_lstm'].history['acc'], 'c-*')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")

    fig.legend([l1, l2, l3, l4, l5, l6],  # The line objects
               labels=line_labels,  # The labels for each line
               loc='lower center',  # Position of legend
               ncol=3,
               borderaxespad=3,  # Small spacing around legend box
               )
    # plt.legend((l1, l2), ('Line 1', 'Line 2'), 'upper left')
    plt.savefig("results/loss_acc_test_modelsV2.png")
    plt.show()


# summarize results
def summarize_results(acc, loss):
    print("Full acc list:", acc)
    print("Full loss list:", loss)

    # Summarize Acc
    m, s = mean(acc), std(acc)
    print('Mean Accuracy: %.3f%% (+/-%.3f)' % (m, s))

    # Summarize loss
    m, s = mean(loss), std(loss)
    print('Mean Loss: %.3f%% (+/-%.3f)' % (m, s))


# run an experiment
def run_experiment(repeats=10):
    # load data
    data_file = 'data/dl_data.csv'

    # ## repeat experiment
    list_acc = list()
    list_loss = list()
    history_sum = dict()
    for r in range(repeats):
        train_x, train_y, test_x, test_y = prepare_data(data_file, time_step=120, features=13)
        # print("Running LSTM")
        # loss, acc, history, y_predict = lstm_model(train_x, train_y, test_x, test_y)
        # history_sum['lstm'] = history

        # loss, acc, history, y_predict = evaluate_conv_2d_lstm_model(train_x, train_y, test_x, test_y)
        # history_sum['conv_2d'] = history
        #
        # print("Running ECG")
        # loss, acc, history, y_predict = ecg_cnn_lstm(train_x, train_y, test_x, test_y)
        # history_sum['ecg'] = history
        # #
        print("Running CNN-LSTM")
        loss, acc, history, y_predict = stacked_cnn_lstm(train_x, train_y, test_x, test_y)
        history_sum['cnn_lstm'] = history

        acc = acc * 100.0
        loss = loss * 100.0
        print('On Step %d test loss: %.3f' % (r + 1, loss))
        print('On Step %d test accuracy: %.3f' % (r + 1, acc))
        list_acc.append(acc)
        list_loss.append(loss)
        history_sum[r] = history
        # Plot the ROC
        roc_auc_for_each_class(3, test_y, y_predict, "ecg")
        # Plot ACC and LOSS
    # summarize results
    # summarize_results(list_acc, list_loss)
    # plot_model(history_sum)


def k_fold_run(fold=3):
    # load data
    data_file = 'data/dl_data.csv'

    seed = 7
    np.random.seed(seed)

    # ## repeat experiment
    list_acc = list()
    list_loss = list()
    history_sum = dict()

    # Define K folds
    kfold = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed)
    data_x, data_y = prepare_k_fold_data(data_file, time_step=120, features=13)

    r = 0
    for train, test in kfold.split(data_x, data_y):
        # to categorical and one hot encoding
        train_y = one_hot_encoding(data_y[train])
        test_y = one_hot_encoding(data_y[test])

        print("Running CNN-LSTM")
        loss, acc, history, y_predict = stacked_cnn_lstm(data_x[train], train_y, data_x[test], test_y)
        history_sum['cnn_lstm'] = history

        # print("Running CAD CNN-LSTM")
        # loss, acc, history, y_predict = ecg_cnn_lstm(data_x[train], train_y, data_x[test], test_y)
        # history_sum['ecg'] = history

        # print("Running LSTM")
        # loss, acc, history, y_predict = lstm_model(data_x[train], train_y, data_x[test], test_y)
        # history_sum['lstm'] = history
        #
        acc = acc * 100.0
        loss = loss * 100.0
        print('On Fold %d test loss: %.3f' % (r + 1, loss))
        print('On Fold %d test accuracy: %.3f' % (r + 1, acc))

        r = r + 1

        # list_acc.append(acc)
        # list_loss.append(loss)
        history_sum[r] = history

        # Plot the ROC
        roc_auc_for_each_class(3, test_y, y_predict, "cnn-lstm")

        break

    # summarize_results(list_acc, list_loss)

    plot_model(history_sum)



# # run the experiment
run_experiment(repeats=1)


# K-Fold experiment
# k_fold_run(fold=5)
