import numpy as np
import tensorflow as tf
import plot_keras_history as pkh
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from DataPreparation import inputsFromFile, labelsFromFile


def createModel(
        regul=0, lrate=0.001, momen=0, dpath="",
        act=1,
        InputN=8520, HiddenN=4270, SecondN=0, OutputN=20):
    inputs = tf.keras.Input(shape=(InputN,))
    activs = [tf.nn.relu, tf.nn.sigmoid]
    activ = activs[act]
    if regul > 0:
        Regularizer = tf.keras.regularizers.L2(l2=regul)
    else:
        Regularizer = None
    x = tf.keras.layers.Dense(
        HiddenN, activation=activ,
        kernel_regularizer=Regularizer
    )(inputs)
    if SecondN > 0:
        x = tf.keras.layers.Dense(
            SecondN, activation=activ,
            kernel_regularizer=Regularizer
        )(x)
    outputs = tf.keras.layers.Dense(OutputN, activation=tf.nn.sigmoid)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=lrate, momentum=momen),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.BinaryCrossentropy(from_logits=True),
            tf.keras.metrics.MeanSquaredError(),
            tf.keras.metrics.BinaryAccuracy()
            ]
    )
    return model


def evaluateFold(model, Dtrain, Dval, epoch_n):
    weights = model.get_weights()
    stopper = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, mode="auto", min_delta=0.0002)
    hist = model.fit(
        Dtrain, epochs=epoch_n, verbose=2,
        validation_data=Dval, callbacks=[stopper])
    score = model.evaluate(Dval, verbose=1)
    model.set_weights(weights)
    return hist, score


def runDemoModel(
        model, epoch_n, dpath="", name="model", debug=False, scale=False):
    x_train = inputsFromFile(dpath + 'Data/train-data.dat')
    y_train = labelsFromFile(dpath + 'Data/train-label.dat')
    if scale:
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    scores = []
    kFold = KFold(n_splits=5)
    count = 0
    for train, test in kFold.split(x_train, y_train):
        DatasetTrain = tf.data.Dataset.from_tensor_slices(
                (x_train[train], y_train[train])).batch(50)
        DatasetTest = tf.data.Dataset.from_tensor_slices(
                (x_train[test], y_train[test])).batch(30)
        hist, score = evaluateFold(model, DatasetTrain, DatasetTest, epoch_n)
        scores.append(score)
        # plot training history
        pkh.plot_history(hist, path='result_{}_{}.png'.format(name, count))
        if debug:
            break
    # average of 5 evaluation results
    nscores = np.array(scores)
    avscore = np.mean(nscores, axis=0)
    return avscore
