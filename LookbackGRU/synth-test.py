from __future__ import print_function, division
from warnings import warn, filterwarnings

import random
import sys
import numpy as np
import pandas as pd
from keras.initializers import RandomNormal

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Conv1D, GRU, Bidirectional, Embedding, Conv2D, TimeDistributed, Flatten, Dropout, LSTM, \
    Reshape, GaussianNoise
from keras.utils import plot_model
from gen import opends, mmax, gen_batch
import matplotlib.pyplot as plt
import embeddings_reader
import metrics
with_embeddings = True
embed_dim = 100
embed_out = 100

def read_embeddings():
    df = pd.read_csv('energy_embeddings.csv')
    print('reading embeddings')
    print(df.head())
    devices_states = list(df.columns.values)
    energy_embeddings = df.values
    energy_embeddings = np.transpose(energy_embeddings)
    print('Number of embeddings: {}'.format(len(devices_states)))
    print('Embedding dimension: {}'.format(energy_embeddings[0].size))
    return devices_states, energy_embeddings

def create_model(input_window):
    '''Creates and returns the Neural Network
    '''
    model = Sequential()

    # 1D Conv
    if with_embeddings:
        devices_states, energy_embeddings = read_embeddings()
        embedding_dimension = energy_embeddings[0].size
        model.add(Reshape((input_window,), input_shape=(input_window, 1)))
        model.add(Embedding(len(devices_states),
                            embedding_dimension,
                            weights=[energy_embeddings],
                            trainable=False))
        # model.add(Embedding(embed_dim, embed_out, input_length=input_window, trainable=True,
        #                     embeddings_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None)))
        # model.add(Dropout(0.2))

        model.add(Conv1D(16, 4, activation="tanh", padding="same", strides=1))
        # model.add(TimeDistributed(Flatten()))
    else:
        model.add(Conv1D(16, 4, activation="linear", input_shape=(input_window,1), padding="same", strides=1))


    #Bi-directional LSTMs
    model.add(Bidirectional(GRU(64, return_sequences=True, activation='tanh'), merge_mode='concat'))
    model.add(Bidirectional(GRU(128, return_sequences=False,  activation='tanh'), merge_mode='concat'))

    # Fully Connected Layers
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer='adam')
    plot_model(model, to_file='model.png', show_shapes=True)

    return model

key_name = 'fridge' # The string ID of the meter
input_window = 200 # Lookback parameter
threshold = 60 # On Power Threshold
test_building = 5 # ID of the building to be used for testing

# ======= Training phase

# Open train sets
X_train = np.load("dataset/trainsets/X-{}.npy".format(key_name))
y_train = np.load("dataset/trainsets/Y-{}.npy".format(key_name))
model = create_model(input_window)

# Train model and save checkpoints
epochs_per_checkpoint = 1
for epochs in range(0,1,epochs_per_checkpoint):
    model.fit(X_train, y_train, batch_size=128, epochs=epochs_per_checkpoint, shuffle=True)
    model.save("SYNTH-LOOKBACK-{}-ALL-{}epochs-1WIN.h5".format(key_name, epochs+epochs_per_checkpoint),model)

# ======= Disaggregation phase
mains, meter= opends(test_building,key_name)
X_test = mains
y_test = meter*mmax

# Predict data
X_batch, Y_batch = gen_batch(X_test, y_test, len(X_test)-input_window, 0, input_window)
pred = model.predict(X_batch)* mmax
pred[pred<0] = 0
pred = np.transpose(pred)[0]
# Save results
np.save('pred.results',pred)
plt.plot(pred, 'b', Y_batch, 'r')
plt.ylabel('predictions')

plt.show()
# Calculate and show metrics
print(embed_dim)
print(embed_out)
print("============ Recall Precision Accurracy F1 {}".format(metrics.recall_precision_accuracy_f1(pred, Y_batch,threshold)))
print("============ relative_error_total_energy {}".format(metrics.relative_error_total_energy(pred, Y_batch)))
print("============ mean_absolute_error {}".format(metrics.mean_absolute_error(pred, Y_batch)))
