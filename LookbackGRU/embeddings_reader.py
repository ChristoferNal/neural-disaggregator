import pandas as pd
import numpy as np

def read():
    df = pd.read_csv('energy_embeddings.csv')
    print('reading embeddings')
    print(df.head())
    devices_states = list(df.columns.values)
    energy_embeddings = df.values
    energy_embeddings = np.transpose(energy_embeddings)
    print('Number of embeddings: {}'.format(len(devices_states)))
    print('Embedding dimension: {}'.format(energy_embeddings[0].size))
    return devices_states, energy_embeddings

