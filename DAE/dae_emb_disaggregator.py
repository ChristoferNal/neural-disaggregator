from DAE.daedisaggregator import DAEDisaggregator
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, Reshape, Dropout, Embedding
from keras.utils import plot_model
import pandas as pd

EMBEDDINGS_CSV = 'embeddings/energy_embeddings_gmm.csv'
TOKENIZATION_WINDOW = 10

class DAEEmbeddingsDisaggregator(DAEDisaggregator):
    def __init__(self, sequence_length, clustering_model):
       # super(DAEDisaggregator, self).__init__()
        self.clustering_model = clustering_model
        self.MODEL_NAME = "AUTOENCODER"
        self.mmax = None
        #mode_tokens = sequence_length % TOKENIZATION_WINDOW
        #if mode_tokens is not 0:
        #    sequence_length += TOKENIZATION_WINDOW - mode_tokens
        self.sequence_length = sequence_length
        self.MIN_CHUNK_LENGTH = self.sequence_length
        self.model = self._create_model(self.sequence_length)

    def train(self, mains, meter, epochs=1, batch_size=16, **load_kwargs):
        main_power_series = mains.power_series(**load_kwargs)
        meter_power_series = meter.power_series(**load_kwargs)

        run = True
        mainchunk = next(main_power_series)
        meterchunk = next(meter_power_series)
        if self.mmax == None:
            self.mmax = meterchunk.max()

        while (run):
            meterchunk = self._normalize(meterchunk, self.mmax)

            self.train_on_chunk(mainchunk, meterchunk, epochs, batch_size)
            try:
                mainchunk = next(main_power_series)
                meterchunk = next(meter_power_series)
            except:
                run = False

    def train_on_chunk(self, mainchunk, meterchunk, epochs, batch_size):
        '''Train using only one chunk

        Parameters
        ----------
        mainchunk : chunk of site meter
        meterchunk : chunk of appliance
        epochs : number of epochs for training
        '''

        s = self.sequence_length
        # up_limit =  min(len(mainchunk), len(meterchunk))
        # down_limit =  max(len(mainchunk), len(meterchunk))

        # Replace NaNs with 0s
        mainchunk.fillna(0, inplace=True)
        meterchunk.fillna(0, inplace=True)
        ix = mainchunk.index.intersection(meterchunk.index)
        mainchunk = mainchunk[ix]
        meterchunk = meterchunk[ix]
        print("len of mainchunk")
        print(len(mainchunk))
        print(type(mainchunk))

        # Create array of batches
        # additional = s - ((up_limit-down_limit) % s)
        factor = s * TOKENIZATION_WINDOW
        additional = factor - (len(ix) % factor)
        print("additional: {}, factor: {}".format(additional, s))
        X_batch = np.append(mainchunk, np.zeros(additional))
        Y_batch = np.append(meterchunk, np.zeros(additional))

        Y_batch = np.mean(Y_batch.reshape(-1, TOKENIZATION_WINDOW), axis=1)
        X_batch = X_batch.reshape(-1, TOKENIZATION_WINDOW)
        X_batch = self.clustering_model.predict(X_batch)
        print("len of tokenized sequence")
        print(len(X_batch))

        X_batch = np.reshape(X_batch, (int(len(X_batch) / s), s, 1))
        Y_batch = np.reshape(Y_batch, (int(len(Y_batch) / s), s, 1))

        self.model.fit(X_batch, Y_batch, batch_size=batch_size, epochs=epochs, shuffle=True)

    def disaggregate(self, mains, output_datastore, meter_metadata, **load_kwargs):
        '''Disaggregate mains according to the model learnt.

        Parameters
        ----------
        mains : nilmtk.ElecMeter
        output_datastore : instance of nilmtk.DataStore subclass
            For storing power predictions from disaggregation algorithm.
        meter_metadata : metadata for the produced output
        **load_kwargs : key word arguments
            Passed to `mains.power_series(**kwargs)`
        '''

        load_kwargs = self._pre_disaggregation_checks(load_kwargs)

        load_kwargs.setdefault('sample_period', 60)
        load_kwargs.setdefault('sections', mains.good_sections())

        timeframes = []
        building_path = '/building{}'.format(mains.building())
        mains_data_location = building_path + '/elec/meter1'
        data_is_available = False

        for chunk in mains.power_series(**load_kwargs):
            if len(chunk) < self.MIN_CHUNK_LENGTH:
                continue
            print("New sensible chunk: {}".format(len(chunk)))

            timeframes.append(chunk.timeframe)
            measurement = chunk.name

            appliance_power = self.disaggregate_chunk(chunk)
            appliance_power[appliance_power < 0] = 0
            appliance_power = self._denormalize(appliance_power, self.mmax)

            # Append prediction to output
            data_is_available = True
            cols = pd.MultiIndex.from_tuples([chunk.name])
            meter_instance = meter_metadata.instance()
            df = pd.DataFrame(
                appliance_power.values, index=appliance_power.index,
                columns=cols, dtype="float32")
            key = '{}/elec/meter{}'.format(building_path, meter_instance)
            output_datastore.append(key, df)

            # Append aggregate data to output
            mains_df = pd.DataFrame(chunk, columns=cols, dtype="float32")
            output_datastore.append(key=mains_data_location, value=mains_df)

        # Save metadata to output
        if data_is_available:
            self._save_metadata_for_disaggregation(
                output_datastore=output_datastore,
                sample_period=load_kwargs['sample_period'],
                measurement=measurement,
                timeframes=timeframes,
                building=mains.building(),
                meters=[meter_metadata]
            )

    def disaggregate_chunk(self, mains):
        '''In-memory disaggregation.

        Parameters
        ----------
        mains : pd.Series to disaggregate
        Returns
        -------
        appliance_powers : pd.DataFrame where each column represents a
            disaggregated appliance.  Column names are the integer index
            into `self.model` for the appliance in question.
        '''
        s = self.sequence_length
        up_limit = len(mains)

        mains.fillna(0, inplace=True)

        factor = s * TOKENIZATION_WINDOW
        additional = factor - (up_limit % factor)
        print("additional: {}, factor: {}".format(additional, s))
        X_batch = np.append(mains, np.zeros(additional))
        X_batch = X_batch.reshape(-1, TOKENIZATION_WINDOW)
        X_batch = self.clustering_model.predict(X_batch)

        X_batch = np.reshape(X_batch, (int(len(X_batch) / s), s, 1))

        pred = self.model.predict(X_batch)
        pred = np.repeat(pred, TOKENIZATION_WINDOW)
        pred = np.reshape(pred, (up_limit + additional))[:up_limit]
        column = pd.Series(pred, index=mains.index, name=0)

        appliance_powers_dict = {}
        appliance_powers_dict[0] = column
        appliance_powers = pd.DataFrame(appliance_powers_dict)
        return appliance_powers

    def _read_embeddings(self):
        df = pd.read_csv(EMBEDDINGS_CSV)
        print('reading embeddings...')
        devices_states = list(df.columns.values)
        energy_embeddings = df.values
        energy_embeddings = np.transpose(energy_embeddings)
        print('Number of embeddings: {}'.format(len(devices_states)))
        print('Embedding dimension: {}'.format(energy_embeddings[0].size))
        return devices_states, energy_embeddings

    def _create_model(self, sequence_len):
        '''Creates the Auto encoder module described in the paper
        '''
        model = Sequential()

        devices_states, energy_embeddings = self._read_embeddings()
        embedding_dimension = energy_embeddings[0].size
        model.add(Reshape((sequence_len,), input_shape=(sequence_len, 1)))
        model.add(Embedding(len(devices_states),
                            embedding_dimension,
                            weights=[energy_embeddings],
                            trainable=False))
        model.add(Conv1D(8, 4, activation="linear", padding="same", strides=1))

        model.add(Flatten())

        # Fully Connected Layers
        model.add(Dropout(0.2))
        model.add(Dense((sequence_len - 0) * 8, activation='relu'))

        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))

        model.add(Dropout(0.2))
        model.add(Dense((sequence_len - 0) * 8, activation='relu'))

        model.add(Dropout(0.2))

        # 1D Conv
        model.add(Reshape(((sequence_len - 0), 8)))
        model.add(Conv1D(1, 4, activation="linear", padding="same", strides=1))
        model.compile(loss='mse', optimizer='adam')
        # model.compile(loss=losses.mean_squared_error, optimizer='sgd')
        plot_model(model, to_file='model.png', show_shapes=True)

        return model
