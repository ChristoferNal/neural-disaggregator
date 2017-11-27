from DAE.dae_emb_disaggregator import DAEEmbeddingsDisaggregator
from DAE.daedisaggregator import DAEDisaggregator
from experiment import Experiment
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

REDD = '../Datasets/REDD/redd.h5'
UK_DALE = '../Datasets/UKDALE/ukdale.h5'
UK_DALE_NAME = "ukdale"
REDD_NAME = "redd"
DEVICE = "kettle"
WINDOW_KETTLE = 128
SAVED_MODEL = "gmm.pkl"
from sklearn.externals import joblib

clustering_model = joblib.load(SAVED_MODEL)

dae = DAEEmbeddingsDisaggregator(WINDOW_KETTLE, clustering_model)
experiment = Experiment(train_dataset_name=UK_DALE_NAME,
                        name="DAE",
                        disaggregator=dae,
                        train_dataset_path=UK_DALE,
                        train_building=1,
                        start=None,
                        end=None,
                        sample_period=6,
                        device=DEVICE,
                        epochs=25)
# exp_uk_fridge.train_model()
# exp_uk_fridge.train_building = 2
# exp_uk_fridge.train_model()
# exp_uk_fridge.train_building = 3
experiment.train_model()

experiment.set_test_params(test_dataset_path=UK_DALE, test_dataset_name=UK_DALE_NAME,
                           test_building=2)
experiment.run_experiment()

dae = DAEEmbeddingsDisaggregator(WINDOW_KETTLE, clustering_model)
experiment = Experiment(train_dataset_name=UK_DALE_NAME,
                        name="DAE",
                        disaggregator=dae,
                        train_dataset_path=UK_DALE,
                        train_building=1,
                        start=None,
                        end=None,
                        sample_period=6,
                        device=DEVICE,
                        epochs=25)
experiment.train_model()
experiment.set_test_params(test_dataset_path=UK_DALE, test_dataset_name=UK_DALE_NAME,
                           test_building=2)
experiment.run_experiment()
