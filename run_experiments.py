from DAE.dae_emb_disaggregator import DAEEmbeddingsDisaggregator
from DAE.daedisaggregator import DAEDisaggregator
from GRU.gru_emb_disaggregator import GRUEmbeddingsDisaggregator
from GRU.grudisaggregator import GRUDisaggregator
from RNN.rnndisaggregator import RNNDisaggregator
from experiment import Experiment
from tensorflow.python.client import device_lib

END = "31-12-2014"

START = "1-1-2014"

print(device_lib.list_local_devices())

REDD = '../Datasets/REDD/redd.h5'
UK_DALE = '../Datasets/UKDALE/ukdale.h5'
UK_DALE_NAME = "ukdale"
REDD_NAME = "redd"
DEVICE = "kettle"

WINDOW_KETTLE = 100
WINDOW_MICROWAVE = 50
WINDOW_WASHING_MACHINE = 200
WINDOW_DISH_WASHER = 100
WINDOW_FIRDGE = 50

KETTLE_EPOCHS = 130
MICROWAVE_EPOCHS = 80
DISHWASHER_EPOCHS = 130
WASHING_MACHINE_EPOCHS = 200
FRIDGE_EPOCHS = 110
# KETTLE_EPOCHS = 15
# MICROWAVE_EPOCHS = 15
# DISHWASHER_EPOCHS = 15
# WASHING_MACHINE_EPOCHS = 15
# FRIDGE_EPOCHS = 15

SAVED_MODEL = "clustering_model/gmm.pkl"
from sklearn.externals import joblib

def test_ukdale_building5(exp):
    exp.set_test_params(test_dataset_path=UK_DALE, test_dataset_name=UK_DALE_NAME,
                               test_building=5)
    exp.set_testing_window(start="1-7-2014", end="30-7-2014")
    exp.run_experiment()

def test_ukdale_building2(exp):
    exp.set_test_params(test_dataset_path=UK_DALE, test_dataset_name=UK_DALE_NAME,
                               test_building=2)
    exp.set_testing_window()
    exp.run_experiment()

def test_ukdale_buidling1_short_period(exp):
    exp.set_test_params(test_dataset_path=UK_DALE, test_dataset_name=UK_DALE_NAME,
                               test_building=1)
    exp.set_testing_window(start="1-1-2015", end="31-1-2015")
    exp.run_experiment()

def test_redd_building1(exp):
    exp.set_test_params(test_dataset_path=REDD, test_dataset_name=REDD_NAME,
                               test_building=1)
    exp.set_testing_window()
    exp.run_experiment()

clustering_model = joblib.load(SAVED_MODEL)
trainable = True

MODEL_NAME = "GRU"
dae = GRUEmbeddingsDisaggregator(WINDOW_KETTLE, clustering_model, trainable)
experiment = Experiment(train_dataset_name=UK_DALE_NAME,
                        name=MODEL_NAME,
                        disaggregator=dae,
                        train_dataset_path=UK_DALE,
                        train_building=1,
                        start=START,
                        end=END,
                        embeddings=True,
                        sample_period=6,
                        device=DEVICE,
                        epochs=KETTLE_EPOCHS,
                        trainable_embeddings = trainable)
buildings = list()
buildings.append(1)
buildings.append(2)
buildings.append(3)
buildings.append(4)
experiment.train_model_across_buildings(buildings)
test_ukdale_building5(experiment)
print("#------------------------------------------------------------------------------------------------------------")
# dae = GRUDisaggregator()
# experiment = Experiment(train_dataset_name=UK_DALE_NAME,
#                         name=MODEL_NAME,
#                         disaggregator=dae,
#                         train_dataset_path=UK_DALE,
#                         train_building=1,
#                         start=START,
#                         end=END,
#                         embeddings=False,
#                         sample_period=6,
#                         device=DEVICE,
#                         epochs=KETTLE_EPOCHS)
# buildings = list()
# buildings.append(1)
# buildings.append(2)
# buildings.append(3)
# buildings.append(4)
# experiment.train_model_across_buildings(buildings)
# test_ukdale_building5(experiment)

#--------------------------------------------------------------
DEVICE = "microwave"

dae = GRUEmbeddingsDisaggregator(WINDOW_MICROWAVE, clustering_model, trainable)
experiment = Experiment(train_dataset_name=UK_DALE_NAME,
                        name=MODEL_NAME,
                        disaggregator=dae,
                        train_dataset_path=UK_DALE,
                        train_building=1,
                        start=START,
                        end=END,
                        embeddings=True,
                        sample_period=6,
                        device=DEVICE,
                        epochs=MICROWAVE_EPOCHS,
                        trainable_embeddings = trainable)
buildings = list()
buildings.append(1)
buildings.append(2)
experiment.train_model_across_buildings(buildings)
test_ukdale_building5(experiment)
test_redd_building1(experiment)
print("#------------------------------------------------------------------------------------------------------------")
# dae = GRUDisaggregator()
# experiment = Experiment(train_dataset_name=UK_DALE_NAME,
#                         name=MODEL_NAME,
#                         disaggregator=dae,
#                         train_dataset_path=UK_DALE,
#                         train_building=1,
#                         start=START,
#                         end=END,
#                         embeddings=False,
#                         sample_period=6,
#                         device=DEVICE,
#                         epochs=MICROWAVE_EPOCHS)
# buildings = list()
# buildings.append(1)
# buildings.append(2)
# experiment.train_model_across_buildings(buildings)
# test_ukdale_building5(experiment)
# test_redd_building1(experiment)

DEVICE = "fridge"

dae = GRUEmbeddingsDisaggregator(WINDOW_FIRDGE, clustering_model, trainable)
experiment = Experiment(train_dataset_name=UK_DALE_NAME,
                        name=MODEL_NAME,
                        disaggregator=dae,
                        train_dataset_path=UK_DALE,
                        train_building=1,
                        start=START,
                        end=END,
                        embeddings=True,
                        sample_period=6,
                        device=DEVICE,
                        epochs=FRIDGE_EPOCHS,
                        trainable_embeddings = trainable)
buildings = list()
buildings.append(1)
buildings.append(2)
buildings.append(4)
experiment.train_model_across_buildings(buildings)
test_ukdale_building5(experiment)
test_redd_building1(experiment)

print("#------------------------------------------------------------------------------------------------------------")
# dae = DAEDisaggregator(WINDOW_FIRDGE)
# experiment = Experiment(train_dataset_name=UK_DALE_NAME,
#                         name=MODEL_NAME,
#                         disaggregator=dae,
#                         train_dataset_path=UK_DALE,
#                         train_building=1,
#                         start=START,
#                         end=END,
#                         embeddings=False,
#                         sample_period=6,
#                         device=DEVICE,
#                         epochs=FRIDGE_EPOCHS)
# buildings = list()
# buildings.append(1)
# buildings.append(2)
# buildings.append(4)
# experiment.train_model_across_buildings(buildings)
# test_ukdale_building5(experiment)
# test_redd_building1(experiment)



DEVICE = "washing machine"

dae = GRUEmbeddingsDisaggregator(WINDOW_WASHING_MACHINE, clustering_model, trainable)
experiment = Experiment(train_dataset_name=UK_DALE_NAME,
                        name=MODEL_NAME,
                        disaggregator=dae,
                        train_dataset_path=UK_DALE,
                        train_building=1,
                        start=START,
                        end=END,
                        embeddings=True,
                        sample_period=6,
                        device=DEVICE,
                        epochs=WASHING_MACHINE_EPOCHS,
                        trainable_embeddings = trainable)
buildings = list()
buildings.append(1)
buildings.append(5)
experiment.train_model_across_buildings(buildings)
test_ukdale_building2(experiment)
test_redd_building1(experiment)

print("#------------------------------------------------------------------------------------------------------------")
# dae = GRUDisaggregator()
# experiment = Experiment(train_dataset_name=UK_DALE_NAME,
#                         name=MODEL_NAME,
#                         disaggregator=dae,
#                         train_dataset_path=UK_DALE,
#                         train_building=1,
#                         start=START,
#                         end=END,
#                         embeddings=False,
#                         sample_period=6,
#                         device=DEVICE,
#                         epochs=WASHING_MACHINE_EPOCHS)
# buildings = list()
# buildings.append(1)
# buildings.append(5)
# experiment.train_model_across_buildings(buildings)
# test_ukdale_building2(experiment)
# test_redd_building1(experiment)

DEVICE = "dish washer"

dae = GRUEmbeddingsDisaggregator(WINDOW_DISH_WASHER, clustering_model, trainable)
experiment = Experiment(train_dataset_name=UK_DALE_NAME,
                        name=MODEL_NAME,
                        disaggregator=dae,
                        train_dataset_path=UK_DALE,
                        train_building=1,
                        start=START,
                        end=END,
                        embeddings=True,
                        sample_period=6,
                        device=DEVICE,
                        epochs=DISHWASHER_EPOCHS,
                        trainable_embeddings = trainable)
buildings = list()
buildings.append(1)
buildings.append(2)
experiment.train_model_across_buildings(buildings)
test_ukdale_building5(experiment)
test_redd_building1(experiment)

print("#------------------------------------------------------------------------------------------------------------")
# dae = GRUDisaggregator()
# experiment = Experiment(train_dataset_name=UK_DALE_NAME,
#                         name=MODEL_NAME,
#                         disaggregator=dae,
#                         train_dataset_path=UK_DALE,
#                         train_building=1,
#                         start=START,
#                         end=END,
#                         embeddings=False,
#                         sample_period=6,
#                         device=DEVICE,
#                         epochs=DISHWASHER_EPOCHS)
# buildings = list()
# buildings.append(1)
# buildings.append(2)
# experiment.train_model_across_buildings(buildings)
# test_ukdale_building5(experiment)
# test_redd_building1(experiment)

