from DAE.dae_emb_disaggregator import DAEEmbeddingsDisaggregator
from DAE.daedisaggregator import DAEDisaggregator
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
SAVED_MODEL = "clustering_model/gmm.pkl"
from sklearn.externals import joblib


def test_ukdale_building2(exp):
    exp.set_test_params(test_dataset_path=UK_DALE, test_dataset_name=UK_DALE_NAME,
                               test_building=2)
    exp.run_experiment()

def test_ukdale_buidling1_short_period(exp):
    exp.set_test_params(test_dataset_path=UK_DALE, test_dataset_name=UK_DALE_NAME,
                               test_building=1)
    exp.set_testing_window(start="1-1-2015", end="31-1-2015")
    exp.run_experiment()

clustering_model = joblib.load(SAVED_MODEL)

dae = DAEEmbeddingsDisaggregator(WINDOW_KETTLE, clustering_model)
experiment = Experiment(train_dataset_name=UK_DALE_NAME,
                        name="DAE",
                        disaggregator=dae,
                        train_dataset_path=UK_DALE,
                        train_building=1,
                        start=START,
                        end=END,
                        embeddings=True,
                        sample_period=6,
                        device=DEVICE,
                        epochs=KETTLE_EPOCHS)
experiment.train_model()
test_ukdale_building2(experiment)
test_ukdale_buidling1_short_period(experiment)
print("#------------------------------------------------------------------------------------------------------------")
dae = DAEDisaggregator(WINDOW_KETTLE)
experiment = Experiment(train_dataset_name=UK_DALE_NAME,
                        name="DAE",
                        disaggregator=dae,
                        train_dataset_path=UK_DALE,
                        train_building=1,
                        start=START,
                        end=END,
                        embeddings=False,
                        sample_period=6,
                        device=DEVICE,
                        epochs=KETTLE_EPOCHS)
experiment.train_model()
test_ukdale_building2(experiment)
test_ukdale_buidling1_short_period(experiment)



#--------------------------------------------------------------
DEVICE = "microwave"

dae = DAEEmbeddingsDisaggregator(WINDOW_MICROWAVE, clustering_model)
experiment = Experiment(train_dataset_name=UK_DALE_NAME,
                        name="DAE",
                        disaggregator=dae,
                        train_dataset_path=UK_DALE,
                        train_building=1,
                        start=START,
                        end=END,
                        embeddings=True,
                        sample_period=6,
                        device=DEVICE,
                        epochs=MICROWAVE_EPOCHS)
experiment.train_model()
test_ukdale_building2(experiment)
test_ukdale_buidling1_short_period(experiment)
print("#------------------------------------------------------------------------------------------------------------")
dae = DAEDisaggregator(WINDOW_MICROWAVE)
experiment = Experiment(train_dataset_name=UK_DALE_NAME,
                        name="DAE",
                        disaggregator=dae,
                        train_dataset_path=UK_DALE,
                        train_building=1,
                        start=START,
                        end=END,
                        embeddings=False,
                        sample_period=6,
                        device=DEVICE,
                        epochs=MICROWAVE_EPOCHS)
experiment.train_model()
test_ukdale_building2(experiment)
test_ukdale_buidling1_short_period(experiment)

DEVICE = "fridge"

dae = DAEEmbeddingsDisaggregator(WINDOW_FIRDGE, clustering_model)
experiment = Experiment(train_dataset_name=UK_DALE_NAME,
                        name="DAE",
                        disaggregator=dae,
                        train_dataset_path=UK_DALE,
                        train_building=1,
                        start=START,
                        end=END,
                        embeddings=True,
                        sample_period=6,
                        device=DEVICE,
                        epochs=FRIDGE_EPOCHS)
experiment.train_model()
test_ukdale_building2(experiment)
test_ukdale_buidling1_short_period(experiment)
print("#------------------------------------------------------------------------------------------------------------")
dae = DAEDisaggregator(WINDOW_FIRDGE)
experiment = Experiment(train_dataset_name=UK_DALE_NAME,
                        name="DAE",
                        disaggregator=dae,
                        train_dataset_path=UK_DALE,
                        train_building=1,
                        start=START,
                        end=END,
                        embeddings=False,
                        sample_period=6,
                        device=DEVICE,
                        epochs=FRIDGE_EPOCHS)
experiment.train_model()
test_ukdale_building2(experiment)
test_ukdale_buidling1_short_period(experiment)



DEVICE = "washing machine"

dae = DAEEmbeddingsDisaggregator(WINDOW_WASHING_MACHINE, clustering_model)
experiment = Experiment(train_dataset_name=UK_DALE_NAME,
                        name="DAE",
                        disaggregator=dae,
                        train_dataset_path=UK_DALE,
                        train_building=1,
                        start=START,
                        end=END,
                        embeddings=True,
                        sample_period=6,
                        device=DEVICE,
                        epochs=WASHING_MACHINE_EPOCHS)
experiment.train_model()
test_ukdale_building2(experiment)
test_ukdale_buidling1_short_period(experiment)
print("#------------------------------------------------------------------------------------------------------------")
dae = DAEDisaggregator(WINDOW_WASHING_MACHINE)
experiment = Experiment(train_dataset_name=UK_DALE_NAME,
                        name="DAE",
                        disaggregator=dae,
                        train_dataset_path=UK_DALE,
                        train_building=1,
                        start=START,
                        end=END,
                        embeddings=False,
                        sample_period=6,
                        device=DEVICE,
                        epochs=WASHING_MACHINE_EPOCHS)
experiment.train_model()
test_ukdale_building2(experiment)
test_ukdale_buidling1_short_period(experiment)

DEVICE = "dish washer"

dae = DAEEmbeddingsDisaggregator(WINDOW_DISH_WASHER, clustering_model)
experiment = Experiment(train_dataset_name=UK_DALE_NAME,
                        name="DAE",
                        disaggregator=dae,
                        train_dataset_path=UK_DALE,
                        train_building=1,
                        start=START,
                        end=END,
                        embeddings=True,
                        sample_period=6,
                        device=DEVICE,
                        epochs=DISHWASHER_EPOCHS)
experiment.train_model()
test_ukdale_building2(experiment)
test_ukdale_buidling1_short_period(experiment)
print("#------------------------------------------------------------------------------------------------------------")
dae = DAEDisaggregator(WINDOW_DISH_WASHER)
experiment = Experiment(train_dataset_name=UK_DALE_NAME,
                        name="DAE",
                        disaggregator=dae,
                        train_dataset_path=UK_DALE,
                        train_building=1,
                        start=START,
                        end=END,
                        embeddings=False,
                        sample_period=6,
                        device=DEVICE,
                        epochs=DISHWASHER_EPOCHS)
experiment.train_model()
test_ukdale_building2(experiment)
test_ukdale_buidling1_short_period(experiment)

