from DAE.daedisaggregator import DAEDisaggregator
from experiment import Experiment
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

REDD = '../Datasets/REDD/redd.h5'
UK_DALE = '../Datasets/UKDALE/ukdale.h5'
UK_DALE_NAME = "ukdale"
REDD_NAME = "redd"

use_embeddings = False
dae = DAEDisaggregator(300, use_embeddings)
exp_uk_fridge = Experiment(train_dataset_name=UK_DALE_NAME,
                                name="DAE",
                                disaggregator=dae,
                                train_dataset_path=UK_DALE,
                                train_building=1,
                                start="20-5-2013",
                                end="31-12-2014",
                                sample_period=6,
                                device='fridge',
                                with_embeddings=use_embeddings,
                                epochs=1)
exp_uk_fridge.train_model()
exp_uk_fridge.set_test_params(test_dataset_path=UK_DALE, test_dataset_name=UK_DALE_NAME,
                              test_building=1)
exp_uk_fridge.set_testing_window(start="1-1-2015", end="31-12-2015")
exp_uk_fridge.run_experiment()

exp_uk_fridge.set_test_params(test_dataset_path=UK_DALE, test_dataset_name=UK_DALE_NAME,
                              test_building=2)
exp_uk_fridge.set_testing_window(start=None, end=None)
exp_uk_fridge.run_experiment()

exp_uk_fridge.set_test_params(test_dataset_path=REDD, test_dataset_name=REDD_NAME,
                              test_building=1)
exp_uk_fridge.run_experiment()
