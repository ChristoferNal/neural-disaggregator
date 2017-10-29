from daedisaggregator import DAEDisaggregator
from Experiment import Experiment

REDD = '../Datasets/REDD/redd.h5'
UK_DALE = '../Datasets/UKDALE/ukdale.h5'

use_embeddings = True
dae = DAEDisaggregator(300, use_embeddings)
exp_uk_fridge = Experiment(train_dataset_name="ukdale",
                                name="DAE",
                                disaggregator=dae,
                                train_dataset_path=UK_DALE,
                                train_building=2,
                                start="20-5-2013",
                                end="9-9-2013",
                                sample_period=1,
                                device='fridge',
                                with_embeddings=use_embeddings,
                                epochs=30)
exp_uk_fridge.train_model()
exp_uk_fridge.set_test_params(test_dataset_path=UK_DALE,
                              test_building=2)
exp_uk_fridge.set_testing_window(start="10-9-2013", end="10-10-2013")
exp_uk_fridge.run_experiment()

exp_uk_fridge.set_test_params(test_dataset_path=UK_DALE,
                              test_building=1)
exp_uk_fridge.set_testing_window(start="10-9-2013", end="10-10-2013")
exp_uk_fridge.run_experiment()

exp_uk_fridge.set_test_params(test_dataset_path=REDD,
                              test_building=1)
exp_uk_fridge.set_testing_window(start="30-4-2011")
exp_uk_fridge.run_experiment()