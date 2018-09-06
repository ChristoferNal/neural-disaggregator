from __future__ import print_function, division
import os
import time
import matplotlib.pyplot as plt
import metrics
import pandas as pd
from nilmtk import DataSet, HDFDataStore


class Experiment:
    def __init__(self, train_dataset_name, name, disaggregator, train_dataset_path, train_building, start, end,
                 sample_period, device, epochs, training_batch=128, embeddings=False, trainable_embeddings=False):
        self.__test_dataset_name = None
        self.__test_dataset = None
        self.__test_building = None
        self.__test_start = None
        self.__test_end = None
        self.__embeddings = embeddings
        self.__trainable_embeddings = trainable_embeddings
        self.__train_dataset_name = train_dataset_name
        self.__name = name
        self.__model = disaggregator
        self.__train_dataset = DataSet(train_dataset_path)
        self.train_building = train_building
        self.__sample_period = sample_period
        self.__meter_key = device
        self.__epochs = epochs
        self.__training_batch = training_batch
        self.__train_dataset.set_window(start=start, end=end)
        self.__train_folder_name = "{}/{}/{}/trainable_emb_{}/{}/{}_epochs{}_{}_{}".format(name, train_dataset_name,
                                                                                           train_building,
                                                                                           trainable_embeddings,
                                                                                           device, embeddings, epochs,
                                                                                           start, end)
        if not os.path.exists(self.__train_folder_name):
            os.makedirs(self.__train_folder_name)

    def set_test_params(self, test_dataset_path, test_dataset_name, test_building):
        self.__test_dataset_name = test_dataset_name
        self.__test_dataset = DataSet(test_dataset_path)
        self.__test_building = test_building

    def set_testing_window(self, start=None, end=None):
        self.__test_start = start
        self.__test_end = end
        self.__test_dataset.set_window(start=start, end=end)

    def train_model(self):
        start = time.time()
        print("training...")
        train_elec = self.__train_dataset.buildings[self.train_building].elec
        train_meter = train_elec.submeters()[self.__meter_key]
        train_mains = train_elec.mains()

        self.__model.train(train_mains, train_meter, epochs=self.__epochs, batch_size=self.__training_batch,
                           sample_period=self.__sample_period)

        self.__model.export_model("{}/trained_model.h5".format(self.__train_folder_name))
        end = time.time()
        print("Train finished in: ", end - start, " seconds.")

    def train_model_across_buildings(self, buildings):
        start = time.time()
        print("training...")
        mainlist = list()
        meterlist = list()
        # TODO: building is not used.
        for building in buildings:
            train_elec = self.__train_dataset.buildings[self.train_building].elec
            train_meter = train_elec.submeters()[self.__meter_key]
            train_mains = train_elec.mains()
            meterlist.append(train_meter)
            mainlist.append(train_mains)

        self.__model.train_across_buildings(mainlist, meterlist, self.__epochs, batch_size=self.__training_batch,
                                            sample_period=self.__sample_period)
        self.__model.export_model("{}/trained_model.h5".format(self.__train_folder_name))
        end = time.time()
        print("Train finished in: ", end - start, " seconds.")

    def run_experiment(self):
        self.__test_model()
        # self.__save_diagram()
        self.__save_results()

    def __test_model(self):
        print("Disagreggating...")
        test_elec, test_meter = self.__get_test_meter()
        test_mains = test_elec.mains()
        self.set_disag_filename()
        output = HDFDataStore(self.disag_filename, 'w')
        self.__model.disaggregate(test_mains, output, test_meter, sample_period=self.__sample_period)
        output.close()

    def set_disag_filename(self):
        self.disag_filename = "{}/{}_{}{}_{}_{}_out.h5".format(self.__train_folder_name, self.__meter_key,
                                                               self.__test_dataset_name,
                                                               self.__test_building, self.__test_start, self.__test_end)

    def __get_test_meter(self):
        test_elec = self.__test_dataset.buildings[self.__test_building].elec
        test_meter = test_elec.submeters()[self.__meter_key]
        return test_elec, test_meter

    def save_diagram(self, show_plot=False):
        test_elec, _ = self.__get_test_meter()
        result = DataSet(self.disag_filename)
        res_elec = result.buildings[self.__test_building].elec
        predicted = res_elec[self.__meter_key]
        print(predicted)
        ground_truth = test_elec[self.__meter_key]
        fig_name = "{}/{}_{}{}_{}_{}".format(self.__train_folder_name, self.__meter_key, self.__test_dataset_name,
                                             self.__test_building,
                                             self.__test_start, self.__test_end)
        predicted.plot()
        ground_truth.plot()
        plt.savefig(fig_name)
        if show_plot:
            plt.show()
        result.store.close()

    def __save_results(self):
        print("========== RESULTS ============")
        columns = ["train_dataset", "test_dataset", "train_building", "test_building", "embeddings", "epochs",
                   "sample_period", "device",
                   "recall", "precision", "accuracy", "f1", "rel_error_total_energy", "mean_abs_error"]
        df_results = pd.DataFrame(columns=columns)
        print("Saving results from file:")
        print(self.disag_filename)
        result = DataSet(self.disag_filename)
        res_elec = result.buildings[self.__test_building].elec
        _, test_meter = self.__get_test_meter()
        rpaf = metrics.recall_precision_accuracy_f1(res_elec[self.__meter_key], test_meter)
        index = 0
        df_results.loc[index, "recall"] = round(rpaf[0], 4)
        df_results.loc[index, "precision"] = round(rpaf[1], 4)
        df_results.loc[index, "accuracy"] = round(rpaf[2], 4)
        df_results.loc[index, "f1"] = round(rpaf[3], 4)

        relative_error = metrics.relative_error_total_energy(res_elec[self.__meter_key], test_meter)
        df_results.loc[index, "rel_error_total_energy"] = round(relative_error, 4)
        mean_abs_error = metrics.mean_absolute_error(res_elec[self.__meter_key], test_meter)
        df_results.loc[index, "mean_abs_error"] = round(mean_abs_error, 4)

        df_results.loc[index, "train_dataset"] = self.__train_dataset_name
        df_results.loc[index, "test_dataset"] = self.__test_dataset_name
        df_results.loc[index, "train_building"] = self.train_building
        df_results.loc[index, "test_building"] = self.__test_building
        df_results.loc[index, "epochs"] = self.__epochs
        df_results.loc[index, "sample_period"] = self.__sample_period
        df_results.loc[index, "device"] = self.__meter_key
        df_results.loc[index, "embeddings"] = self.__embeddings

        print(df_results)

        df_results.to_csv("{}/{}_building{}_{}_{}.csv".format(self.__train_folder_name, self.__test_dataset_name,
                                                              self.__test_building,
                                                              self.__test_start, self.__test_end))

        print("Recall: {}".format(rpaf[0]))
        print("Precision: {}".format(rpaf[1]))
        print("Accuracy: {}".format(rpaf[2]))
        print("F1 Score: {}".format(rpaf[3]))
        print("Relative error in total energy: {}".format(relative_error))
        print("Mean absolute error(in Watts): {}\n".format(mean_abs_error))
        print("Train dataset: {}".format(self.__train_dataset_name))
        print("Train building: {}".format(self.train_building))
        print("Test dataset: {}".format(self.__test_dataset_name))
        print("Test building: {}".format(self.__test_building))
        print("Epochs: {}".format(self.__epochs))
        print("Sample period: {}".format(self.__sample_period))
        print("Device: {}".format(self.__meter_key))
        result.store.close()
