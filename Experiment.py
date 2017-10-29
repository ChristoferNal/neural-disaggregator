from __future__ import print_function, division
import time

import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt
from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from daedisaggregator import DAEDisaggregator
import metrics
import matplotlib.pyplot as plt


class Experiment:
    def __init__(self, train_dataset_name, name, disaggregator, train_dataset_path, train_building, start, end, sample_period, device,
                 with_embeddings, epochs):
        self.__test_dataset = None
        self.__test_building = None
        self.__train_dataset_name = train_dataset_name
        self.__name = name
        self.__model = disaggregator
        self.__train_dataset = DataSet(train_dataset_path)
        self.__train_building = train_building
        self.__sample_period = sample_period
        self.__meter_key = device
        self.__with_embeddings = with_embeddings
        self.__epochs = epochs
        self.__train_dataset.set_window(start=start, end=end)
        self.__train_folder_name = "{}_{}_building{}_dev{}_embed{}_epochs{}_{}_{}".format(name, train_dataset_name, train_building, device,
                                                                                          with_embeddings, epochs, start, end)

    def set_test_params(self, test_dataset_path, test_building):
        self.__test_dataset = DataSet(test_dataset_path)
        self.__test_building = test_building

    def set_testing_window(self, start, end):
        self.__test_dataset.set_window(start=start, end=end)

    def train_model(self):
        start = time.time()
        print("training...")
        train_elec = self.__train_dataset.buildings[self.__train_building].elec
        train_meter = train_elec.submeters()[self.__meter_key]
        train_mains = train_elec.mains()

        self.__model.train(train_mains, train_meter, epochs=self.__epochs, sample_period=self.__sample_period)

        self.__model.export_model("{}/trained_model.h5".format(self.__train_folder_name))
        end = time.time()
        print("Train finished in: ", end - start, " seconds.")

    def run_experiment(self):
        predicted, ground_truth = self.__test_model()
        self.__save_diagram(predicted, ground_truth)
        self.__save_results()

    def __test_model(self):
        print("Disagreggating...")
        test_elec, test_meter = self.__get_test_meter()
        test_mains = test_elec.mains()
        self.disag_filename = "{}_disag-out.h5".format(self.__name)
        output = HDFDataStore(self.disag_filename, 'w')
        self.__model.disaggregate(test_mains, output, test_meter, sample_period=self.__sample_period)
        output.close()

        result = DataSet(self.disag_filename)
        res_elec = result.buildings[self.__test_building].elec
        predicted = res_elec[self.__meter_key]
        ground_truth = test_elec[self.__meter_key]

        return predicted, ground_truth

    def __get_test_meter(self):
        test_elec = self.__test_dataset.buildings[self.__test_building].elec
        test_meter = test_elec.submeters()[self.__meter_key]
        return test_elec, test_meter

    def __save_diagram(self, predicted, ground_truth, show_plot=False):
        fig_name = "{}_{}".format(self.__name, self.__meter_key)
        predicted.plot()
        ground_truth.plot()
        plt.savefig(fig_name)
        if show_plot:
            plt.show()

    def __save_results(self):
        print("========== RESULTS ============")
        columns = ["train_building", "test_building", "embedings", "epochs", "sample_period", "device",
                   "recall", "precision", "accuracy", "f1", "rel_error_total_energy", "mean_abs_error"]
        df_results = pd.DataFrame(columns=columns)
        result = DataSet(self.disag_filename)
        res_elec = result.buildings[self.__test_building].elec
        test_meter = self.__get_test_meter()
        rpaf = metrics.recall_precision_accuracy_f1(res_elec[self.__meter_key], test_meter)
        df_results["recall"] = rpaf[0]
        df_results["precision"] = rpaf[1]
        df_results["accuracy"] = rpaf[2]
        df_results["f1"] = rpaf[3]

        relative_error = metrics.relative_error_total_energy(res_elec[self.__meter_key], test_meter)
        df_results["rel_error_total_energy"] = relative_error
        mean_abs_error = metrics.mean_absolute_error(res_elec[self.__meter_key], test_meter)
        df_results["mean_abs_error"] = mean_abs_error

        df_results["train_building"] = self.__train_building
        df_results["test_building"] = self.__test_building
        df_results["embedings"] = self.__with_embeddings
        df_results["epochs"] = self.__epochs
        df_results["sample_period"] = self.__sample_period
        df_results["device"] = self.__meter_key

        df_results.to_csv("{}/results_building{}.csv".format(self.__test_building))

        print("Recall: {}".format(rpaf[0]))
        print("Precision: {}".format(rpaf[1]))
        print("Accuracy: {}".format(rpaf[2]))
        print("F1 Score: {}".format(rpaf[3]))
        print("Relative error in total energy: {}".format(relative_error))
        print("Mean absolute error(in Watts): {}\n".format(mean_abs_error))
        print("Train building: {}".format(self.__train_building))
        print("Test building: {}".format(self.__test_building))
        print("With Embeddings: {}".format(self.__with_embeddings))
        print("Epochs: {}".format(self.__epochs))
        print("Sample period: {}".format(self.__sample_period))
        print("Device: {}".format(self.__meter_key))
