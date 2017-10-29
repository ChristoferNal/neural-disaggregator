from __future__ import print_function, division
import time

from matplotlib import rcParams
import matplotlib.pyplot as plt

from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from daedisaggregator import DAEDisaggregator
import metrics

print("========== OPEN DATASETS ============")
train = DataSet('../../Datasets/REDD/redd.h5')
test = DataSet('../../Datasets/REDD/redd.h5')

train.set_window(end="30-4-2011")
test.set_window(start="30-4-2011")

train_building = 1
test_building = 1
meter_key = 'fridge'
with_embeddings = False
epochs = 30
sample_period = 6
train_elec = train.buildings[train_building].elec
test_elec = test.buildings[test_building].elec

# meter_group = train.buildings[1].elec
# print("\nMeter Group")
# print(meter_group)
# df_all_meters = meter_group.dataframe_of_meters()
# for col in df_all_meters.columns:
#     meter = meter_group[col]
#     print(meter.metadata)

train_meter = train_elec.submeters()[meter_key]
train_mains = train_elec.mains().all_meters()[0]
test_mains = test_elec.mains().all_meters()[0]
dae = DAEDisaggregator(256, with_embeddings)


start = time.time()
print("========== TRAIN ============")
dae.train(train_mains, train_meter, epochs=epochs, sample_period=sample_period)
dae.export_model("model-redd100.h5")
end = time.time()
print("Train =", end-start, "seconds.")


print("========== DISAGGREGATE ============")
disag_filename = 'disag-out.h5'
output = HDFDataStore(disag_filename, 'w')
dae.disaggregate(test_mains, output, train_meter, sample_period=sample_period)
output.close()

result = DataSet(disag_filename)
res_elec = result.buildings[1].elec
predicted = res_elec[meter_key]
ground_truth = test_elec[meter_key]

import matplotlib.pyplot as plt
predicted.plot()
ground_truth.plot()
plt.show()
print("========== RESULTS ============")
result = DataSet(disag_filename)
res_elec = result.buildings[test_building].elec
rpaf = metrics.recall_precision_accuracy_f1(res_elec[meter_key], test_elec[meter_key])
print("============ Recall: {}".format(rpaf[0]))
print("============ Precision: {}".format(rpaf[1]))
print("============ Accuracy: {}".format(rpaf[2]))
print("============ F1 Score: {}".format(rpaf[3]))

print("============ Relative error in total energy: {}".format(metrics.relative_error_total_energy(res_elec[meter_key], test_elec[meter_key])))
print("============ Mean absolute error(in Watts): {}".format(metrics.mean_absolute_error(res_elec[meter_key], test_elec[meter_key])))
print("Train building: {}".format(train_building))
print("Test building: {}".format(test_building))
print("With Embeddings: {}".format(with_embeddings))
print("Epochs: {}".format(epochs))
print("Sample period: {}".format(sample_period))
print("Device: {}".format(meter_key))