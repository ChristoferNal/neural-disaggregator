from __future__ import print_function, division
import time

from matplotlib import rcParams
import matplotlib.pyplot as plt

from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from daedisaggregator import DAEDisaggregator
import metrics

print("========== OPEN DATASETS ============")
train = DataSet('../../Datasets/UKDALE/ukdale.h5')
test = DataSet('../../Datasets/UKDALE/ukdale.h5')

train.set_window(start="20-5-2013", end="9-9-2013")
test.set_window(start="10-9-2013", end="10-10-2013")

train_building = 2
test_building = 2
sample_period = 1
meter_key = 'laptop computer'
with_embeddings = True
epochs = 30

train_elec = train.buildings[train_building].elec
test_elec = test.buildings[test_building].elec

train_meter = train_elec.submeters()[meter_key]
test_meter = test_elec.submeters()[meter_key]
train_mains = train_elec.mains()
test_mains = test_elec.mains()

print('---------------------------------------------------------------------------')
print("train_elec")
print(train_meter.metadata)
print('---------------------------------------------------------------------------')
print("test elec")
print(test_mains)
print('---------------------------------------------------------------------------')
dae = DAEDisaggregator(300, with_embeddings)

start = time.time()
print("========== TRAIN ============")
dae.train(train_mains, train_meter, epochs=epochs, sample_period=sample_period)

# for i in range(3):
#     print("CHECKPOINT {}".format(epochs))
#     dae.train(train_mains, train_meter, epochs=15, sample_period=sample_period)
#     epochs += 5
#     dae.export_model("UKDALE-DAE-h{}-{}-{}epochs.h5".format(train_building,
#                                                         meter_key,
#                                                         epochs))


end = time.time()
print("Train =", end-start, "seconds.")


print("========== DISAGGREGATE ============")
disag_filename = "disag-out.h5"
output = HDFDataStore(disag_filename, 'w')
dae.disaggregate(test_mains, output, test_meter, sample_period=sample_period)
output.close()

result = DataSet(disag_filename)
res_elec = result.buildings[test_building].elec
predicted = res_elec[meter_key]
ground_truth = test_elec[meter_key]

import matplotlib.pyplot as plt
predicted.plot()
ground_truth.plot()
plt.show()

print("========== RESULTS ============")
result = DataSet(disag_filename)
res_elec = result.buildings[test_building].elec
rpaf = metrics.recall_precision_accuracy_f1(res_elec[meter_key], test_meter)
print("============ Recall: {}".format(rpaf[0]))
print("============ Precision: {}".format(rpaf[1]))
print("============ Accuracy: {}".format(rpaf[2]))
print("============ F1 Score: {}".format(rpaf[3]))

print("============ Relative error in total energy: {}".format(metrics.relative_error_total_energy(res_elec[meter_key], test_meter)))
print("============ Mean absolute error(in Watts): {}".format(metrics.mean_absolute_error(res_elec[meter_key], test_meter)))
print("Train building: {}".format(train_building))
print("Test building: {}".format(test_building))
print("With Embeddings: {}".format(with_embeddings))
print("Epochs: {}".format(epochs))
print("Sample period: {}".format(sample_period))
print("Device: {}".format(meter_key))

print(res_elec[meter_key].on_power_threshold())
print(test_meter.on_power_threshold())