import matplotlib.pyplot as plt
from nilmtk import DataSet
# from gen import opends
import pandas as pd

RESULTS_EMBEDDED = "/home/christoforos/PycharmProjects/neural-disaggregator/LookbackGRU/results/SYNTH-LOOKBACK-fridge-ALL-1epochs-1WIN-Embedded.h5"
df = pd.read_hdf(RESULTS_EMBEDDED)
df.plot()


result = DataSet(RESULTS_EMBEDDED)
res_elec = result.buildings[1].elec

key_name = 'fridge' # The string ID of the meter
# mains, meter= opends(5,key_name)
# X_test = mains

predicted = res_elec['fridge']
# ground_truth = test_elec['microwave']

predicted.plot()
# ground_truth.plot()
plt.show()