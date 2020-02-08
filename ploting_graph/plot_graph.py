import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import researchpy as rp
from scipy import stats

# data = pd.read_csv("dataset.csv", delimiter=',', index_col=0)
# data1 = data.loc['2019-03-18 11:48:08':'2019-03-18 11:52:08']
# print(data1.head())
# data2 = data.loc['2019-03-18 12:01:18':'2019-03-18 12:06:56']
#
# plt.figure()
# d = len(data1.BR)
# d2 = len(data2.BR)
#
# plt.plot(range(d), data1.BR, 'b-', label='Resting BR')
# plt.plot(range(250), data2.BR[:250], 'r-', label='VR BR')
#
# plt.title('Resting BR and VR BR')
# plt.xlabel('Time(s)')
# plt.ylabel('Breathing Rate')
# plt.legend()
# plt.savefig('resting_vs_vr_br.png')
# plt.show()

data = pd.read_csv("raw_data_all_without_zero_remove.csv", delimiter=',', index_col=0)
dataSick = data.loc[data.Feedback > 2]
dataNormal = data.loc[data.Feedback <= 2]

# ## agg backend is used to create plot as a .png file
# mpl.use('agg')
#
# ## combine these different collections into a list
# data_to_plot = [dataNormal.HR, dataSick.HR, dataNormal.PC_HR, dataSick.PC_HR]
#
# # Create a figure instance
# fig = plt.figure(1, figsize=(9, 6))
#
# # Create an axes instance
# ax = fig.add_subplot(111)
#
# # Create the boxplot
# bp = ax.boxplot(data_to_plot)
# ax.set_xticklabels(['Resting + VR-Resting HR', 'VR-Simulation HR', 'Resting + VR-Resting Change', 'VR-Simulation Change'])
# ax.get_xaxis().tick_bottom()
# ax.get_yaxis().tick_left()
# # Save the figure
# fig.savefig('test.png', bbox_inches='tight')


print("Normal Data stat: ")
sample_normal = dataNormal.sample(n=500, random_state=1)
print(sample_normal.describe().to_csv("normal_desc.csv"))

print("Sick Data stat: ")
sample_sick = dataSick.sample(n=500, random_state=1)
print(sample_sick.describe().to_csv("sick_desc.csv"))




ttest_rslt = stats.ttest_ind(sample_sick['GSR'], sample_normal['GSR'])
print("Sick vs resting GSR: t-test: ", ttest_rslt)

ttest_rslt = stats.ttest_ind(sample_sick['HR'], sample_normal['HR'])
print("Sick vs Resting HR: t-test: ", ttest_rslt)

ttest_rslt = stats.ttest_ind(sample_sick['BR'], sample_normal['BR'])
print("Sick vs Resting BR: t-test: ", ttest_rslt)

ttest_rslt = stats.ttest_ind(sample_sick['HRV'], sample_normal['HRV'])
print("Sick vs Resting HRV: t-test: ", ttest_rslt)


