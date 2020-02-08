import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


# Group by Epoch Wise results
def epoch_wise_summary(file):
    data = pd.read_csv(file, delimiter=',')
    group = data.groupby([data['Epoch']])[
        'HR', 'PC_HR', 'HR_MIN', 'HR_MAX', 'PC_HRV', 'GSR', 'PC_GSR', 'GSR_MIN', 'GSR_MAX', 'Feedback'].mean().reset_index()
    group = group.sort_values('Epoch', axis=0, ascending=True)
    print("Epoch Wise Summary")
    print(group)
    group.to_csv("epoch_wise_summary.csv")

    return group


def individual_wise_summary(file):
    data = pd.read_csv(file, delimiter=',')
    group = data.groupby([data['Individual']])['Epoch'].max().reset_index()
    group = group.sort_values('Individual', axis=0, ascending=True)
    print("indiv Wise Summary")
    print(group)
    # group.to_csv("epoch_wise_summary.csv")

    return group


def plot_epoch_summary_graph(data, x, y):
    x_ticks = ['Resting-baseline', 'VR-Resting', 'Loop 1', 'Loop 2', 'Loop 3', 'Loop 4', 'Loop 5', 'Loop 6',
               'Loop 7', 'Loop 8', 'Loop 9', 'Loop 10', 'Loop 11', 'Loop 12', 'Loop 13']

    plt.close('all')
    plt.plot(data[x], data[y], 'r--', marker='o')
    # plt.xlabel('Roller Coas')
    plt.ylabel('Sickness Rating')
    plt.xticks(data[x], x_ticks, rotation='vertical')
    plt.tight_layout()
    plt.savefig('epoch_vs_' + y + '.png', dpi=100)
    plt.show()


def pearson_corr(data, x, y):
    r, p = stats.pearsonr(data[x], data[y])
    print("Pearson Correlation of ", x, "and", y)
    print("R value: ", r, "p value: ", p)


def correlation_heatmap(data):
    plt.close('all')
    corrmat = data.corr()
    f, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(corrmat, ax=ax, cmap="YlGnBu", linewidths=0.1)
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=100)


def remove_outliers_z_score(data):
    z = np.abs(stats.zscore(data))
    threshold = 3  # z score value for which we will consider it an outlier
    outliers = np.where(z > 3)
    dataset_without_outliers = data[(z < 3).all(axis=1)]

    return dataset_without_outliers


def ssq_sr_summary(file):
    data = pd.read_csv(file, delimiter=',')
    print(data)
    pearson_corr(data, 'Last_SR', 'SSQ_Total-Post')
    pearson_corr(data, 'Last_SR', 'Nausea_Total(post)')
    pearson_corr(data, 'Last_SR', 'Ocolomotor_Total(post)')
    pearson_corr(data, 'Last_SR', 'Disorientation(Total)_Post')

    return data



# Epoch Summary Graph
# epoch_summary = epoch_wise_summary('raw_data.csv')
# plot_epoch_summary_graph(epoch_summary, 'Epoch', 'Feedback')

# epoch_summary = individual_wise_summary('raw_data.csv')
# plot_epoch_summary_graph(epoch_summary, 'Epoch', 'BR')
# plot_epoch_summary_graph(epoch_summary, 'Epoch', 'HRV')
# plot_epoch_summary_graph(epoch_summary, 'Epoch', 'GSR')

### SSQ and SR Rating
ssq = ssq_sr_summary('old_history_data/ssq_and_sickness_rating.csv')
plot_ssq(ssq)

# Correaltion heatmap of epoch summary
# correlation_heatmap(epoch_summary)

# Pearson correaltion
# pearson_corr(epoch_summary, 'HR', 'Feedback')
# pearson_corr(epoch_summary, 'BR', 'Feedback')
# pearson_corr(epoch_summary, 'HRV', 'Feedback')
# pearson_corr(epoch_summary, 'GSR', 'Feedback')
#
#
# print("Overall COrrelation")
# data = pd.read_csv('raw_data.csv', delimiter=',')
# print(data.head())
# Correaltion heatmap of epoch summary
# correlation_heatmap(data)
# pearson_corr(data, 'HR', 'Feedback')
# pearson_corr(data, 'BR', 'Feedback')
# pearson_corr(data, 'HRV', 'Feedback')
# pearson_corr(data, 'GSR', 'Feedback')
#
