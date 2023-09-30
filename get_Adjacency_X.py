import numpy as np
import pandas as pd
import torch
import sklearn
from sklearn import preprocessing


def get_X_a():
    # path = 'hour_p_year.csv'
    path = 'hour_p_week.csv'
    data_raw = pd.read_csv(path)
    data_raw.values
    data = pd.DataFrame(data_raw)
    single_t_graph = []
    all_t_graph = []
    for i, row in data.iterrows():
        for j in range(len(row)):
            single_t_graph.append([row[j], row[j]])
        all_t_graph.append(single_t_graph)
        single_t_graph = []
    # np.save('/load_1y.npy',all_t_graph)
    np.save('load_ohio_1y.npy', all_t_graph)
    return
get_X_a()

def get_ohioA():
    path = 'xy.csv'
    data_raw = pd.read_csv(path)
    data_raw.values
    data = pd.DataFrame(data_raw).drop(['Bus Name'], axis=1)
    data = data.values

    A = np.ones((44, 44))
    # print(A)
    for i in range(44):
        for j in range(44):
            A[i][j] = np.sqrt(
                (data[i][0] - data[j][0]) ** 2 + (data[i][1] - data[j][1]) ** 2)
    for i in range(44):
        sortA = np.sort(A[i])
        for j in range(44):
            A[i][j] = 0 if A[i][j] < sortA[29] else 1
    np.save('cor_ohio_l2.npy', A)


# get_ohioA()


def get_A():
    # path = 'hour_p_year.csv'
    path='hour_p_week.csv'
    data_raw = pd.read_csv(path)
    data_raw.values
    data = pd.DataFrame(data_raw)

    cor_pearson = data.corr(method='pearson')
    cor_sprm = data.corr(method='spearman')

    cor_pearson = abs(preprocessing.minmax_scale(abs(cor_pearson)))
    from sklearn.metrics.cluster import normalized_mutual_info_score as nms
    # mutual = np.zeros((44,44))
    # for i in range(44):
    #     for j in range(44):
    #         mutual[i][j] = np.abs(nms(data['Bus {}'.format(i + 1)], data['Bus {}'.format(j + 1)]))
    # mutual = pd.DataFrame(mutual)
    # print(mutual)
    np.save('cor_pearson.npy', cor_pearson)
    np.save('cor_sprm.npy', cor_sprm)
    # np.save('/cor_mut.npy', mutual)
    return


get_A()


def load_metr_la_data():
    get_A()
    # get_ohioA()
    get_X_a()
    A = np.load("cor_pearson.npy")
    X = np.load("load_ohio_1y.npy")
    # X = np.load("c:/Users/19553/Desktop/Load_history_1y.npy")
    # print("np.shape(X) = ", np.shape(X))
    X = X.transpose((1, 2, 0))
    X = X.astype(np.float32)
    A = A.astype(np.float32)
    # Normalization using Z-score method

    means = np.asarray([np.mean(X[0]), np.mean(X[1])])
    # np.mean(X, axis= (0,2))
    X = X - means.reshape(1, -1, 1)

    stds = np.std(X, axis=(0, 2))

    X = X / stds.reshape(1, -1, 1)
    return A, X, means, stds


A, X, means, stds = load_metr_la_data()


def get_normalized_adj(A):
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5  # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    # Save samples
    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        target.append(X[:, 0, i + num_timesteps_input: j])

    return torch.from_numpy(np.array(features)), \
           torch.from_numpy(np.array(target))
