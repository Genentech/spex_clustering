import glob
import os
import pickle
import shutil
import time
import csv
import numpy as np
import umap
import hdbscan
import matplotlib.pyplot as plt

from decimal import Decimal


def run(**kwargs):

    cluster = kwargs.get("cluster")
    z_score = kwargs.get("z_score")
    folder_in_test_orig = "testing"
    folder_in_test_zscore = "testing"
    fn_out_train_csv = "training_dml.csv"
    folder_out_test = "testing_dml"
    data_storage = kwargs.get('data_storage')  # , 'C:\\temp\\DATA_STORAGE'
    related_task = kwargs.get('related_task')  # "348554471")
    related_parent = kwargs.get('related_parent')

    def load_related_data():
        path = os.path.join(data_storage, "jobs", related_task, related_parent, "result.pickle")
        file_path = os.path.join(os.path.dirname(__file__), "related_data.pickle")
        if not os.path.isfile(path):
            return {}
        else:
            shutil.copyfile(
                path,
                file_path,
            )
            with open(file_path, "rb") as infile:
                data = pickle.load(infile)
                return data
    # 30
    n_neighbors = int(kwargs.get("knn"))
    # 0.3
    min_dist = float(kwargs.get("min_dist"))
    related_data = load_related_data()

    test_data_orig = np.array(list(related_data.get('transformed', []))).astype(float)
    test_data_zscore = np.array(list(related_data.get('z_score', []))).astype(float)

    # 5,7,8,9,11,12,15,16,17,18,19,21,22,24,26,27
    # marker_list = sys.argv[9].split(',')
    # markers = [int(x) for x in marker_list]
    markers = kwargs.get("markers")
    if not markers:
        markers = [1, 2]

    print(cluster)
    print(z_score)
    print(folder_in_test_orig)
    print(folder_in_test_zscore)
    print(fn_out_train_csv)
    print(folder_out_test)
    print(n_neighbors)
    print(min_dist)
    print(markers)


    if not os.path.exists(folder_out_test):
        os.makedirs(folder_out_test)

    # for file in glob.glob(folder_in_test_orig + "/*.csv"):
    #     fn_in_test_list_orig.append(file)
    #     fn_in_test_list_zscore.append(
    #         folder_in_test_zscore + "/" + os.path.basename(file)
    #     )
    #     fn_out_test_csv_list.append(folder_out_test + "/" + os.path.basename(file))

    length = 1
    # Переворачиваем массив т.е. колонку с кластерами в строку
    train_labels = cluster[:, -1]  # cluster_id in one row
    # обрезаем колонку
    cluster = cluster[:, :-1]  # without cluster column
    train_data_for_calc = z_score[:, markers]  # тут мы выводим данные z_score без координат, чисто по каналам
    t1 = time.time()
    # передаем в маппер строку со спискеом кластер id
    # передаем в маппер массив данных по каналам без координат
    min_dist = 0.5
    n_neighbors = 50

    mapper = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42).fit(
        X=train_data_for_calc,
        y=train_labels
    )
    t2 = time.time()
    print("training umap done ", t2 - t1)

    # headers = headers[:-1]
    # headers.append("umap_x")
    # headers.append("umap_y")
    # headers.append("cluster_id")
    result1 = np.column_stack(
        (cluster, mapper.embedding_, train_labels.astype(float))
    )

    # fig, ax = plt.subplots(1, figsize=(14, 10))
    embedding = mapper.embedding_
    # plt.scatter(*embedding.T, s=0.1, cmap='Spectral', alpha=1.0)
    # plt.setp(ax, xticks=[], yticks=[])
    # cbar = plt.colorbar(boundaries=list(set(train_labels)))
    # cbar.set_ticks(list(set(train_labels)))

    #plt.show()


    (H, W) = cluster.shape
    # np.savetxt(
    #     fn_out_train_csv,
    #     result1,
    #     fmt="%.5f",
    #     delimiter=",",
    #     header=",".join(headers),
    #     comments="",
    # )
    res = {'dml': result1}

    print("training done")

    for i in range(length):
        # fn_in_test_orig = fn_in_test_list_orig[i]
        # fn_in_test_zscore = fn_in_test_list_zscore[i]
        # fn_out_test_csv = fn_out_test_csv_list[i]

        # with open(fn_in_test_orig, "r") as f:
        #     reader = csv.reader(f, delimiter=",")
        #     headers2 = next(reader)
        #     test_data_orig = np.array(list(reader)).astype(float)
        # with open(fn_in_test_zscore, "r") as f:
        #     reader = csv.reader(f, delimiter=",")
        #     headers2 = next(reader)
        #     test_data_zscore = np.array(list(reader)).astype(float)

        test_data_for_calc = test_data_zscore[:, markers]
        t0 = time.time()
        test_embedding = mapper.transform(test_data_for_calc)
        t1 = time.time()
        test_labels = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=50).fit_predict(
            test_embedding
        )
        t2 = time.time()

        # test_data[:, markers] = test_data_for_calc
        result2 = np.column_stack((test_data_orig, test_embedding, test_labels))


        # fig, ax = plt.subplots(1, figsize=(14, 10))
        # embedding = mapper.embedding_
        # plt.scatter(*embedding.T, s=0.1, cmap='Spectral', alpha=1.0)
        # plt.setp(ax, xticks=[], yticks=[])
        # plt.show()

        # np.savetxt(
        #     fn_out_test_csv,
        #     result2,
        #     fmt="%.5f",
        #     delimiter=",",
        #     header=",".join(headers),
        #     comments="",
        # )
        res.update({f'dml_{i}': result2})
        # print("{0} done: {1}, {2}".format(fn_out_test_csv, t1 - t0, t2 - t1))

    return res
