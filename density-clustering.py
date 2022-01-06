import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN, OPTICS, cluster_optics_dbscan
import matplotlib.pyplot as plt

import os
import csv


## 0. Set hyperparameters - MinPts and eps
# MinPts
M = [5, 10, 50]
# Percentile to choose eps
P = [5, 10, 30, 50, 70, 90, 95]

# specify directory of dataset
dataset_dir = "./data/"
# write probability result to file
output_dir = "output_prob.csv"
with open(output_dir, 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
# write prediction result to file
header = ['file_id', 'predicted_anomaly']
with open('predictions_density_clustering.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

for filename in sorted(os.listdir(dataset_dir)):
    pred_outlier = 0

    if filename.endswith(".txt"):
        ## 1. Load data
        print("Now processing {}".format(filename))
        # get info from file name
        breakpoint = int(filename.split("_")[-1].split(".")[0]) # split point for test data
        file_id = filename[0:3]
        # load data as pandas dataframe
        data_path = dataset_dir + filename
        data = pd.read_csv(data_path, header=None)

        ## 2. Data preprocessing
        # 2.1. Managing data in different formats
        # if data given in a row, convert to one data point per row
        if len(data) > 1: # time series data given in many rows
            X = data
        elif len(data) == 1: # time series data given in one line
            data = pd.read_csv(data_path, sep="  ", header=None, engine='python')
            X = data.T
        else:
            print('bug')
            continue

        # 2.2 Standardize X values
        X_mean = X.mean()
        X_std = X.std()
        X_norm = (X - X_mean) / X_std
        X_norm = X_norm.to_numpy() # convert to numpy array

        # 2.3 Detrend by simple differencing
        diff = []
        for i in range(1, len(X_norm)):
            value = X_norm[i] - X_norm[i - 1]
            diff.append(value)
        X_norm = np.array(diff)

        ## 3. MODEL: Ensemble to get abnormality estimate of each point ##
        ## 3.1. Find distances for set M
        MinPts_eps = {} # hyperparameters (minpts, eps) candidates for each component in ensemble
        model_count = 0 # number of components in ensemble
        # generate hyperparameter pairs (minpts, eps)
        for m in M:
            D_q = []
            # find average distance d_i from point i to its immediate neighbors in the window
            # (the neighbor MinPts on one side and MinPts on the other side along the time axis)
            for i in range(m, len(X_norm) - m, 1):
                d_i = np.mean(abs(X_norm[i-m:i+m+1]-X_norm[i]))
                D_q.append(d_i)
            # 3.2. Find eps as P percentiles of D_q
            for percentile in P:
                eps = np.percentile(D_q, percentile)
                if eps > 0:
                    model_count += 1
                    if m not in MinPts_eps.keys():
                        MinPts_eps[m] = [eps]
                    else:
                        MinPts_eps[m].append(eps)

        # 3.3. Apply density-based clustering to all derived pairs of (MinPts,eps) and obtain binary tokens for each observation
        binary_tokens_agg = np.zeros(len(X_norm))
        for m in MinPts_eps.keys():
            eps_s = MinPts_eps[m] # list of eps candidates for this minpts
            # build the clustering model
            model = OPTICS(min_samples=m, max_eps=eps_s[-1])
            # train the model
            model.fit(X_norm)
            for eps in eps_s:
                # produce the labels according to the DBSCAN technique with eps in eps_s
                labels = cluster_optics_dbscan(reachability=model.reachability_,
                                               core_distances=model.core_distances_,
                                               ordering=model.ordering_, eps=eps)
                # outlier points are labelled as -1,
                # convert outlier points -1 to 1, and cluster points to 0
                binary_tokens = abs(labels * (labels < 0))
        # 3.4. Calculate abnormality estimate by combining results from individual components
                binary_tokens_agg += binary_tokens
                print('done clustering on MinPts_eps pair {}'.format((m,eps)))
        prob_anomaly = binary_tokens_agg / model_count # abnormality estimate

        ## 4. Results
        # 4.1. write probablility to file 
        output_prob = prob_anomaly[breakpoint:]
        record = [file_id] + [x for x in output_prob]
        with open('output_prob.csv', 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(record)

        # 4.2. Get outlier point (prediction from density-based clustering only)
        outlier_prob = max(prob_anomaly[breakpoint:]) # find the maximum abnormality estimate of this dataset
        outlier_index = np.where(prob_anomaly == outlier_prob) # find where does this maximum occur
        # if maximum occur at more than one point, pick first point in the test set
        if len(outlier_index)>0:
            for i in outlier_index[0]:
                if i >= breakpoint:
                    pred_outlier = i
                    break
        # write result to file
        record = [file_id, pred_outlier]
        with open('predictions_density_clustering.csv', 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(record)

        # 4.3. visualize the predicted point
        plt.figure()
        plt.plot(X, '-bo', markevery=list(outlier_index))
        plt.show()



