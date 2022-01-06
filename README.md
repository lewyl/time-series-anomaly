# Univariate Time Series Anomaly Detection

## Model
An ensemble of OPTICS models is used to detect anomalies in multi-dataset univariate time series. The ensembling method is an implementation of [Chesnokov, M.Y. Time Series Anomaly Searching Based on DBSCAN Ensembles. Sci. Tech. Inf. Proc. 46, 299–305 (2019)][1]. The model predicts the probability of each point being an outlier. For faster running time using the scikit-learn library, OPTICS algorithm is used instead of DBSCAN.


## Data
KDD Cup 2021 details can be found at
https://compete.hexagon-ml.com/practice/competition/39/

## Requirements
- numpy
- pandas
- scikit-learn
- matplotlib
- os
- csv



[1]: https://doi.org/10.3103/S0147688219050010 "Chesnokov, M.Y. Time Series Anomaly Searching Based on DBSCAN Ensembles. Sci. Tech. Inf. Proc. 46, 299–305 (2019)."
