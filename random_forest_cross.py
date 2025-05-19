import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

features_df = pd.read_csv("elliptic_txs_features.csv", header=None)
classes_df = pd.read_csv("elliptic_txs_classes.csv")

features_df.columns = ["txId", "time_step"] + [f"feature_{i}" for i in range(1, 166)]
classes_df.columns = ["txId", "class"]

data_df = pd.merge(features_df, classes_df, on="txId")

data_df = data_df[data_df["class"] != "unknown"]

data_df["class"] = data_df["class"].map({"1": 0, "2": 1})

from sklearn.model_selection import TimeSeriesSplit, cross_val_score

data_df = data_df.sort_values('time_step')

X = data_df.drop(["txId", "time_step", "class"], axis=1)
y = data_df["class"]

model = RandomForestClassifier(n_estimators=100, random_state=42)

tscv = TimeSeriesSplit(n_splits=5)

f1_scores = []

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    f1 = f1_score(y_test, y_pred)
    f1_scores.append(f1)

print("Średni F1-score:", np.mean(f1_scores))
