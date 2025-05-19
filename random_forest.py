import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

features_df = pd.read_csv("elliptic_txs_features.csv", header=None)
classes_df = pd.read_csv("elliptic_txs_classes.csv")

features_df.columns = ["txId", "time_step"] + [f"feature_{i}" for i in range(1, 166)]
classes_df.columns = ["txId", "class"]

data_df = pd.merge(features_df, classes_df, on="txId")

data_df = data_df[data_df["class"] != "unknown"]

data_df["class"] = data_df["class"].map({"1": 0, "2": 1})

train_df = data_df[data_df["time_step"] <= 35]
test_df = data_df[data_df["time_step"] > 35]

X_train = train_df.drop(["txId", "time_step", "class"], axis=1)
y_train = train_df["class"]

X_test = test_df.drop(["txId", "time_step", "class"], axis=1)
y_test = test_df["class"]

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

print("F1-score:", f1_score(y_test, y_pred))
