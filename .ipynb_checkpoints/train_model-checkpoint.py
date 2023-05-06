import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from imblearn.over_sampling import RandomOverSampler, SMOTE


def show_loan_distrib(data):
    count = ""
    if isinstance(data, pd.DataFrame):
        count = data["conformity_class"].value_counts()
    else:
        count = data.value_counts()

    count.plot(
        kind="pie", explode=[0, 0.1], figsize=(6, 6), autopct="%1.1f%%", shadow=True
    )
    plt.ylabel("Loan: conformity Vs. nonconforming")
    plt.legend(["nonconforming", "conformity"])
    plt.show()


# Visualize the proportion of borrowers
load_data = pd.read_csv("synthetic_conformity_data.csv")

load_data["conformity_class"] = load_data.apply(
    lambda row: 1 if row["conformity_class"] == "conforming" else 0,
    axis=1,
)
# show_loan_distrib(load_data)

X = load_data.drop("conformity_class", axis=1)
y = load_data["conformity_class"]

# Random Oversampler
# ros = RandomOverSampler(random_state=32)
# X_ros_res, y_ros_res = ros.fit_resample(X, y)

# # SMOTE
smote = SMOTE(random_state=32)
X_smote_res, y_smote_res = smote.fit_resample(X, y)

# Visualize the proportion of borrowers
# data_smote = X_smote_res
# data_smote["conformity_class"] = y_smote_res
# show_loan_distrib(data_smote)

X_train, X_test, y_train, y_test = train_test_split(
    X_smote_res, y_smote_res, test_size=0.30, stratify=y, random_state=2022
)

modelo = LinearSVC()

# print(X_train)

# modelo.fit(X_train, y_train)
# previsoes = modelo.predict(X_test)

# acuracia = accuracy_score(y_test, previsoes) * 100
# print("A acurácia foi %.2f%%" % acuracia)
