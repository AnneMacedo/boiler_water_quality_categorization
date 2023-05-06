import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from imblearn.over_sampling import SMOTE

import lazypredict
from lazypredict.Supervised import LazyClassifier
from IPython.display import display

# Visualize the proportion of borrowers
load_data = pd.read_csv("synthetic_conformity_data.csv")

load_data["conformity_class"] = np.where(
    load_data["conformity_class"] == "conforming", 1, 0
)


X = load_data.drop("conformity_class", axis=1)
y = load_data["conformity_class"]

# # SMOTE
smote = SMOTE(random_state=32)
X_smote_res, y_smote_res = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_smote_res, y_smote_res, test_size=0.30, random_state=999
)

clf = LazyClassifier(verbose=0, ignore_warnings=True)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

print(models)

