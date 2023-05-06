import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE

from lazypredict.Supervised import LazyClassifier, LazyRegressor

# Visualize the proportion of borrowers
load_data = pd.read_csv("synthetic_conformity_data.csv")

load_data["conformity_class"] = np.where(
    load_data["conformity_class"] == "conforming", 1, 0
)

dummie_data = pd.get_dummies(load_data)


X = dummie_data.drop("conformity_class", axis=1)
y = load_data["conformity_class"]

# # SMOTE
smote = SMOTE(random_state=32)
X_smote_res, y_smote_res = smote.fit_resample(X, y)


X_train, X_test, y_train, y_test = train_test_split(
    X_smote_res, y_smote_res, test_size=0.30, random_state=999
)

# Classifier
clf = LazyClassifier(verbose=0, ignore_warnings=True)
clf_models, clf_predictions = clf.fit(X_train, X_test, y_train, y_test)

print(clf_models)

# Regressor
reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
reg_models, reg_predictions = reg.fit(X_train, X_test, y_train, y_test)

print(reg_models)
