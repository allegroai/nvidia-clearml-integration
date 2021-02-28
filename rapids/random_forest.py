# The speedup obtained by using cuML's Random Forest implementation
# becomes much higher when using larger datasets.
# Uncomment and use the n_samples value provided below to see the
# difference in the time required to run Scikit-learn's vs cuML's
# implementation with a large dataset.
import numpy as np

from clearml import Task

Task.init(project_name="Rapids", task_name="RandomForestClassifier")

# Make classification parameters
n_samples = 2**12
n_features = 399
n_informative = 300
n_clases = 2


# RandomForestClassifier parameters
n_estimators=40
max_depth=16
max_features=1.0

# Other parameters
random_state=32

"""## Generate Data"""

import cudf
import pandas as pd
from cuml.datasets import make_classification
from cuml.preprocessing.model_selection import train_test_split

X, y = make_classification(n_samples=n_samples,
                           n_features=n_features,
                           n_informative=n_informative,
                           n_classes=n_clases,
                           random_state=random_state)

# Create cuDF DataFrame and Series from CuPy ndarray.
X = cudf.DataFrame(X)
y = cudf.Series(y)

# Split dataset into training and testing datasets.
X_train_cudf, X_test_cudf, y_train_cudf, y_test_cudf = \
    train_test_split(X, y, test_size=0.2, random_state=random_state)

# Copy dataset from GPU memory to host memory.
# This is done to later compare CPU and GPU results.
X_train_skl = X_train_cudf.to_pandas()
X_test_skl = X_test_cudf.to_pandas()
y_train_skl = y_train_cudf.to_pandas()
y_test_skl = y_test_cudf.to_pandas()

"""## Scikit-learn Model

### Fit & Evaluate
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model_skl = RandomForestClassifier(n_estimators=n_estimators,
                                   max_depth=max_depth,
                                   max_features=max_features,
                                   random_state=random_state)
model_skl.fit(X_train_skl, y_train_skl)


predict_skl = model_skl.predict(X_test_skl)
acc_skl = accuracy_score(y_test_skl, predict_skl)

"""## cuML Model

### Fit & Evaluate
"""

from cuml.ensemble import RandomForestClassifier
from cuml.metrics import accuracy_score


model_cuml = RandomForestClassifier(n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    max_features=max_features,
                                    random_state=random_state)
model_cuml.fit(X_train_cudf, y_train_cudf)


predict_cuml = model_cuml.predict(X_test_cudf)
acc_cuml = accuracy_score(y_test_cudf, predict_cuml)

"""## Compare Scikit-learn and cuML results"""

print("Scikit-learn accuracy:\t%s" % acc_skl)
print("cuML accuracy:\t\t%s" % acc_cuml)

"""## Pickle the cuML random forest classification model"""

import joblib

# save the trained cuml model into a file
filename = 'cuml_random_forest_model.sav'
joblib.dump(model_cuml, filename)

# delete the previous model to ensure that there is no leakage of pointers.
# this is not strictly necessary but just included here for demo purposes.
del model_cuml

# load the previously saved cuml model from a file
pickled_model_cuml = joblib.load(filename)

"""### Predict using the pickled model"""


pred_after_pickling = pickled_model_cuml.predict(X_test_cudf)

fil_acc_after_pickling = accuracy_score(y_test_cudf, pred_after_pickling)

"""### Compare results before and after pickling"""

print("cuML accuracy of the RF model before pickling:\t%s" % acc_cuml)
print("cuML accuracy of the RF model after pickling:\t%s" % fil_acc_after_pickling)
