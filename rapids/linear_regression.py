# The speedup obtained by using cuML's Linear Regression implementation
# becomes much higher when using larger datasets.
# Uncomment and use the n_samples value provided below to see the
# difference in the time required to run Scikit-learn's vs cuML's
# implementation with a large dataset.
from clearml import Task

Task.init(project_name="Rapids", task_name="LinearRegression")

n_samples = 2 ** 20
n_features = 399
random_state = 23

"""## Generate Data"""

import cudf
import pandas as pd
from cuml import make_regression
from cuml.preprocessing.model_selection import train_test_split

X, y = make_regression(n_samples=n_samples,
                       n_features=n_features,
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

### Fit, predict and evaluate
"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


ols_skl = LinearRegression(fit_intercept=True,
                           normalize=True,
                           n_jobs=-1)
ols_skl.fit(X_train_skl, y_train_skl)


predict_skl = ols_skl.predict(X_test_skl)


r2_score_skl = r2_score(y_test_skl, predict_skl)

"""## cuML Model

### Fit, predict and evaluate
"""

from cuml.linear_model import LinearRegression
from cuml.metrics import r2_score


ols_cuml = LinearRegression(fit_intercept=True,
                            normalize=True,
                            algorithm='eig')

ols_cuml.fit(X_train_cudf, y_train_cudf)


predict_cuml = ols_cuml.predict(X_test_cudf)


r2_score_cuml = r2_score(y_test_cudf, predict_cuml)

"""## Compare Results"""

print("R^2 score (SKL):  %s" % r2_score_skl)
print("R^2 score (cuML): %s" % r2_score_cuml)
