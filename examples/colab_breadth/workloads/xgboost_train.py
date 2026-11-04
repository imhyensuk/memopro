"""E053 (docs/research/0268): gradient boosting with XGBoost on 3 M x 64 float32 rows (hist, 40
rounds). XGBoost keeps its quantized matrix and trees in its own allocations, outside NumPy."""

import numpy as np
import xgboost as xgb
from common import done, sha

N, D = 3_000_000, 64
rng = np.random.default_rng(0)
X = rng.standard_normal((N, D), dtype=np.float32)
y = X[:, 0] * X[:, 1] + np.sin(X[:, 2]) + 0.1 * rng.standard_normal(N, dtype=np.float32) > 0
dtrain = xgb.QuantileDMatrix(X, label=y.astype(np.float32))
params = {
    "tree_method": "hist",
    "max_depth": 8,
    "eta": 0.2,
    "objective": "binary:logistic",
    "nthread": 2,
    "seed": 0,
}
booster = xgb.train(params, dtrain, num_boost_round=40)
pred = booster.predict(xgb.DMatrix(X[:200_000]))
done(pred=sha(pred), xgboost=xgb.__version__)
