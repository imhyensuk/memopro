"""E053 (docs/research/0268): scikit-learn on a large dense table — 1.2 M x 256 float32
features, PCA to 32 dimensions (randomized SVD), then a random forest (30 trees) on 400 k rows."""

import numpy as np
from common import done, sha
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

N, D = 1_200_000, 256
rng = np.random.default_rng(0)
basis = rng.standard_normal((16, D)).astype(np.float32)
X = rng.standard_normal((N, 16), dtype=np.float32) @ basis
X += 0.1 * rng.standard_normal((N, D), dtype=np.float32)
y = (X[:, :8].sum(axis=1) > 0).astype(np.int8)
Z = PCA(n_components=32, svd_solver="randomized", random_state=0).fit_transform(X)
rf = RandomForestClassifier(n_estimators=30, max_depth=12, n_jobs=-1, random_state=0)
rf.fit(Z[:400_000], y[:400_000])
pred = rf.predict(Z[400_000:])
done(
    pca=sha(np.ascontiguousarray(Z[:1000])),
    accuracy=float((pred == y[400_000:]).mean()),
    pred=sha(pred),
)
