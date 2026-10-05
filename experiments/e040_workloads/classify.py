"""E040 W2 (docs/research/0181): an ordinary scikit-learn program, run unchanged — a linear
classifier (SGD, 5 epochs) on sparse TF-IDF-like features of 400 k documents x 200 k terms
(Zipf-distributed term ids, ~120 terms per document), labels from a hidden linear rule."""

import numpy as np
import scipy.sparse as sp
from common import done, sha
from sklearn.linear_model import SGDClassifier

DOCS, TERMS, PER_DOC = 400_000, 200_000, 120
rng = np.random.default_rng(0)
nnz = DOCS * PER_DOC
cols = (rng.zipf(1.3, nnz) - 1) % TERMS
rows = np.repeat(np.arange(DOCS), PER_DOC)
vals = np.log1p(rng.integers(1, 5, nnz)).astype(np.float64)
X = sp.csr_matrix((vals, (rows, cols)), shape=(DOCS, TERMS))
X.sum_duplicates()
idf = np.log(DOCS / (1 + np.bincount(X.indices, minlength=TERMS)))
X = X.multiply(idf).tocsr()
w = rng.standard_normal(TERMS)
y = (X @ w > np.median(X @ w)).astype(np.int8)
clf = SGDClassifier(max_iter=5, tol=None, random_state=0).fit(X, y)
done(coef=sha(np.ascontiguousarray(clf.coef_)), accuracy=float(clf.score(X, y)), nnz=int(X.nnz))
