"""E040 W3 (docs/research/0181): an ordinary NumPy simulation, run unchanged — 2-D heat
diffusion on a 4096 x 4096 float32 grid (explicit scheme, 300 steps), keeping every 10th field
(30 snapshots) in memory, then per-cell time mean and maximum over the history."""

import numpy as np
from common import done, sha

N, STEPS, EVERY = 4096, 300, 10
y, x = np.mgrid[0:N, 0:N].astype(np.float32) / N
u = np.zeros((N, N), np.float32)
for cx, cy, s in ((0.3, 0.3, 0.05), (0.7, 0.6, 0.08), (0.5, 0.8, 0.03)):
    u += np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * s * s)).astype(np.float32)
del x, y
history = np.empty((STEPS // EVERY, N, N), np.float32)
for step in range(STEPS):
    lap = -4 * u
    lap[1:] += u[:-1]
    lap[:-1] += u[1:]
    lap[:, 1:] += u[:, :-1]
    lap[:, :-1] += u[:, 1:]
    u += np.float32(0.2) * lap
    if step % EVERY == 0:
        history[step // EVERY] = u
mean = history.mean(axis=0)
peak = history.max(axis=0)
done(mean=sha(mean), peak=sha(peak), history=sha(history), final=sha(u))
