"""Ecosystem integrations (architecture §4.3, 0032 I8).

Each module imports its framework only when used, so ``import memopro`` stays free of
transformers, lightning and IPython (0032 I4).

- ``ipython``   ``%load_ext memopro``: ``%hibernate``, ``%wake``, ``%memopro status``
- ``hf``        census callback for the Hugging Face ``Trainer`` (v0.1)
- ``lightning`` census callback for Lightning (v0.1)
"""
