Dev check for 0344 (whole-process ceiling on the PyTorch path), 8 GB M1 with other apps open, MPS.
`torch_ft.py <model> <ceiling>`: memopro.finetune on E034's ten texts, seq_len 129, batch 1.
`torch_gen.py <model> <ceiling>`: memopro.generate, 32 new tokens, draft "auto".
`*_old.json` = main before 0344 (d532a25/3e9be4d), `*_new*.json` = this change.
peak_MiB = process lifetime peak footprint (rt.process_peak_footprint).
