# Directional extent supervision for lesion diameter (MIDL 2027)

Code, manifests and per-case results for "Does Directional Extent Supervision Improve Lesion Diameter Agreement? A Controlled Study on Ovarian and Breast Ultrasound Masks".

Repository: https://github.com/nhynhi23411/MIDL2027_Directional_Extent_Ultrasound

## Layout
- `code/` losses (`losses.py`), metrics (`metrics.py`), audit, grouped splits (`prep.py`), Modal training app (`modal_app.py`), statistics (`analyze*.py`), figures (`make_*.py`).
- `manifests/` dataset manifest, annotation issue log, grouped split manifest, frozen hyperparameters, measurement protocol.
- `results/` per-case results for the first study (`final_*`), v2 (`v2final_*`) and v3 (`v3final_*`), pilots, ablation, paired contrasts.
- `paper/` LaTeX source, tables, figures.

## Reproduce
1. Download MMOTU and BUSI from their public releases into `data/` (images are not redistributed).
2. `python code/audit.py && python code/prep.py`
3. `modal run code/modal_app.py --stage final` (also `v2final`, `v3final`), then `python code/analyze.py`, `analyze_v2.py`, `analyze_v3.py`, `make_tables_v3.py`.
4. `python paper/build.py`

Data are used under the terms set by their authors. No clinical claims are made.
