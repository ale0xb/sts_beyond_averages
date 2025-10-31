# Beyond Averages: Learning with Annotator Disagreement in STS

> **Code for the paper:**
> **Alejandro Benito-Santos & Adrián Ghajari (2025).
> “Beyond Averages: Learning with Annotator Disagreement in STS.”
> *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025).*

Sentence Textual Similarity (STS) pipeline that goes beyond single averaged scores by modeling full ordinal label distributions (0–5). The pipeline:

* Loads and preprocesses the STS dataset with per‑item Krippendorff’s alpha and stratified splits by agreement.
* Runs a pretrained CrossEncoder baseline and converts scalars to distributions with a heteroscedastic TruncNormal head.
* Trains a distributional CrossEncoder (6 classes) using ordinal/distance‑aware losses (e.g., OrdinalLogLoss).
* Evaluates with JSD/EMD (with/without a distance matrix), correlations on STS‑B and STS‑15, and reports per‑strata metrics.

## Requirements

* Python 3.10+
* CUDA‑enabled GPU (the code moves models to `cuda` explicitly) and recent NVIDIA drivers.
* PyTorch 2.6 with a matching CUDA build.
* See `requirements.txt` for exact versions.

Tip: If installing PyTorch separately, follow the selector at pytorch.org for your platform, then `pip install -r requirements.txt` excluding extras you don’t need.

## Data

STS‑15 disaggregated dataset downloaded from [here](http://ixa2.si.ehu.es/stswiki/images/2/21/STS2015-en-rawdata-scripts.zip).

The pipeline computes Krippendorff’s alpha per item and a normalized ordinal distance matrix used by the losses and EMD.

## How To Run

```bash
python pipelineV2.py --base_dir output
```

* Creates a timestamped experiment folder under `output/exp-YYYY-MM-DD_HH-MM-SS/` with subfolders:

  * `trained_models/` for checkpoints and the final saved model
  * `evaluation/` for evaluation CSVs
  * CSV dumps of splits: `train.csv`, `val.csv`, `test.csv`
  * `splits_stats.txt` with stratified counts
* Downloads/caches the STS‑B test split via `datasets` for correlation reporting.

## Pipeline Stages (high level)

* Preprocess: loads `data/text.clean`, computes per‑item agreement (Krippendorff’s alpha), assigns terciles, and performs 60/20/20 stratified splits.
* Baseline: `cross-encoder/stsb-roberta-large` predicts scalar similarity; a per‑strata TruncNormal head converts scalars to distributions; evaluated with JSD/EMD.
* Train (Soft): CrossEncoder with 6 outputs (`roberta-large`) trained with distributional loss (default `OrdinalLogLoss`).
* Calibrate (Soft): δ‑aware temperature scaling per (mode, tercile) to minimize EMD on validation, then evaluate on test.
* Train (Hard): Standard scalar regression CrossEncoder; evaluated directly and with a TruncNormal head for distributional metrics.
* Correlations: STS‑B (Pearson/Spearman) and STS‑15 (computed from probabilistic expectations) are reported for baselines and trained models.

## Outputs & Metrics

* Global metrics: average `JSD`, `EMD_with` (uses distance matrix), and `EMD_no` (without distance matrix). Printed and saved as CSV.
* Slice metrics: the same metrics per `(mode, tercile)` slice; summary tables printed and saved as CSV.
* Correlations: STS‑B and STS‑15 Pearson/Spearman; per‑strata RMSE tables for STS‑15 are printed and saved.
* Consolidated CSVs are written under the experiment `evaluation/` folder; additional summary CSVs are saved in a `calibration` subfolder that the script creates.


## Project Structure

* `pipelineV2.py`: Orchestrates the full pipeline, CLI, outputs.
* `data/text.clean`: Input dataset (see format above).
* `preprocess/`: Data loader, Krippendorff utilities.
* `training/`: Custom distributional losses and evaluators for Sentence‑Transformers CrossEncoder.
* `evaluate_ord/`: JSD/EMD metrics and utilities for converting model logits to calibrated scores.

## Notes & Tips

* **GPU required:** the code calls `.to("cuda")` for models. For CPU‑only runs, adapt those calls to `cpu` and expect significant slowdown.
* **Hugging Face datasets:** STS‑B test is fetched with `datasets`. If offline, ensure it’s cached locally or set `HF_DATASETS_OFFLINE=1`.
* **Reproducibility:** seeds are set for Python/NumPy/PyTorch and cuDNN deterministic flags.

## Example Session

1. Create/activate a virtual env
   `python -m venv .venv && source .venv/bin/activate`
2. Install deps
   `pip install --upgrade pip`
   Install PyTorch matching your CUDA, then `pip install -r requirements.txt`
3. Run
   `python pipelineV2.py --base_dir output`
4. Inspect results
   Check `output/exp-*/evaluation/` and printed tables; CSVs include global and per‑slice metrics.

## License

MIT License. See `LICENSE` file for details.

## Citation

If you use this repository in your research, please cite:

```bibtex
@inproceedings{benito-ghajari-2025-beyond,
  title        = {Beyond Averages: Learning with Annotator Disagreement in STS},
  author       = {Benito-Santos, Alejandro and Ghajari, Adri{\'a}n},
  booktitle    = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025)},
  year         = {2025}
}
```