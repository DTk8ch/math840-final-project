# MATH840 Final Project

This project forecasts 48 hours of electricity generation, with a 95% interval.
The fixed final file is `TKACH_project.csv`. The report is `report.ipynb`.

Use Python 3.12. Run commands from this folder.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/run_report.py --isolate
```

You can also open `report.ipynb` in Jupyter and choose **Run All**. The notebook
reads `train.csv`, `weather_forecast.csv`, the final CSV and `results/`.
It does not train models. Its charts and tables are saved in the notebook.
The script runs all cells in a fresh Jupyter in-process kernel. The isolation
check uses only the notebook, data and saved results in a temporary folder.

To reproduce the unchanged forecast:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python forecast.py
```

`src/` contains the fixed model code and settings. Completed historical
forecasts provide interval calibration and adaptive weights. The final point
models are fitted again. Missing historical caches can be rebuilt from the data.
`_project.ipynb` is the original class template used for scoring and checks.

Two test blocks and a recent diagnostic are reported. October 2026 is not a
new independent test. Weather errors are simulated assumptions. Some metrics
and coverage results are worse in some periods. Read the limits in the report.

ChatGPT helped with planning and task analysis. Codex helped write, check and
improve the code, and prepare the report. Failed experiments are included.
