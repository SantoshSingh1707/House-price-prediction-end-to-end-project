<!--
Auto-generated README — focus: quick start for developers.
-->
# House Price Prediction — End-to-End

[![Build Status](https://img.shields.io/badge/build-manual-blue.svg)](https://github.com)
[![License](https://img.shields.io/badge/license-See%20LICENSE-lightgrey.svg)](LICENSE)

A reproducible end-to-end machine learning project that trains, evaluates, and serves a house-price prediction model. The repository uses a ZenML + MLflow-oriented pipeline to manage experiments and artifacts.

## Key features

- Reproducible training pipeline ([pipelines/training_pipeline.py](pipelines/training_pipeline.py))
- Data ingestion, splitting, feature engineering and model evaluation steps ([steps](steps))
- Local experiment tracking with MLflow and local model serving support
- Example prediction client for testing a running model server ([sample_predict.py](sample_predict.py))

## Quick start

These steps get a developer from a fresh clone to a trained model and a running prediction example on Windows.

Prerequisites

- Python 3.8+ installed
- Git
- (Optional) Docker, if you prefer containerized serving

1. Clone the repository

```bash
git clone https://github.com/SantoshSingh1707/House-price-prediction-end-to-end-project

cd House-Price-prediction-End-to-End-Project
```

2. Create and activate a virtual environment (Windows example)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

3. Run the training pipeline

```bash
python run_pipeline.py
```

This will execute the ZenML/ML pipeline defined in [pipelines/training_pipeline.py](pipelines/training_pipeline.py) and persist artifacts under `artifacts/` and `mlruns_local/`.

4. Inspect runs with MLflow UI

Get the MLflow tracking URI used by the pipeline and start the UI:

```bash
python -c "from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri; print(get_tracking_uri())"
mlflow ui --backend-store-uri '<TRACKING_URI_FROM_ABOVE>'
```

5. Serve the model (example)

After a successful run, locate the model under `mlruns_local/` and serve it with MLflow:

```bash
mlflow models serve -m ./mlruns_local/<RUN_ID>/artifacts/model -p 5001 --no-conda
```

6. Test predictions

With the model server running at `http://127.0.0.1:5001`, run the example client:

```bash
python sample_predict.py
```

The script `sample_predict.py` posts a JSON payload to the MLflow model endpoint and prints the response.

## Project layout

- [src](src) — core modules (data ingestion, feature engineering, model building, evaluation)
- [pipelines](pipelines) — pipeline definitions and orchestration
- [steps](steps) — pipeline step implementations used by ZenML
- [artifacts](artifacts) — generated artifacts and intermediate outputs (do not commit large binaries)
- [mlruns_local](mlruns_local) — local MLflow run artifacts
- [requirements.txt](requirements.txt) — Python dependencies
- [run_pipeline.py](run_pipeline.py) — entry point to run training pipeline
- [run_deployment.py](run_deployment.py) — deployment helper (if present)
- [sample_predict.py](sample_predict.py) — example prediction client

## Where to get help

- Open an issue or pull request on this repository.
- For ZenML or MLflow-specific questions consult their docs:
  - https://docs.zenml.io
  - https://mlflow.org

## Contributing

Contributions are welcome. Please open issues for bugs or feature requests and submit pull requests against the `main` branch. See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines (if present).

## Maintainers

Maintained by the repository owners — please use GitHub Issues or Pull Requests to get in touch.

## License

See the [LICENSE](LICENSE) file in this repository for license details.

## A note on data and artifacts

- The dataset used in analysis is available under `extracted_data/AmesHousing.csv`.
- Large artifacts and ML runs are placed under `artifacts/` and `mlruns_local/`; add them to `.gitignore` if you do not want to push them.

---
