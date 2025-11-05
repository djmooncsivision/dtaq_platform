# MLOps Platform Skeleton

This repository provides a microservice-oriented skeleton for an MLOps platform. Each functional area is isolated in its own top-level directory so that new capabilities can be added by introducing additional sibling directories.

## Top-Level Services

- `data_acquisition/` – microservice for sourcing, validating, and ingesting raw data from external systems.
- `data_preparation/` – pipelines for cleansing, feature engineering, and dataset version management.
- `model_training/` – training orchestration, experiment tracking hooks, and reusable pipelines.
- `model_evaluation/` – offline evaluation flows, metric computation, and report generation.
- `model_serving/` – real-time/batch serving endpoints, model registry integration, and deployment manifests.
- `common_utils/` – shared libraries, schemas, and contracts used across services.
- `platform_orchestration/` – workflow scheduling, monitoring, and cross-service automation.
- `dtaq_func_pdf_to_csv/` – conversion pipeline that extracts tabular data from PDFs and emits normalized CSV outputs.
- `api_gateway/` – FastAPI entry point that exposes consolidated APIs and proxies calls to the underlying services.
- `docs/` – design references, decision records, and high-level runbooks.

Refer to `docs/architecture.md` for a deeper architectural overview and recommended patterns for expanding the platform.

## Running the API Gateway

1. Create and activate a Python virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Serve the API (with auto-reload in development): `make serve`

By default the gateway starts on `PORT=8000`, but if that port is occupied it automatically increments to the next open port (up to 100 ports ahead). The current port is reported in the terminal when the server starts. Adjust service URLs in `api_gateway/app/config.py` to point at deployed microservices or local mocks.
