# MLOps Platform Architecture

## Overview

The platform follows a microservice architecture (MSA) so each capability can evolve independently, scale according to demand, and be deployed on its own cadence. Every functional area is encapsulated in a directory that hosts a service-specific codebase, infrastructure definitions, and documentation.

```
├── data_acquisition/
├── data_preparation/
├── model_training/
├── model_evaluation/
├── model_serving/
├── common_utils/
├── platform_orchestration/
├── dtaq_func_pdf_to_csv/
└── api_gateway/
```

## Core Services

- **Data Acquisition**
  - Ingests raw data from streaming and batch sources.
  - Publishes validated datasets to shared storage (e.g., object store, data warehouse).
  - Exposes ingestion metrics via service-level telemetry.

- **Data Preparation**
  - Runs feature engineering, data quality checks, and dataset versioning.
  - Produces curated training/evaluation-ready datasets.
  - Uses message/event triggers from acquisition or orchestration layers.

- **Model Training**
  - Orchestrates training jobs across compute backends (Kubernetes, serverless, or managed ML services).
  - Logs artifacts and metrics to the model registry and experiment tracker.
  - Provides APIs/CLIs for triggering ad-hoc or scheduled training runs.

- **Model Evaluation**
  - Performs offline evaluation, bias/fairness checks, and champion/challenger comparisons.
  - Generates reports and gatekeeper signals for deployment decisions.
  - Interfaces with model serving to validate post-deployment performance.

- **Model Serving**
  - Serves models in real-time or batch modes with autoscaling.
  - Integrates with configuration management for model version selection.
  - Emits live performance/latency metrics to the monitoring stack.

- **Common Utils**
  - Shared libraries (feature definitions, schema validation, client SDKs).
  - Contracts and interfaces adopted by the other services to reduce coupling.
  - Utilities should remain backward compatible to avoid ripple effects.

- **Platform Orchestration**
  - Workflow schedulers (e.g., Airflow, Argo) for cross-service pipelines.
  - Automation for retraining triggers, evaluation gates, and deployment rollouts.
  - Centralized observability dashboards connecting telemetry from all services.

- **dtaq_func_pdf_to_csv**
  - Specialized extraction service that converts PDF-based tables to normalized CSV datasets.
  - Publishes structured outputs that downstream preparation pipelines can consume.
  - Re-uses shared parsing/util utilities from `common_utils`.

- **API Gateway**
  - FastAPI-based aggregation layer that exposes unified endpoints for platform consumers.
  - Provides a façade over the domain services and brokers requests to their public APIs.
  - Central place to enforce authentication, request validation, and response normalization.
  - Development server auto-selects an available port starting at `8000`, reducing clashes with other local services.

## Cross-Cutting Concerns

- **API Contracts**: Define gRPC/REST/OpenAPI specs per service. Store shared schemas in `common_utils`.
- **Messaging**: Use a message bus (Kafka, Pub/Sub, etc.) for event-driven handoffs between services.
- **Storage**: Adopt shared data storage patterns (bronze/silver/gold) to keep datasets discoverable and versioned.
- **Security**: Centralized authentication/authorization layer, secrets management, and audit logging per service.
- **CI/CD**: Independent pipelines per service plus an umbrella workflow that validates end-to-end pipelines.

## Directory Layout Pattern

Each service directory uses a consistent skeleton so teams can add new functionality by cloning the layout:

```
service-name/
├── service/
│   ├── src/
│   └── tests/
├── configs/
├── docs/
└── Dockerfile (or deployment manifests)
```

Create new services by duplicating this pattern at the repository root, wiring them into orchestration workflows, and registering their API contracts within `common_utils`.

## API Gateway Layout

```
api_gateway/
├── app/
│   ├── api/
│   │   └── routes/
│   ├── core/
│   ├── services/
│   └── main.py
└── tests/
```

The gateway loads configuration from environment variables (see `api_gateway/app/config.py`) to discover downstream service URLs. Each router module exposes FastAPI endpoints that will later proxy calls to the respective microservice once implementations are available.
