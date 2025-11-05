from fastapi import FastAPI

from . import data_acquisition, data_preparation, model_training, model_evaluation, model_serving, pdf_to_csv


def include_all_routers(app: FastAPI) -> None:
    """Register routers for each domain-specific microservice."""
    app.include_router(data_acquisition.router, prefix="/data-acquisition", tags=["data-acquisition"])
    app.include_router(data_preparation.router, prefix="/data-preparation", tags=["data-preparation"])
    app.include_router(model_training.router, prefix="/model-training", tags=["model-training"])
    app.include_router(model_evaluation.router, prefix="/model-evaluation", tags=["model-evaluation"])
    app.include_router(model_serving.router, prefix="/model-serving", tags=["model-serving"])
    app.include_router(pdf_to_csv.router, prefix="/pdf-to-csv", tags=["pdf-to-csv"])
