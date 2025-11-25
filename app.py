from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Body

# Loaders
from loaders.monthlyperfomance_loader import MonthlyPerformanceLoader
from loaders.lifeexpectancy_loader import LifeExpectancyLoader
from loaders.tire_index_loader import TireIndexLoader


# Predict function for lifeexpectancy
from projects.lifeexpectancy.predict_llanta import predict_llanta

# Settings (must contain S3 credentials + DB credentials)
from core.settings import settings

from projects.tire_index.predict_tireindex import predict_tire_index



# ---------------------------------------------
# AVAILABLE PROJECTS
# ---------------------------------------------
projects_available = {
    "monthlyperformance": MonthlyPerformanceLoader,
    "lifeexpectancy": LifeExpectancyLoader,
    "tire_index": TireIndexLoader
}

# GLOBALS IN MEMORY
loaded_projects: Dict[str, Any] = {}
loaded_models: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------
# FASTAPI APP
# ---------------------------------------------
app = FastAPI(
    title="Kaltire Central Inference API",
    description="Multi-project inference service for ML models stored in S3",
    version="1.0.0",
)


# ---------------------------------------------
# Helper — resolve model name for monthlyperformance
# ---------------------------------------------
def resolve_model_name(project: str, alias: str) -> str:
    models_for_project = loaded_models.get(project, {})

    if alias in models_for_project:
        return alias

    candidates = [
        f"kms_prediction_model_{alias}",
        f"{project}_{alias}",
    ]

    for c in candidates:
        if c in models_for_project:
            return c

    raise KeyError(
        f"Model alias '{alias}' not found for project '{project}'. "
        f"Available: {list(models_for_project.keys())}"
    )


# ---------------------------------------------
# Startup: load all projects
# ---------------------------------------------
@app.on_event("startup")
def startup_load_all():
    bucket = settings.S3_BUCKET

    for project, LoaderClass in projects_available.items():
        loader = LoaderClass(bucket)
        loader.load_models()  # LIFEEXP + MONTHLY PERFORMANCE

        loaded_projects[project] = loader

        # monthlyperformance uses loader.models (multiple)
        # lifeexpectancy has only model + preprocessor
        if hasattr(loader, "models"):
            loaded_models[project] = loader.models
        else:
            loaded_models[project] = {"default": loader.model}

    print("🔥 Projects loaded:", list(loaded_projects.keys()))


# ---------------------------------------------
# HEALTH
# ---------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "projects_loaded": list(loaded_projects.keys())}


# ---------------------------------------------
# GET PROJECTS
# ---------------------------------------------
@app.get("/projects")
def list_projects():
    return {"projects": list(loaded_projects.keys())}


# ---------------------------------------------
# GET MODELS PER PROJECT (monthlyperformance only)
# ---------------------------------------------
@app.get("/projects/{project}/models")
def list_models(project: str):
    if project not in loaded_models:
        raise HTTPException(404, f"Project '{project}' not found")

    return {
        "project": project,
        "models": list(loaded_models[project].keys())
    }


# ===========================================================
# 📌 PREDICT for MONTHLYPERFORMANCE (standard ML pipeline)
# ===========================================================
@app.post("/predict/{project}/{model}")
def predict_model(
    project: str,
    model: str,
    payload: Dict[str, Any] = Body(...)
):
    if project != "monthlyperformance":
        raise HTTPException(400, "Este endpoint solo es para monthlyperformance.")

    if project not in loaded_models:
        raise HTTPException(404, f"Project '{project}' not found")

    try:
        internal_model_name = resolve_model_name(project, model)
    except KeyError as e:
        raise HTTPException(404, str(e))

    model_pack = loaded_models[project][internal_model_name]
    model_obj = model_pack["model"]
    config = model_pack["config"]

    feature_order = config.get("feature_names") or config.get("features")

    if not feature_order:
        raise HTTPException(500, "Este modelo no define orden de features.")

    try:
        X = [[payload[f] for f in feature_order]]
    except KeyError as e:
        raise HTTPException(
            400,
            f"Missing feature {str(e)}. Required: {feature_order}"
        )

    try:
        prediction = model_obj.predict(X)
    except Exception as e:
        raise HTTPException(500, f"Inference error: {str(e)}")

    return {
        "project": project,
        "model_requested": model,
        "model_used": internal_model_name,
        "features_used": feature_order,
        "prediction": round(float(prediction[0]), 2),
    }


# ===========================================================
# 📌 PREDICT for LIFEEXPECTANCY (SQL + preprocessing + ML)
# ===========================================================
@app.post("/predict/lifeexpectancy")
def predict_lifeexpectancy(
    payload: Dict[str, Any] = Body(...)
):
    nro = payload.get("nrointerno")
    if not nro:
        raise HTTPException(400, "Debe enviar nrointerno")

    loader = loaded_projects["lifeexpectancy"]
    db_settings = {
        "host": settings.DB_HOST,
        "port": settings.DB_PORT,
        "user": settings.DB_USER,
        "password": settings.DB_PASSWORD,
        "database": settings.DB_NAME,
    }

    result = predict_llanta(nrointerno=nro, loader=loader, db_settings=db_settings)
    return result


# ---------------------------------------------
# RELOAD A SINGLE PROJECT
# ---------------------------------------------
@app.post("/reload/{project}")
def reload_project(project: str):
    if project not in projects_available:
        raise HTTPException(404, f"Project '{project}' not found.")

    bucket = settings.S3_BUCKET
    LoaderClass = projects_available[project]

    loader = LoaderClass(bucket)
    loader.load_models()

    loaded_projects[project] = loader

    if hasattr(loader, "models"):
        loaded_models[project] = loader.models
    else:
        loaded_models[project] = {"default": loader.model}

    return {
        "status": "reloaded",
        "project": project,
        "models": list(loaded_models[project].keys())
    }


# ---------------------------------------------
# RELOAD ALL PROJECTS
# ---------------------------------------------
@app.post("/reload-all")
def reload_all():
    startup_load_all()
    return {
        "status": "all_reloaded",
        "projects": {
            p: list(m.keys()) for p, m in loaded_models.items()
        }
    }


# ===========================================================
# 📌 PREDICT for Tire_index (SQL + preprocessing + ML)
# ===========================================================

@app.post("/predict/tireindex")
def predict_tireindex_endpoint(payload: Dict[str, Any]):
    loader = loaded_projects["tire_index"]      # tu TireIndexLoader
    model = loader.model                        # pipeline sklearn
    model_config = loader.config                # dict del config.json

    db_settings = {
        "host": settings.DB_HOST,
        "port": settings.DB_PORT,
        "user": settings.DB_USER,
        "password": settings.DB_PASSWORD,
        "database": settings.DB_NAME,
    }

    result = predict_tire_index(payload, model, model_config, db_settings)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result)

    return result

# ---------------------------------------------
# Local Development Entry
# ---------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
