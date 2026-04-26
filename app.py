# app.py
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import numpy as np
import time
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif, SequentialFeatureSelector

app = FastAPI(title="TP2 - Sélection d'Attributs API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

@app.get("/")
def serve_index():
    return FileResponse(BASE_DIR / "index.html")

@app.get("/health")
def health():
    return {"status": "ok"}

# Initialisation au démarrage
digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
CACHE = {}

# Modèles Pydantic
class Phase1Response(BaseModel):
    n_samples: int
    n_features: int
    n_classes: int
    train_shape: list
    test_shape: list

class Phase2Response(BaseModel):
    f_scores: list
    selected_pixels: list
    feature_names: list

class Phase3Response(BaseModel):
    selected_pixels: list
    feature_names: list
    cv_score: float

class Phase4Response(BaseModel):
    selected_pixels: list
    feature_names: list
    cv_score: float

class MethodResult(BaseModel):
    accuracy: float
    n_features: int
    time_sec: float

class Phase5Response(BaseModel):
    baseline: MethodResult
    filter: MethodResult
    forward: MethodResult
    backward: MethodResult

@app.get("/api/phase1", response_model=Phase1Response)
def api_phase1():
    return Phase1Response(
        n_samples=X.shape[0],
        n_features=X.shape[1],
        n_classes=len(np.unique(y)),
        train_shape=list(X_train.shape),
        test_shape=list(X_test.shape)
    )

@app.get("/api/phase2", response_model=Phase2Response)
def api_phase2():
    if "phase2" not in CACHE:
        start_time = time.time()
        selector = SelectKBest(score_func=f_classif, k=10)
        selector.fit(X_train, y_train)
        mask = selector.get_support()
        CACHE["phase2"] = {
            "f_scores": selector.scores_.tolist(),
            "selected_pixels": np.where(mask)[0].tolist(),
            "feature_names": [f"px_{i}" for i in np.where(mask)[0]],
            "mask": mask,
            "time": time.time() - start_time
        }
    return Phase2Response(**CACHE["phase2"])

@app.get("/api/phase3", response_model=Phase3Response)
def api_phase3():
    if "phase3" not in CACHE:
        print("Forward Selection en cours...")
        start_time = time.time()
        knn = KNeighborsClassifier(n_neighbors=5)
        sfs = SequentialFeatureSelector(direction='forward', n_features_to_select=10, cv=5, estimator=knn)
        sfs.fit(X_train, y_train)
        mask = sfs.get_support()
        CACHE["phase3"] = {
            "selected_pixels": np.where(mask)[0].tolist(),
            "feature_names": [f"px_{i}" for i in np.where(mask)[0]],
            "cv_score": 0.0,
            "mask": mask,
            "time": time.time() - start_time
        }
    return Phase3Response(**CACHE["phase3"])

@app.get("/api/phase4", response_model=Phase4Response)
def api_phase4():
    if "phase4" not in CACHE:
        print("Backward Elimination en cours...")
        start_time = time.time()
        knn = KNeighborsClassifier(n_neighbors=5)
        sfs = SequentialFeatureSelector(direction='backward', n_features_to_select=10, cv=5, estimator=knn)
        sfs.fit(X_train, y_train)
        mask = sfs.get_support()
        CACHE["phase4"] = {
            "selected_pixels": np.where(mask)[0].tolist(),
            "feature_names": [f"px_{i}" for i in np.where(mask)[0]],
            "cv_score": 0.0,
            "mask": mask,
            "time": time.time() - start_time
        }
    return Phase4Response(**CACHE["phase4"])

@app.get("/api/phase5", response_model=Phase5Response)
def api_phase5():
    # S'assurer que les phases 2, 3, 4 ont été exécutées
    if "phase2" not in CACHE: api_phase2()
    if "phase3" not in CACHE: api_phase3()
    if "phase4" not in CACHE: api_phase4()
    
    def evaluate(mask):
        start_t = time.time()
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X_train[:, mask], y_train)
        acc = clf.score(X_test[:, mask], y_test)
        t = time.time() - start_t
        return acc, t
    
    # Baseline
    acc_base, t_base = evaluate(np.ones(64, dtype=bool))
    
    # Models
    acc_filter, _ = evaluate(CACHE["phase2"]["mask"])
    acc_fwd, _ = evaluate(CACHE["phase3"]["mask"])
    acc_bwd, _ = evaluate(CACHE["phase4"]["mask"])
    
    return Phase5Response(
        baseline=MethodResult(accuracy=acc_base * 100, n_features=64, time_sec=0.01), # Temps indicatif
        filter=MethodResult(accuracy=acc_filter * 100, n_features=10, time_sec=CACHE["phase2"]["time"]),
        forward=MethodResult(accuracy=acc_fwd * 100, n_features=10, time_sec=CACHE["phase3"]["time"]),
        backward=MethodResult(accuracy=acc_bwd * 100, n_features=10, time_sec=CACHE["phase4"]["time"])
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)