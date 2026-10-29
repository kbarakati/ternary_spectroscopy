"""Organizer-only development API. Deploy behind HTTPS with rate limiting and persistent storage for a real event."""
import os
import secrets
import threading
import numpy as np
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
from ternary_private_simulator import TernarySpectroscopySimulator, true_spectrum, ENERGY

app = FastAPI(title="ATHENA spectral measurement service", docs_url=None, redoc_url=None)
LOCK = threading.Lock()
SESSIONS = {}
BUDGET = float(os.environ.get("HACKATHON_BUDGET", "30"))
TARGET_X = float(os.environ.get("TARGET_X", "0.30"))
TARGET_Y = float(os.environ.get("TARGET_Y", "0.40"))

class PixelRequest(BaseModel):
    wavelength: float = Field(ge=0, le=10)
    time: float = Field(gt=0)

class FullRequest(BaseModel):
    time: float = Field(gt=0)

class Submission(BaseModel):
    energy: list[float]
    predicted_intensity: list[float]

@app.get("/info")
def info():
    return {"task": "single-spectrum", "composition": {"x": TARGET_X,"y": TARGET_Y},
            "time_budget": BUDGET, "wavelength_range": [0.,10.], "channels": len(ENERGY)}

@app.post("/session")
def create_session():
    token = secrets.token_urlsafe(24)
    # 30+ pixel requests possible depending on choice of acquisition duration
    with LOCK:
        SESSIONS[token] = {"sim": TernarySpectroscopySimulator(seed=secrets.randbits(32), time_budget=BUDGET), "submitted": False}
    return {"session_id": token, "time_budget": BUDGET}

def get_session(session_id):
    if not session_id or session_id not in SESSIONS:
        raise HTTPException(status_code=401, detail="Invalid session ID")
    s = SESSIONS[session_id]
    if s["submitted"]:
        raise HTTPException(status_code=403, detail="Session already submitted")
    return s

@app.post("/pixel")
def pixel(req: PixelRequest, x_session_id: str | None = Header(default=None)):
    with LOCK:
        sim = get_session(x_session_id)["sim"]
        try: out = sim.draw_pixel(TARGET_X, TARGET_Y, req.time, req.wavelength)
        except (ValueError, RuntimeError) as e: raise HTTPException(status_code=400, detail=str(e))
    return out

@app.post("/spectrum")
def spectrum(req: FullRequest, x_session_id: str | None = Header(default=None)):
    with LOCK:
        sim = get_session(x_session_id)["sim"]
        try: out = sim.draw_noisy_spectrum(TARGET_X,TARGET_Y,req.time)
        except (ValueError, RuntimeError) as e: raise HTTPException(status_code=400, detail=str(e))
    return {**out, "energy": np.asarray(out["energy"]).tolist(), "spectrum": np.asarray(out["spectrum"]).tolist()}

@app.get("/budget")
def budget(x_session_id: str | None = Header(default=None)):
    with LOCK:
        sim = get_session(x_session_id)["sim"]
        return {"time_used": sim.used_time, "time_remaining": max(0.,sim.time_budget-sim.used_time)}

@app.post("/submit")
def submit(req: Submission, x_session_id: str | None = Header(default=None)):
    with LOCK:
        session=get_session(x_session_id)
        if len(req.energy)!=len(ENERGY) or len(req.predicted_intensity)!=len(ENERGY):
            raise HTTPException(status_code=400, detail="Submit exactly 512 channels")
        energy=np.asarray(req.energy,dtype=float)
        pred=np.asarray(req.predicted_intensity,dtype=float)
        if not (np.all(np.isfinite(pred)) and np.allclose(energy,ENERGY,rtol=0,atol=1e-7)):
            raise HTTPException(status_code=400, detail="Invalid energy grid or nonfinite prediction")
        sim=session["sim"]
        reference=true_spectrum(TARGET_X,TARGET_Y,ENERGY)
        rmse=float(np.sqrt(np.mean((pred-reference)**2)))
        session["submitted"]=True
        return {"accepted": True, "rmse": rmse, "time_used": sim.used_time,
                "time_budget": sim.time_budget, "measurements": len(sim.history)}
