"""Private ATHENA benchmark API: development/prototype use, not competition hardened."""
import os
import secrets
import threading
import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from ternary_private_simulator import TernarySpectroscopySimulator, true_spectrum, ENERGY

app = FastAPI(title='ATHENA blind spectral benchmark', docs_url=None, redoc_url=None)
LOCK = threading.RLock()
SESSIONS = {}  # Development only; not persistent; use one worker
BUDGET_1 = float(os.environ.get('HACKATHON_BUDGET', '30'))
BUDGET_2 = float(os.environ.get('HACKATHON_CUBE_BUDGET', '900'))
TARGET_X, TARGET_Y = float(os.environ.get('TARGET_X', '.30')), float(os.environ.get('TARGET_Y', '.40'))
GRID_N = int(os.environ.get('CUBE_GRID_N', '15'))
if not 2 <= GRID_N <= 40:
    raise ValueError('CUBE_GRID_N must be 2..40')
CUBE_COORDS = np.array([(i/(GRID_N-1),j/(GRID_N-1)) for i in range(GRID_N) for j in range(GRID_N-i)],dtype=float)

class SessionRequest(BaseModel):
    challenge: int = Field(default=1, ge=1, le=2)

class PixelRequest(BaseModel):
    wavelength: float = Field(ge=0, le=10)
    time: float = Field(gt=0)
    x: float | None = None
    y: float | None = None

class FullRequest(BaseModel):
    time: float = Field(gt=0)
    x: float | None = None
    y: float | None = None

class SpectrumSubmission(BaseModel):
    energy: list[float]
    predicted_intensity: list[float]

class CubeSubmission(BaseModel):
    predictions: list[list[float]]

@app.get('/info')
def info():
    return {'tasks':['single-spectrum','ternary-spectral-cube'], 'composition':{'x':TARGET_X,'y':TARGET_Y},
            'time_budget':BUDGET_1, 'cube_time_budget':BUDGET_2,
            'wavelength_range':[float(ENERGY[0]), float(ENERGY[-1])], 'channels':len(ENERGY),
            'cube_grid':CUBE_COORDS.tolist(), 'cube_shape':[len(CUBE_COORDS),len(ENERGY)],
            'full_spectrum_measurement_allowed':False}

@app.post('/session')
def new_session(request:SessionRequest):
    challenge=request.challenge
    budget=BUDGET_1 if challenge == 1 else BUDGET_2
    token=secrets.token_urlsafe(24)
    with LOCK:
        SESSIONS[token]={'sim':TernarySpectroscopySimulator(seed=secrets.randbits(32),time_budget=budget),
                         'challenge':challenge,'submitted':False}
    return {'session_id':token,'challenge':challenge,'time_budget':budget}

def get_session(session_id):
    if not session_id or session_id not in SESSIONS:
        raise HTTPException(401,'Invalid session ID')
    session=SESSIONS[session_id]
    if session['submitted']:
        raise HTTPException(403,'Episode already submitted')
    return session

def location(req,challenge):
    if challenge == 1:
        if req.x is not None or req.y is not None:
            raise HTTPException(400,'Challenge 1 uses a fixed composition; omit x and y')
        return TARGET_X,TARGET_Y
    if req.x is None or req.y is None:
        raise HTTPException(400,'Challenge 2 requires x and y')
    x,y=float(req.x),float(req.y)
    if not (np.isfinite(x) and np.isfinite(y) and x>=0 and y>=0 and x+y<=1+1e-12):
        raise HTTPException(400,'Composition must have x>=0, y>=0, x+y<=1')
    return x,y

@app.post('/pixel')
def pixel(req:PixelRequest,x_session_id:str|None=Header(None)):
    with LOCK:
        s=get_session(x_session_id)
        x,y=location(req,s['challenge'])
        try:
            return s['sim'].draw_pixel(x,y,req.time,req.wavelength)
        except (RuntimeError,ValueError) as exc:
            raise HTTPException(400,str(exc)) from exc

@app.post('/spectrum')
def spectrum(req:FullRequest,x_session_id:str|None=Header(None)):
    # Intentionally disabled: full spectrum at cost of one pixel would undercut sparse wavelength benchmark.
    raise HTTPException(403,'Full-spectrum acquisition disabled for benchmark tasks; use /pixel')

@app.get('/budget')
def budget(x_session_id:str|None=Header(None)):
    with LOCK:
        sim=get_session(x_session_id)['sim']
        return {'time_used':sim.used_time,'time_remaining':max(0.,sim.time_budget-sim.used_time),
                'measurements':len(sim.history)}

def validate_energy(energy):
    e=np.asarray(energy,dtype=float)
    if e.shape != ENERGY.shape or not np.all(np.isfinite(e)) or not np.allclose(e,ENERGY,atol=1e-7,rtol=0):
        raise HTTPException(400,'Invalid spectral grid')

@app.post('/submit')
def submit(req:SpectrumSubmission,x_session_id:str|None=Header(None)):
    with LOCK:
        s=get_session(x_session_id)
        if s['challenge']!=1:
            raise HTTPException(400,'This endpoint is for Challenge 1')
        validate_energy(req.energy)
        pred=np.asarray(req.predicted_intensity,dtype=float)
        if pred.shape!=ENERGY.shape or not np.all(np.isfinite(pred)):
            raise HTTPException(400,'Provide 512 finite predicted intensities')
        truth=true_spectrum(TARGET_X,TARGET_Y,ENERGY)
        rmse=float(np.sqrt(np.mean((pred-truth)**2)))
        s['submitted']=True
        return {'accepted':True,'rmse':rmse,'time_used':s['sim'].used_time,'measurements':len(s['sim'].history)}

@app.post('/submit-cube')
def submit_cube(req:CubeSubmission,x_session_id:str|None=Header(None)):
    with LOCK:
        s=get_session(x_session_id)
        if s['challenge']!=2:
            raise HTTPException(400,'This endpoint is for Challenge 2')
        pred=np.asarray(req.predictions,dtype=float)
        if pred.shape!=(len(CUBE_COORDS),len(ENERGY)) or not np.all(np.isfinite(pred)):
            raise HTTPException(400,f'Expected finite matrix shape {(len(CUBE_COORDS),len(ENERGY))}')
        reference=np.stack([true_spectrum(float(x),float(y),ENERGY) for x,y in CUBE_COORDS])
        rmse=float(np.sqrt(np.mean((pred-reference)**2)))
        s['submitted']=True
        return {'accepted':True,'rmse':rmse,'time_used':s['sim'].used_time,'measurements':len(s['sim'].history)}
