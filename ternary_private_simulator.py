"""ORGANIZER ONLY. Never distribute this module to participants."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

SEED = 2202
ENERGY = np.linspace(0.0, 10.0, 512)
PHASE_NAMES = ['A rich','B rich','C rich','AB compound','BC compound','AC compound','Ternary solution']
PHASE_COLORS = ['#d85f57','#4b96cf','#e8b34f','#8266aa','#46a68c','#c679ad','#b2d66a','#505765']
COEXIST_THRESHOLD_PREVIOUS = 0.32
COEXIST_THRESHOLD = 1.5 * COEXIST_THRESHOLD_PREVIOUS  # 0.48: 50% higher score-gap cutoff
CENTERS = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],
                    [.55,.45,0.],[0.,.48,.52],[.5,0.,.5],[.33,.34,.33]])
WIDTHS = np.array([.29,.29,.29,.18,.18,.18,.30])

def composition(x, y):
    x, y = float(x), float(y)
    if not (np.isfinite(x) and np.isfinite(y) and x >= 0 and y >= 0 and x+y <= 1+1e-12):
        raise ValueError('Invalid ternary composition: require x>=0, y>=0, x+y<=1.')
    return np.array([x,y,max(0.,1-x-y)])

def phase_scores(x,y):
    a,b,c = composition(x,y)
    q=np.array([a,b,c]); d2=np.sum((CENTERS-q)**2,axis=1)
    scores=-d2/(2*WIDTHS**2)
    scores[6]+=.58*np.sin(12*a+3*b)*np.cos(11*b-2*c)+.18*np.sin(22*a*b)
    for k in (3,4,5): scores[k]+=.24*np.sin(18*a+13*b+k)
    return scores

def phase_state(x,y):
    '''Returns category, 7-dimensional fractions, top-two gap and raw hidden scores.'''
    s=phase_scores(x,y)
    order=np.argsort(s)[::-1]
    gap=float(s[order[0]]-s[order[1]])
    primary=int(order[0]); secondary=int(order[1])
    label=7 if gap<COEXIST_THRESHOLD else primary
    # Only top two families participate in a local coexistence region.
    # Fractions interpolate from 50:50 at equal scores to 100:0 at cutoff.
    fractions=np.zeros(7)
    if label==7:
        fraction_secondary=.5*(1-gap/COEXIST_THRESHOLD)
        fractions[secondary]=fraction_secondary
        fractions[primary]=1-fraction_secondary
    else:
        fractions[primary]=1.
    return {'label':label,'fractions':fractions,'gap':gap,'scores':s,
            'primary':primary,'secondary':secondary}



# Peak tuples are (position, amplitude, Gaussian sigma).
PEAKS = {
    0: [(2.25,1.10,.22)],
    1: [(1.75,.95,.21),(4.55,.76,.24),(7.45,.92,.24)],
    2: [(3.12,1.05,.23),(6.63,.83,.28)],
    3: [(2.05,.83,.22),(4.65,.90,.29),(7.25,.60,.25)],
    4: [(2.75,.77,.27),(5.25,.78,.24),(7.14,.85,.26)],
    5: [(2.35,.86,.24),(4.90,.62,.29),(6.45,.88,.27)],
    6: [(2.35,.80,.25),(4.90,.80,.29),(7.15,.75,.27)]
}

def gaussian(z, center, width):
    return np.exp(-.5*((z-center)/width)**2)

def family_spectrum(family, x, y, energy=ENERGY):
    a,b,c=composition(x,y)
    mixing=1-(a*a+b*b+c*c)
    out=np.zeros_like(energy,dtype=float)
    # Exact end member signatures at pure A/B/C.
    pure=((family==0 and a==1) or (family==1 and b==1) or (family==2 and c==1))
    for j,(center,amp,width) in enumerate(PEAKS[family]):
        shift=0. if pure else (.43*(b-a)+.23*c + .10*j*(a-c))
        broaden=1. if pure else (1+1.10*mixing*(1 if family==6 else .6))
        strength=amp if pure else amp*(1+.12*(a-b)*(j-1))
        out+=strength*gaussian(energy,center+shift,width*broaden)
    return out

def true_spectrum(x,y,energy=ENERGY):
    st=phase_state(x,y)
    signal=np.zeros_like(energy,dtype=float)
    for k,weight in enumerate(st['fractions']):
        if weight>0: signal+=weight*family_spectrum(k,x,y,energy)
    return signal+.035+.006*energy



class TernarySpectroscopySimulator:
    def __init__(self, seed=SEED, energy=ENERGY, time_budget=500.):
        self.rng=np.random.default_rng(seed)
        self.energy=np.asarray(energy,dtype=float)
        self.time_budget=float(time_budget)
        self.used_time=0.
        self.history=[]

    def noise_std(self,x,y,measurement_time):
        composition(x,y)
        t=float(measurement_time)
        if not np.isfinite(t) or t <= 0: raise ValueError('measurement_time must be finite and > 0')
        z=self.energy
        signal=true_spectrum(x,y,z)
        # hidden spatial / energy-dependent noise envelope
        spatial=1+.27*np.sin(11*x+5*y)**2+.18*np.cos(9*y-2*x)**2
        spectral=1+.16*np.sin(1.5*z+4*x-2*y)**2
        shot=(.065*np.sqrt(.25+signal)*spatial*spectral)**2/t
        floor=(.0065*(1+.30*np.cos(.8*z+3*y)**2))**2
        return np.sqrt(shot+floor)

    def noise_at(self,x,y,measurement_time,return_std=False):
        sigma=self.noise_std(x,y,measurement_time)
        independent=self.rng.normal(size=len(self.energy))
        # correlated structure is generated with an independent normal sequence
        raw=self.rng.normal(size=len(self.energy))
        kernel=np.exp(-.5*(np.arange(-10,11)/3.5)**2)
        kernel/=np.linalg.norm(kernel)  # approximately unit variance away from edges
        correlated=np.convolve(raw,kernel,mode='same')
        draw=sigma*(.92*independent+np.sqrt(1-.92**2)*correlated)
        return (draw,sigma) if return_std else draw

    def acquire(self,x,y,measurement_time):
        composition(x,y)
        t=float(measurement_time)
        if not np.isfinite(t) or t<=0: raise ValueError('measurement_time must be positive')
        if self.used_time+t>self.time_budget+1e-10:
            raise RuntimeError(f'Budget exceeded: {self.used_time+t:.3f} > {self.time_budget:.3f}')
        measurement=true_spectrum(x,y,self.energy)+self.noise_at(x,y,t)
        self.used_time+=t
        result={'x':float(x),'y':float(y),'z':float(1-x-y),'measurement_time':t,
                'energy':self.energy.copy(),'spectrum':measurement,
                'time_used':self.used_time,'time_remaining':max(0.,self.time_budget-self.used_time)}
        self.history.append({'x':float(x),'y':float(y),'time':t})
        return result



# Add scalar-channel measurement and a named full-spectrum method to the simulator.
def _draw_pixel(self, x, y, time, wavelength):
    composition(x, y)
    t = float(time)
    w = float(wavelength)
    if not np.isfinite(w) or not (self.energy.min() <= w <= self.energy.max()):
        raise ValueError('wavelength must be inside the instrument spectral range')
    if not np.isfinite(t) or t <= 0:
        raise ValueError('time must be finite and positive')
    if self.used_time + t > self.time_budget + 1e-10:
        raise RuntimeError('Measurement-time budget exceeded')
    # Interpolate between detector channels, obtaining a scalar random reading.
    # Drawing the full noise field also preserves the noise spectral correlation.
    clean = float(true_spectrum(x, y, np.array([w]))[0])
    detector_noise = self.noise_at(x, y, t)
    noise_value = float(np.interp(w, self.energy, detector_noise))
    value = clean + noise_value
    self.used_time += t
    result = {'x':float(x), 'y':float(y), 'wavelength':w,
              'measurement_time':t, 'intensity':float(value),
              'time_used':self.used_time,
              'time_remaining':max(0., self.time_budget-self.used_time)}
    self.history.append({'x':float(x),'y':float(y),'time':t,
                         'kind':'pixel','wavelength':w})
    return result

TernarySpectroscopySimulator.draw_pixel = _draw_pixel
TernarySpectroscopySimulator.draw_noisy_spectrum = (
    lambda self,x,y,time: self.acquire(x,y,time)
)


