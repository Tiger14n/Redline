"""PIPELINE V3: one switch-penalty Viterbi (DBN) decode over a (tempo x phase) lattice,
run directly on the model's ACTIVATION CURVES. Replaces the whole peak-pick -> dedupe ->

Why: peak-picking quantizes the activations into a brittle point set; every v2 stage after
it patches information that was already thrown away (measured: calibrated conf injected
post-picking moved nothing). Here the decoder sees the full curves and the transition
model carries the grid through silence on prior (= human behavior).

State: (tau, phi) -- beat interval tau in frames (50fps), countdown phase phi in [0, tau).
phi==0 is a beat. Transitions per frame:
  (tau, phi>0) -> (tau, phi-1)                cost 0          (countdown)
  (tau, 0)     -> (tau', tau'-1)              kappa*|log r|   (+LAM_SWITCH if |log r|>delta)
  (tau, 0)     -> (tau, phi0), phi0!=tau-1    LAM_ANCHOR      (offset re-anchor, same tempo)
Observation per frame, weighted by calibrated conf c (low conf => flat likelihood =>
decoder coasts on the transition model): phi==0 states get c*log(b~), others c*log(1-b).

Redlines = decoded beats where tempo switched (|log r|>delta) or a re-anchor fired;
each segment then gets ONE (offset, mpb) by confidence-weighted least squares on the

"""
from __future__ import annotations
import argparse, glob, math, pickle, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))




# ---------------------------------------------------------------- lattice ----
LFRAME_MS = 10.0          # lattice runs at 100fps: integer intervals cover fast BPMs
                          # (240bpm = 25 lattice frames EXACTLY; at 50fps it is 12.5 ->
                          # unrepresentable -> decoder rationally halves every fast song)


class Lattice:
    def __init__(self, tau_min=16, tau_max=200):                   # BPM 30..375 @100fps
        self.taus = np.arange(tau_min, tau_max + 1)                # [K]
        self.K = len(self.taus)
        self.starts = np.concatenate([[0], np.cumsum(self.taus)[:-1]])  # state idx of phi=0
        self.S = int(self.taus.sum())
        self.tau_of = np.repeat(self.taus, self.taus)              # [S]
        self.phi_of = np.concatenate([np.arange(t) for t in self.taus])
        self.entry_idx = self.starts + self.taus - 1               # states (tau, tau-1)

    def trans_matrix(self, kappa, delta, lam_switch, lam_oct=0.0):
        """C[i,j] = additive log-score of beat-entry transition tau_i -> tau_j.
        lam_oct: extra penalty for OCTAVE-SIZED jumps (C-coverage diagnostic: 46/83
        bad C-sections are mid-song local level switches into quiet parts; the human
        convention keeps ONE level through half-time sections, so near-2x transitions
        get a dedicated cost real tempo changes rarely pay)."""
        lr = np.abs(np.log(self.taus[None, :] / self.taus[:, None].astype(np.float64)))
        C = kappa * lr
        C[lr > delta] += lam_switch
        if lam_oct > 0:
            ln2 = math.log(2.0)
            C[(np.abs(lr - ln2) < 0.06) | (np.abs(lr - 2 * ln2) < 0.06)] += lam_oct
        return -C

    def trans_matrix_step(self, kappa, lam_step, lam_oct):
        """GLIDE-PROOF cost shaping: linear kappa*|log r| charges a monotone walk
        exactly what it charges one jump (triangle equality), so per-jump penalties
        are dodged by walking -- the loophole that laundered phase steps AND level
        switches. A flat lam_step on EVERY tau change makes a 30-step octave walk
        cost ~30x lam_step while one honest jump costs kappa*ln2 + lam_step + lam_oct;
        level changes are forced into single jumps where lam_oct can tax octaves."""
        lr = np.abs(np.log(self.taus[None, :] / self.taus[:, None].astype(np.float64)))
        C = kappa * lr
        C[lr > 1e-9] += lam_step
        ln2 = math.log(2.0)
        C[(np.abs(lr - ln2) < 0.06) | (np.abs(lr - 2 * ln2) < 0.06)] += lam_oct
        return -C


def upsample2(x):
    """50fps curve -> 100fps by linear interpolation."""
    out = np.empty(2 * len(x), x.dtype)
    out[0::2] = x
    out[1:-1:2] = 0.5 * (x[:-1] + x[1:])
    out[-1] = x[-1]
    return out


def viterbi(b, conf, lat: Lattice, kappa=30.0, delta=0.045,
            lam_switch=40.0, gamma=0.0, lam_oct=0.0, lam_step=0.0):
    """Decode 100fps curves. -> (path_tau[T], beat_mask[T]) in lattice frames.
    conf (if given, gamma>0) weights ONLY the beat-state observation (it is calibrated
    to 'correct beat here', ~beat-density mean -- as a global weight it flattens the
    informative non-beat evidence)."""
    T = len(b)
    b = np.clip(b.astype(np.float64), 1e-5, 1 - 1e-5)
    # beat-state observation tolerates the peak landing one lattice frame off
    bw = np.maximum(b, np.maximum(np.roll(b, 1), np.roll(b, -1)))
    logb, log1mb = np.log(bw), np.log1p(-b)
    beat_obs = logb - log1mb
    if conf is not None and gamma > 0:
        beat_obs = beat_obs * np.clip(conf, 0.05, 1.0) ** gamma

    if lam_step > 0:
        Cmat = lat.trans_matrix_step(kappa, lam_step, lam_oct)     # glide-proof shaping
    else:
        Cmat = lat.trans_matrix(kappa, delta, lam_switch, lam_oct)  # [K,K] additive
    V = np.zeros(lat.S)
    V += log1mb[0]
    V[lat.starts] += beat_obs[0]

    src_tau = np.zeros((T, lat.K), np.int16)                       # entry backpointer
    starts, entry_idx = lat.starts, lat.entry_idx
    kix = np.arange(lat.K)

    for t in range(1, T):
        B = V[starts]                                              # beat scores [K]
        nv = np.empty_like(V)
        nv[:-1] = V[1:]                                            # countdown shift
        sc = B[:, None] + Cmat                                     # [K src, K dst]
        ad = sc.argmax(0)
        src_tau[t] = ad
        nv[entry_idx] = sc[ad, kix]
        nv += log1mb[t]
        nv[starts] += beat_obs[t]
        V = nv

    # backtrace
    s = int(np.argmax(V))
    path_tau = np.zeros(T, np.int32)
    beat_mask = np.zeros(T, bool)
    for t in range(T - 1, -1, -1):
        tau, phi = int(lat.tau_of[s]), int(lat.phi_of[s])
        path_tau[t] = tau
        if phi == 0:
            beat_mask[t] = True
        if t == 0:
            break
        if phi == tau - 1:
            s = int(starts[src_tau[t, np.searchsorted(lat.taus, tau)]])
        else:
            s = s + 1
    return path_tau, beat_mask


# ----------------------------------------------------- segments -> redlines ----

# NOTE: the legacy V3Timing solver (dp_fit/merge era) stays in the dev repo;
# Redline ships only the active lattice/viterbi decode used by the solver.
