"""Timing solver: per-frame model curves -> osu! redlines.

Curves (beat/down/off/tempo/change/red) -> viterbi beat decode -> tempo-curve
segmentation -> per-segment redline fits -> refinement passes (octave anchoring,
downbeat phase, live re-anchoring, notation-head placement).

  TimingSolver().predict(signal, sr) -> [(offset_ms, ms_per_beat, meter), ...]
"""
from __future__ import annotations
import argparse, sys, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from redline.const import FRAME_MS                                               # noqa: E402
from redline.targets import BPM_MIN                                              # noqa: E402
from redline.lattice import Lattice, viterbi, upsample2                      # noqa: E402
# NOTE: redline.infer (Model2Beats) and redline.curves (model_curves) are imported LAZILY
# inside the checkpoint path below — they pull in torch, so a curves-only consumer
# (TimingSolver(ckpt=None).predict_from_curves) stays torch-free.

import os as _os
DEFAULT_CKPT = _os.environ.get("REDLINE_CKPT", str(Path(__file__).resolve().parent.parent / "models" / "redline-v1.pt"))


class UnifiedRedlineSolverRobust:
    def __init__(self, max_error_ms=15.0):
        self.max_error_ms = max_error_ms

    def _get_p_bounds(self, y):
        M = len(y)
        if M <= 1: return 0.0, np.inf
        
        Y = np.asarray(y)
        dy = np.diff(Y)
        P_est = float(np.median(dy))
        if P_est < 10.0: return np.inf, 0.0
        
        # Check for Tail Drift (The lag fix):
        if M >= 4:
            tail_dy = dy[-2:]
            if np.all(tail_dy > P_est + 8.0) or np.all(tail_dy < P_est - 8.0):
                return np.inf, 0.0
        
        m = np.concatenate(([0], np.cumsum(np.round(dy / P_est))))
        P_lin = (Y[-1] - Y[0]) / (m[-1] - m[0]) if m[-1] > m[0] else P_est
        
        unrolled = Y - (Y[0] + m * P_lin)
        median_e = float(np.median(unrolled))
        
        err = np.abs(unrolled - median_e)
        mask = err <= (self.max_error_ms * 1.5)
        
        if np.sum(~mask) > max(1, 0.02 * M): 
            return np.inf, 0.0
            
        Y_clean = Y[mask]
        m_clean = m[mask]
        
        if len(Y_clean) <= 1: return np.inf, 0.0
        
        j_idx, k_idx = np.triu_indices(len(Y_clean), k=1)
        dY = Y_clean[k_idx] - Y_clean[j_idx]
        dM = m_clean[k_idx] - m_clean[j_idx]
        
        valid = dM > 0
        dY = dY[valid]
        dM = dM[valid]
        
        if len(dM) == 0: return np.inf, 0.0
        
        p_min = np.max((dY - 2 * self.max_error_ms) / dM)
        p_max = np.min((dY + 2 * self.max_error_ms) / dM)
        return float(p_min), float(p_max)

    def _check_feasibility(self, y):
        p_min, p_max = self._get_p_bounds(y)
        return p_min <= p_max

    def _get_max_segment_length(self, beats, start_idx, max_allowed_L=None):
        N = len(beats)
        max_L = N - start_idx
        if max_allowed_L is not None:
            max_L = min(max_L, max_allowed_L)
            
        if max_L <= 2: return max_L
            
        low, high = 2, max_L
        best_L = 2
        
        while low <= high:
            mid = (low + high) // 2
            if self._check_feasibility(beats[start_idx : start_idx + mid]):
                best_L = mid
                low = mid + 1
            else:
                high = mid - 1
        return best_L

    def _find_cleanest_bpm(self, lo, hi, y):
        lo, hi = min(lo, hi), max(lo, hi)
        
        search_lo = max(10.0, lo - 1.5)
        search_hi = hi + 1.5
        
        # Increase resolution to 60 to perfectly hit .1, .2, .3, .4, .5 fractions
        grid = np.arange(np.ceil(search_lo * 60 - 1e-5), np.floor(search_hi * 60 + 1e-5) + 1) / 60.0
        if len(grid) == 0: return (lo + hi) / 2.0
            
        center = (lo + hi) / 2.0
        best_score = -1000
        best_cand = center
        
        Y = np.asarray(y)
        ignore_k = 0 if len(Y) < 8 else max(1, int(0.02 * len(Y)))
        span = max(hi - lo, 1.0) 
        
        for c in grid:
            c_round = round(c * 60) / 60.0
             
            score = 0
            if abs(c_round % 10) < 1e-4 or abs(c_round % 10 - 10) < 1e-4: score = 10
            elif abs(c_round % 5) < 1e-4 or abs(c_round % 5 - 5) < 1e-4: score = 8
            elif abs(c_round % 2) < 1e-4 or abs(c_round % 2 - 2) < 1e-4: score = 5
            elif abs(c_round % 1) < 1e-4 or abs(c_round % 1 - 1) < 1e-4: score = 3
            elif abs((c_round * 2) % 1) < 1e-4 or abs((c_round * 2) % 1 - 1) < 1e-4: score = 1
            elif abs((c_round * 3) % 1) < 1e-4 or abs((c_round * 3) % 1 - 1) < 1e-4: score = 0.5
            elif abs((c_round * 4) % 1) < 1e-4 or abs((c_round * 4) % 1 - 1) < 1e-4: score = 0.25
            elif abs((c_round * 5) % 1) < 1e-4 or abs((c_round * 5) % 1 - 1) < 1e-4: score = 0.2
            elif abs((c_round * 10) % 1) < 1e-4 or abs((c_round * 10) % 1 - 1) < 1e-4: score = 0.1
            
            dist = abs(c_round - center) / span
            score -= dist
            
            if c_round < lo or c_round > hi:
                cand_p = 60000.0 / c_round
                m_temp = np.round((Y - Y[0]) / cand_p)
                unrolled_errors = Y - (Y[0] + m_temp * cand_p)
                sorted_errs = np.sort(unrolled_errors)
                
                if ignore_k > 0:
                    min_e = sorted_errs[ignore_k]
                    max_e = sorted_errs[-1 - ignore_k]
                else:
                    min_e = sorted_errs[0]
                    max_e = sorted_errs[-1]
                    
                max_err = (max_e - min_e) / 2.0
                
                if max_err > self.max_error_ms:
                    excess = max_err - self.max_error_ms
                    if excess > 6.0: 
                        continue
                    score -= (excess * 1.5)
            
            if score > best_score:
                best_score = score
                best_cand = c_round
                
        return best_cand

    def _optimize_segment(self, y):
        M = len(y)
        if M <= 1: return float(y[0]) if M == 1 else 0.0, 500.0

        p_min, p_max = self._get_p_bounds(y)
        
        if p_min > p_max or p_max == np.inf:
            dy = np.diff(y)
            P_est = float(np.median(dy))
            p_min, p_max = P_est, P_est

        p_min = max(p_min, 10.0) 
        bpm_min, bpm_max = 60000.0 / p_max, 60000.0 / p_min
        
        clean_bpm = self._find_cleanest_bpm(bpm_min, bpm_max, y)
        clean_p = 60000.0 / clean_bpm
        
        Y = np.asarray(y)
        m_temp = np.round((Y - Y[0]) / clean_p)
        unrolled_errors = Y - (Y[0] + m_temp * clean_p)
        
        median_e = float(np.median(unrolled_errors))
        err_from_median = np.abs(unrolled_errors - median_e)
        mask = err_from_median <= self.max_error_ms * 1.5
        
        if np.sum(mask) > 0:
            valid_errs = unrolled_errors[mask]
            min_e = np.min(valid_errs)
            max_e = np.max(valid_errs)
            minimax_shift = (max_e + min_e) / 2.0
            O_opt = Y[0] + minimax_shift
        else:
            O_opt = Y[0] + median_e
            
        O_opt_mod = O_opt % clean_p
        
        shift = round((Y[0] - O_opt_mod) / clean_p)
        O_final = O_opt_mod + shift * clean_p
        
        return float(O_final), float(clean_p)

    def _precise_fit_seg(self, seg):
        """Exact least-squares offset+BPM on the beats (no clean-BPM rounding, no minimax) -- the
        same fit the tempo solver's precise_fit uses. Centers the grid on the BULK of beats (LS,
        median residual ~0) instead of the residual range (minimax, pulled by jittery extremes),
        so the redline offset lands ON the beats. Keeps DP's segmentation, fixes its ~3ms offset
        bias. One outlier-rejection pass via MAD."""
        s = np.asarray(seg, float)
        if len(s) < 2:
            return (float(s[0]) if len(s) else 0.0), 500.0
        g = np.diff(s); g = g[g > 0]
        mpb_ref = float(np.median(g)) if len(g) else 500.0
        if mpb_ref <= 0:
            return float(s[0]), 500.0
        steps = np.maximum(1, np.round(np.diff(s) / mpb_ref)).astype(int)
        idx = np.concatenate([[0], np.cumsum(steps)]).astype(float)
        mpb, intercept = np.polyfit(idx, s, 1)
        res = s - (intercept + idx * mpb)
        mad = float(np.median(np.abs(res - np.median(res)))) * 1.4826 + 1e-6
        good = np.abs(res - np.median(res)) <= 3 * mad + 5.0
        if min(8, len(s)) <= int(good.sum()) < len(s):
            mpb, intercept = np.polyfit(idx[good], s[good], 1)
        return float(intercept), float(mpb)

    def process_beats(self, bt_list, boundaries=None):
        N = len(bt_list)
        if N == 0: return []
        if boundaries is None: boundaries = []
            
        redlines = []
        i = 0
        
        while i < N:
            max_allowed_L = N - i
            for b_time in boundaries:
                if b_time > bt_list[i] + 100: 
                    b_idx = int(np.argmin(np.abs(np.asarray(bt_list) - b_time)))
                    if b_idx > i:
                        max_allowed_L = min(max_allowed_L, b_idx - i + 1)
                    break
            
            L = self._get_max_segment_length(bt_list, i, max_allowed_L)
            segment_beats = bt_list[i : i + L]
            
            O_opt, P_opt = self._optimize_segment(segment_beats)
            redlines.append((O_opt, P_opt))
            
            if i + L >= N: break
            i = i + max(1, L - 1)
            
        return redlines

    def process_beats_dp(self, bt_list, boundaries=None, precise=False):
            N = len(bt_list)
            if N == 0: return []
            if boundaries is None: boundaries = []
                
            # Convert boundary timestamps to nearest beat indices for fast lookup
            boundary_indices = set()
            bt_array = np.asarray(bt_list)
            for b_time in boundaries:
                idx = int(np.argmin(np.abs(bt_array - b_time)))
                boundary_indices.add(idx)
                
            # dp[i] represents the minimum cost to partition beats[0...i-1]
            dp = [float('inf')] * (N + 1)
            dp[0] = 0.0
            
            # backtrack[i] stores the start index 'j' that gave the optimal cost for dp[i]
            backtrack = [0] * (N + 1)
            
            # To prevent O(N^2) lag, we cap the max segment length we search.
            # 150 beats is about ~30-40 seconds of continuous music without a redline.
            MAX_SEARCH_LEN = 150 
            
            for i in range(1, N + 1):
                # Look backwards to find all valid segments that end at 'i'
                start_j = max(0, i - MAX_SEARCH_LEN)
                
                for j in range(start_j, i):
                    segment = bt_list[j:i]
                    L = i - j
                    
                    # If segment is invalid under the 15ms constraint, skip it
                    if not self._check_feasibility(segment):
                        continue
                    
                    # --- COST FUNCTION DESIGN ---
                    # 1. Base cost (minimize total red lines)
                    cost = 1.0 
                    
                    # 2. The "Anti-Crumb" Penalty
                    # Heavily penalize short segments (less than 1 measure / 4 beats)
                    if L < 4:
                        is_boundary_start = (j in boundary_indices) or (j == 0)
                        is_boundary_end = (i in boundary_indices) or (i == N)
                        
                        # If the short segment bridges two intentional boundaries, it's okay.
                        # Otherwise, penalize it heavily so the algorithm prefers balanced lengths.
                        if not (is_boundary_start or is_boundary_end):
                            cost += 10.0 * (4 - L) 
                    
                    # 3. Boundary alignment reward (optional)
                    # Slightly reward segments that cleanly align with detected IBI boundaries
                    if j in boundary_indices:
                        cost -= 0.1 
                    
                    total_cost = dp[j] + cost
                    
                    # Update DP table if we found a better path
                    if total_cost < dp[i]:
                        dp[i] = total_cost
                        backtrack[i] = j
                        
            # --- PATH RECONSTRUCTION ---
            segments = []
            curr = N
            while curr > 0:
                prev = backtrack[curr]
                segments.append((prev, curr))
                curr = prev
                
            segments.reverse() # Chronological order: [(0, 150), (150, 300), ...]
            
            # --- NEW: POST-PROCESS SEGMENT MERGING ---
            # Merge adjacent segments if they can be represented by a single constant BPM
            merged_segments = []
            for seg in segments:
                if not merged_segments:
                    merged_segments.append(seg)
                else:
                    prev_start, prev_end = merged_segments[-1]
                    curr_start, curr_end = seg
                    
                    # Test if the combined segment is mathematically feasible
                    combined_segment = bt_list[prev_start : curr_end]
                    if self._check_feasibility(combined_segment):
                        # Merge them! Update the end bound of the last segment
                        merged_segments[-1] = (prev_start, curr_end)
                    else:
                        merged_segments.append(seg)
            
            # fit offsets and BPMs per segment
            redlines = []
            for (j, i) in merged_segments:
                segment_beats = bt_list[j:i]
                O_opt, P_opt = (self._precise_fit_seg(segment_beats) if precise
                                else self._optimize_segment(segment_beats))
                redlines.append((O_opt, P_opt))
                
            # --- POST-PROCESS: Deduplicate artificial phase-continuous splits ---
            if not redlines: return []
            final_redlines = [redlines[0]]
            for i in range(1, len(redlines)):
                prev_off, prev_mpb = final_redlines[-1]
                curr_off, curr_mpb = redlines[i]
                
                prev_bpm = 60000.0 / prev_mpb
                curr_bpm = 60000.0 / curr_mpb
                
                # If BPM difference is extremely small (< 0.2 BPM difference)
                if abs(prev_bpm - curr_bpm) < 0.2:
                    beats_elapsed = round((curr_off - prev_off) / prev_mpb)
                    expected_off = prev_off + beats_elapsed * prev_mpb
                    
                    # If the new section seamlessly continues the phase grid of the previous section
                    # We use a relaxed tolerance (35ms) here compared to max_error_ms because DP split 
                    # it specifically due to cumulative drift over hundreds of beats.
                    # If they align within ~35ms, it's just drift, not an intentional re-anchor.
                    if abs(curr_off - expected_off) <= max(35.0, self.max_error_ms * 2.0):
                        # Skip adding this redline, let the previous one carry over
                        continue
                        
                final_redlines.append(redlines[i])
                
            return final_redlines
    
class TimingSolver:
    """The timing solver: model curves in, redlines out."""

    def __init__(self, device="cuda", ckpt=DEFAULT_CKPT,
                 max_error_ms=9.0, lam_switch=1.0,
                 use_ibi_change=True, ibi_th=0.05,
                 use_meter2=False, refine_placement=True, conf_thresh=0.40,
                 solver="tempo", chg_th=0.35, chg_min_gap=1500.0,
                 merge_tol=22.0, merge_bpm_frac=0.03, min_seg_beats=4,
                 use_rubato=False, rubato_th=10.0, rubato_stride=4, precise_fit=True,
                 split_unfit=True, split_th=25.0,
                 octave_anchor=True, anchor_conf=0.20, anchor_ratio=1.4,
                 snap_bounds=True, bpm_snap=0.0, drift_tol=0.0,
                 fitgate=True, fitgate_resid=12.0, fitgate_frac=0.20,
                 fitgate_act=0.45, fitgate_run=8,
                 overlay_group=True, overlay_tol=8.0, overlay_max_beats=16,
                 prune_crumbs=True, prune_conf=0.45, prune_ratio_tol=0.08,
                 live_reanchor=True, live_gate_resid=6.0, live_gate_frac=0.25,
                 live_snap=8.0, live_jit=3.0, anchor_music=True,
                 red_place=True, red_th=0.5, red_min_gap_beats=0.0, const_cover=0.85,
                 const_collapse=True, const_collapse_cover=0.90, anchor_downbeat=True,
                 const_collapse_ppm=6.0, const_collapse_resid=3.0,
                 red_live_ppm=6.0):
        self.unified_solver = UnifiedRedlineSolverRobust(max_error_ms=max_error_ms)
        self.lam_switch = lam_switch
        self.use_ibi_change = use_ibi_change
        self.ibi_th = ibi_th
        self.use_meter2 = use_meter2
        self.refine_placement = refine_placement
        self.conf_thresh = conf_thresh
        # change-head-driven segmentation ("chg"): split decoded beats at change-head peaks,
        # fit one redline per segment, merge neighbours whose BPM did not actually change.
        self.solver = solver
        self.chg_th = chg_th                  # change-head peak threshold
        self.chg_min_gap = chg_min_gap        # min ms between change peaks
        self.merge_tol = merge_tol            # ms residual to call two segments one tempo
        self.merge_bpm_frac = merge_bpm_frac  # BPM fraction to call two segments one tempo
        self.min_seg_beats = min_seg_beats
        self.use_rubato = use_rubato          # drift re-anchor split (off = cleaner; gated when on)
        self.rubato_th = rubato_th            # median residual (ms) to trigger a drift split
        self.rubato_stride = rubato_stride    # beats per drift redline (higher = fewer redlines)
        self.precise_fit = precise_fit        # exact LS BPM (beats<10ms) vs round-BPM snapping
        self.split_unfit = split_unfit        # split tempo segments a single line can't fit (missed changes)
        self.split_th = split_th              # 95th-pct residual (ms) above which a segment is split
        self.octave_anchor = octave_anchor    # hold established tempo through break/half-time octave excursions
        self.anchor_conf = anchor_conf        # mean beat-activation above which a redline is "confident" (real)
        self.anchor_ratio = anchor_ratio      # fold a redline toward the anchor if its tempo is >this factor off
        self.snap_bounds = snap_bounds        # un-lag tempo-run boundaries by snapping to the IBI step
        self.bpm_snap = bpm_snap              # snap precise-fit BPM to a musical value within this many BPM (0=off)
        self.bpm_snap_min_beats = 96          # only snap LONG segments (short ones don't drift past 10ms)
        self.drift_tol = drift_tol            # optional split of segments whose halves fit different BPMs; default 0 (off)
        self.fitgate = fitgate                # fit-failure overlay: per-beat redlines where a constant grid provably can't fit the decoded beats (live drummers)
        self.fitgate_resid = fitgate_resid    # final-grid residual (ms) above which a beat counts as misfit
        self.fitgate_frac = fitgate_frac      # min fraction of a segment's strong beats misfit to allow the overlay there (clean maps never qualify)
        self.fitgate_act = fitgate_act        # beat-activation floor: only REAL beats can trigger/receive the overlay
        self.fitgate_run = fitgate_run        # min misfit run length in beats (short blips = decode glitches, left alone)
        self.overlay_group = overlay_group    # human-style overlay: one line per 1-3 MEASURES (greedy growth) instead of per beat
        self.overlay_tol = overlay_tol        # max residual (ms) of a grouped line vs the decoded beats it covers
        self.overlay_max_beats = overlay_max_beats
        self.prune_crumbs = prune_crumbs      # drop low-confidence same-pulse crumb redlines (fade-out/soft-bridge octave confusions or redundant re-anchors)
        self.prune_conf = prune_conf          # a redline's span-mean beat-activation below this = "low confidence"
        self.prune_ratio_tol = prune_ratio_tol  # tempo must be within this fold-ratio of the confident previous grid (unison OR octave) to be a droppable crumb; a genuinely different soft tempo is KEPT
        self.live_reanchor = live_reanchor      # per-barline redlines on globally-drifting (live) maps — the human timing convention
        self.live_gate_resid = live_gate_resid  # a decoded beat "misfits" the grid beyond this (ms)
        self.live_gate_frac = live_gate_frac    # global gate: fraction of misfit beats to call the map live
        self.live_snap = live_snap              # measures fitting the grid within this keep grid offset+mpb (protects p10)
        self.live_jit = live_jit                # median local-IBI deviation above this = decode jitter -> don't re-anchor
        self.anchor_music = anchor_music        # add a grid-exact first redline at the music-start barline (mapper anchor convention)
        self.red_place = red_place              # notation-head placement (needs a --red_head model; inert otherwise)
        self.red_th = red_th                    # red-curve peak threshold
        self.const_cover = const_cover          # single-line decoded-beat coverage above which the map is constant-like (suppress dense overlays)
        self.const_collapse = const_collapse    # collapse tracking re-anchors to ONE redline when a refitted global grid covers the song
        self.const_collapse_cover = const_collapse_cover   # coverage bar for that collapse (0.90: above the drift regime, tuned on the bench dev half)
        self.anchor_downbeat = anchor_downbeat  # write the first redline on the model's measure start (whole-beat shift; grid unchanged)
        self.const_collapse_ppm = const_collapse_ppm       # notation-head peaks/min above which a map is convention-timed -> never collapse
        self.const_collapse_resid = const_collapse_resid   # also collapse when the global grid's MEDIAN residual is <= this (0 = off); robust where coverage isn't
        self.red_live_ppm = red_live_ppm        # notation-head peaks/min above which the song counts as convention-timed (live)
        self.red_min_gap_beats = red_min_gap_beats  # min beats between NOTATION-ONLY (grid-exact) placements.
                                                # default 0 (off): mapper same-BPM re-anchors are often
                                                # only ~4 beats apart, so any gap costs real recall.
        self._rubato_perbeat_mask = []        # rubato specialist (--solver rubato): per-redline "is per-beat overlay" flags
        if ckpt is not None:
            from redline.infer import Model2Beats                # lazy: pulls in torch
            self.m2b = Model2Beats(ckpt, device=device)
        else:
            self.m2b = None
        self.lat = Lattice()

    def predict(self, sig, sr, return_beats=False):
        # want_tempo=True so solver="tempo" + the octave anchor get the per-frame tempo curve
        # (without it _segment_tempo silently falls back to the change-head segmenter).
        from redline.curves import model_curves                  # lazy: pulls in torch
        curves = model_curves(self.m2b, sig, sr, want_hidden=False, want_tempo=True)
        return self.predict_from_curves(curves, return_beats=return_beats)

    def decode_beats(self, curves):
        ab, ao = curves["beat"], curves["off"]
        path_tau, bm = viterbi(upsample2(ab), None, self.lat, lam_switch=self.lam_switch)
        bf = np.where(bm)[0]
        if len(bf) < 8:
            return []
        f20 = np.minimum(bf // 2, len(ao) - 1)
        bt_list = list(map(float, (f20 + ao[f20]) * FRAME_MS))
        return self._trim_phantom(bt_list, ab)

    def _trim_phantom(self, bt_list, ab, th=0.15):
        """The bar-pointer lattice has no rest state, so viterbi MUST emit beats through
        leading/trailing silence (phantom ~min-BPM pulses on the noise floor). Those phantoms
        become crumb redlines with garbage BPM before the music starts (and the octave anchor
        can densify them into a full phantom grid). Strip edge beats with no activation
        support; interior beats are never touched."""
        ab = np.asarray(ab)
        n = len(bt_list)
        if n == 0:
            return bt_list
        fr = np.clip(np.round(np.asarray(bt_list) / FRAME_MS).astype(int), 1, len(ab) - 2)
        sup = np.maximum(ab[fr - 1], np.maximum(ab[fr], ab[fr + 1]))
        i = 0
        while i < n and sup[i] < th:
            i += 1
        j = n
        while j > i and sup[j - 1] < th:
            j -= 1
        return bt_list[i:j]

    def _change_boundaries(self, ach, bt_list):
        """Beat indices where the change head fires -> segment starts. The change head was
        trained on MAJOR redlines (tempo/meter/phase steps), so its peaks ARE the genuine
        segmentation points. Each peak is snapped to the nearest decoded beat."""
        from redline import npsignal as ss
        bt = np.asarray(bt_list, float)
        ach = np.asarray(ach)
        if ach.max() > 1.1 or ach.min() < -0.1:            # logits -> prob
            ach = 1.0 / (1.0 + np.exp(-np.clip(ach, -100, 100)))
        pk = ss.find_peaks(ach, height=self.chg_th,
                           distance=max(1, int(self.chg_min_gap / FRAME_MS)))[0] * FRAME_MS
        idxs = sorted({int(np.argmin(np.abs(bt - p))) for p in pk})
        return [i for i in idxs if 0 < i < len(bt)]

    def _fit_seg(self, seg):
        """One redline (offset_ms, mpb) for a run of beat times. precise_fit=True -> exact
        least-squares slope+intercept on the beats (no round-BPM snapping), which keeps the
        grid on the beats (beats<10ms) at the cost of non-round BPM values. precise_fit=False
        -> the unified solver's clean-BPM optimizer (rounds to musical BPMs)."""
        if not self.precise_fit:
            return self.unified_solver._optimize_segment(seg)    # clean-BPM (rounds tempo)
        s = np.asarray(seg, float)
        if len(s) < 2:
            return (float(s[0]) if len(s) else 0.0), 500.0
        g = np.diff(s); g = g[g > 0]
        mpb_ref = float(np.median(g)) if len(g) else 500.0
        if mpb_ref <= 0:
            return float(s[0]), 500.0
        steps = np.maximum(1, np.round(np.diff(s) / mpb_ref)).astype(int)
        idx = np.concatenate([[0], np.cumsum(steps)]).astype(float)
        mpb, intercept = np.polyfit(idx, s, 1)
        res = s - (intercept + idx * mpb)                        # one outlier-rejection pass
        mad = float(np.median(np.abs(res - np.median(res)))) * 1.4826 + 1e-6
        good = np.abs(res - np.median(res)) <= 3 * mad + 5.0
        sf, xf = s, idx
        if min(8, len(s)) <= int(good.sum()) < len(s):
            sf, xf = s[good], idx[good]
            mpb, intercept = np.polyfit(xf, sf, 1)
        # BPM decimal-snap: the LS slope jitters ~0.01-0.02 BPM around the true tempo; on a long
        # section that accumulates into >10ms grid drift (FREEDOM DiVE: 284.21 vs GT 284.23 over
        # 726 beats). If the LS BPM sits right next to a musical value, snap to it. Self-protecting:
        # only fires when LS ~= clean, so genuinely non-round BPMs keep their exact LS slope.
        if self.bpm_snap > 0 and mpb > 1.0 and len(sf) >= self.bpm_snap_min_beats:
            bpm_ls = 60000.0 / mpb
            clean = self.unified_solver._find_cleanest_bpm(bpm_ls - 0.07, bpm_ls + 0.07, sf)
            if abs(clean - bpm_ls) <= self.bpm_snap:
                mpb = 60000.0 / clean
                intercept = float(np.median(sf - xf * mpb))      # re-anchor offset to snapped slope
        return float(intercept), float(mpb)

    def _merge_segments(self, segs, bt_list):
        """Greedily absorb a segment into the previous redline when its beats still lie on
        that grid (BPM within merge_bpm_frac AND median phase residual < merge_tol). Drops
        spurious change-head peaks where the tempo did not actually change -> no over-seg.
        Re-fits the combined span so a long merged run stays drift-free."""
        if not segs:
            return []
        out = [list(segs[0])]
        for O1, P1, s1, e1 in segs[1:]:
            O0, P0, s0, e0 = out[-1]
            sb = np.asarray(bt_list[s1:e1], float)
            resid = np.abs((sb - O0) - np.round((sb - O0) / P0) * P0) if P0 > 1 else np.array([1e9])
            bpm_close = abs(60000.0 / P1 - 60000.0 / P0) <= max(0.5, self.merge_bpm_frac * 60000.0 / P0)
            if bpm_close and float(np.median(resid)) < self.merge_tol:
                O, P = self._fit_seg(bt_list[s0:e1])             # refit combined span
                out[-1] = [O, P, s0, e1]
            else:
                out.append([O1, P1, s1, e1])
        return out

    def _segment_chg(self, curves, bt_list):
        """Change-head-driven redline segmentation -> list[(offset_ms, mpb)]."""
        N = len(bt_list)
        bidx = self._change_boundaries(curves["change"], bt_list)
        bounds = sorted(set([0] + bidx + [N]))
        segs = []
        for j in range(len(bounds) - 1):
            s, e = bounds[j], bounds[j + 1]
            # fold a too-short tail/segment into the previous one (avoid crumb redlines)
            if e - s < self.min_seg_beats and segs:
                segs[-1][3] = e
                continue
            seg = bt_list[s:e]
            if len(seg) < 2:
                continue
            O, P = self._fit_seg(seg)
            segs.append([O, P, s, e])
        # re-fit any segment whose bounds were extended by the short-tail fold
        for k, (O, P, s, e) in enumerate(segs):
            if e - s >= 2:
                segs[k][0], segs[k][1] = self._fit_seg(bt_list[s:e])
        merged = self._merge_segments(segs, bt_list)
        return [(O, P) for O, P, _, _ in merged]

    @staticmethod
    def _canon_bpm(b):
        """Fold a BPM into [120,240) by octave so half/double are treated as the same tempo
        (octave is explicitly out of scope -> octave jumps must NOT create boundaries)."""
        if b <= 0:
            return b
        while b >= 240.0:
            b /= 2.0
        while b < 120.0:
            b *= 2.0
        return b

    def _tempo_runs(self, bpm_curve, tol=0.04, min_run=18):
        """Segment the per-frame BPM curve into octave-folded constant runs -> [(f0,f1)]."""
        from redline.npsignal import medfilt
        n = len(bpm_curve)
        if n == 0:
            return []
        k = min(51, n if n % 2 else n - 1) | 1
        sm = medfilt(np.asarray(bpm_curve, float), kernel_size=max(1, k))
        runs = []
        f0 = 0
        ref = self._canon_bpm(sm[0])
        for f in range(1, n):
            c = self._canon_bpm(sm[f])
            if ref > 0 and abs(c - ref) / ref > tol:
                if f - f0 >= min_run:
                    runs.append((f0, f)); f0 = f
                ref = c
            else:
                ref = self._canon_bpm(float(np.median(sm[f0:f + 1])))
        runs.append((f0, n))
        return runs

    def _split_unfit(self, bt_list, s, e, depth=0):
        """Recursively split [s,e) wherever a single redline can't fit the beats within
        max_error_ms (a tempo change the tempo curve smoothed over). Split at the beat of
        largest grid residual. Gated by min_seg_beats + depth so constant segments (which fit)
        are never split -> recovers missed changes without over-segmenting clean maps."""
        O, P = self._fit_seg(bt_list[s:e])
        seg = np.asarray(bt_list[s:e], float)
        if not self.split_unfit or e - s < 2 * self.min_seg_beats or depth >= 8 or P <= 1:
            return [[O, P, s, e]]
        resid = np.abs((seg - O) - np.round((seg - O) / P) * P)
        # DRIFT-AWARE: live-tempo maps (DragonForce) change BPM by 1-3 (<4%) so the tempo-run
        # tolerance merges them; but the segment's two halves then fit different mean tempi. Split
        # at the best IBI change-point when the half-tempi differ by > drift_tol (robust to per-
        # beat jitter via group means -> constant maps stay whole).
        k_rel, bl, br = self._best_ibi_split(seg) if self.drift_tol > 0 else (-1, 0.0, 0.0)
        drift = k_rel > 0 and abs(bl - br) > self.drift_tol
        # robust: 95th-pct residual ignores lone jitter outliers (real changes push MANY beats
        # off-grid, jitter pushes one) -> constant maps with a bad beat aren't split.
        if not drift and float(np.percentile(resid, 95)) <= self.split_th:     # fits -> keep whole
            return [[O, P, s, e]]
        bi = s + (k_rel if drift else int(np.argmax(resid)))     # drift -> change-point; else worst beat
        bi = max(s + self.min_seg_beats, min(e - self.min_seg_beats, bi))
        if bi <= s or bi >= e:
            return [[O, P, s, e]]
        return self._split_unfit(bt_list, s, bi, depth + 1) + self._split_unfit(bt_list, bi, e, depth + 1)

    def _best_ibi_split(self, seg):
        """1D change-point on inter-beat intervals: the split index minimizing within-group IBI
        variance (two mean-IBI groups), O(N) via cumsums. Robust to per-beat jitter because it
        compares GROUP MEANS, not single beats. Returns (beat_idx, bpm_left, bpm_right)."""
        ibi = np.diff(seg); n = len(ibi)
        m = self.min_seg_beats
        if n <= 2 * m:
            return -1, 0.0, 0.0
        c1 = np.cumsum(ibi); c2 = np.cumsum(ibi * ibi)
        tot1, tot2 = c1[-1], c2[-1]
        ks = np.arange(m, n - m)
        if len(ks) == 0:
            return -1, 0.0, 0.0
        sl = c1[ks - 1]; sr = tot1 - sl
        ql = c2[ks - 1]; qr = tot2 - ql
        cost = (ql - sl * sl / ks) + (qr - sr * sr / (n - ks))
        j = int(np.argmin(cost)); k = int(ks[j])
        ml = c1[k - 1] / k; mr = (tot1 - c1[k - 1]) / (n - k)
        return k, (60000.0 / ml if ml > 0 else 0.0), (60000.0 / mr if mr > 0 else 0.0)

    def _snap_tempo_boundary(self, bt, bi, win=14, half=6):
        """The medfilt+argmax tempo curve lags a real tempo change by ~seconds, but the decoded
        BEATS change spacing AT the change. Within +-win beats of the lagged boundary, snap to the
        beat where the local median inter-beat-interval steps the most (scale/octave-invariant via
        |log ratio|). Fixes the tempo solver's boundary LAG without touching its precision."""
        ibi = np.diff(bt)
        if len(ibi) < 2 * half + 1:
            return bi
        lo = max(half, bi - win); hi = min(len(ibi) - half, bi + win)
        if hi <= lo:
            return bi
        best_k, best_step = bi, -1.0
        for k in range(lo, hi + 1):
            left = float(np.median(ibi[k - half:k])); right = float(np.median(ibi[k:k + half]))
            if min(left, right) <= 1.0:
                continue
            step = abs(math.log(right / left))
            if step > best_step:
                best_step, best_k = step, k
        return best_k

    def _segment_tempo(self, curves, bt_list):
        """Tempo-curve-driven segmentation: boundaries where the model's per-frame BPM steps
        (octave-folded), UNION beat-validated change-head boundaries (rapid steps the smoothed
        tempo curve misses), then one robust redline fit per segment + same-tempo consolidation."""
        if "tempo" not in curves:
            return self._segment_chg(curves, bt_list)
        tempo = np.asarray(curves["tempo"])
        bpm_curve = BPM_MIN + tempo.argmax(-1).astype(float)        # per-frame BPM (hard argmax)
        runs = self._tempo_runs(bpm_curve)
        bt = np.asarray(bt_list, float)
        N = len(bt)
        bounds = [0]
        for f0, f1 in runs[1:]:
            t = f0 * FRAME_MS
            bi = int(np.argmin(np.abs(bt - t)))
            if self.snap_bounds:
                bi = self._snap_tempo_boundary(bt, bi)          # un-lag: snap to the IBI step
            if 0 < bi < N:
                bounds.append(bi)
        # UNION: change-head proposals kept only when the local BPM actually steps across them
        # (octave-folded) -- adds the rapid changes the medfilt'd tempo curve smooths over,
        # while beat-validation drops the head's spurious peaks (no over-seg on clean maps).
        for ci in self._change_boundaries(curves["change"], bt_list):
            lo, hi = max(0, ci - 10), min(N, ci + 10)
            if ci - lo < 3 or hi - ci < 3:
                continue
            bl = self._canon_bpm(self._local_bpm(bt, lo, ci))
            br = self._canon_bpm(self._local_bpm(bt, ci, hi))
            if bl <= 0 or br <= 0 or abs(bl - br) / bl <= self.merge_bpm_frac:
                continue
            bounds.append(ci)
        bounds = sorted(set(bounds + [N]))
        segs = []
        for j in range(len(bounds) - 1):
            s, e = bounds[j], bounds[j + 1]
            if e - s < self.min_seg_beats and segs:
                segs[-1][3] = e; continue
            if e - s < 2:
                continue
            segs.extend(self._split_unfit(bt_list, s, e))        # recover changes the curve missed
        merged = self._merge_segments(segs, bt_list)
        return [(O, P) for O, P, _, _ in merged]

    def _local_bpm(self, bt, lo, hi):
        if hi - lo < 3:
            return 0.0
        d = np.diff(bt[lo:hi])
        d = d[d > 1]
        return 60000.0 / float(np.median(d)) if len(d) else 0.0

    def _segment_chgval(self, curves, bt_list, win=10, refine_rad=4):
        """Change head PROPOSES boundaries (high recall, low threshold); the decoded BEATS
        VALIDATE them (keep a split only if the local BPM actually changes across it, octave-
        folded). This filters the change head's false positives AND auto-drops same-BPM
        convention re-anchors. Each kept boundary is then snapped to the beat that best splits
        the two tempi (precise placement). Finally same-tempo consolidation."""
        bt = np.asarray(bt_list, float)
        N = len(bt)
        cand = self._change_boundaries(curves["change"], bt_list)   # uses self.chg_th
        kept = []
        for ci in cand:
            lo, hi = max(0, ci - win), min(N, ci + win)
            if ci - lo < 3 or hi - ci < 3:
                continue
            bl = self._canon_bpm(self._local_bpm(bt, lo, ci))
            br = self._canon_bpm(self._local_bpm(bt, ci, hi))
            if bl <= 0 or br <= 0:
                continue
            if abs(bl - br) / bl <= self.merge_bpm_frac:
                continue                                            # no real tempo change -> drop
            # precise placement: pick split beat in [ci-rad, ci+rad] minimising L+R IBI spread
            best, best_sc = ci, 1e18
            for s in range(max(lo + 3, ci - refine_rad), min(hi - 3, ci + refine_rad) + 1):
                dl, dr = np.diff(bt[lo:s]), np.diff(bt[s:hi])
                sc = (np.std(dl) if len(dl) else 0) + (np.std(dr) if len(dr) else 0)
                if sc < best_sc:
                    best_sc, best = sc, s
            kept.append(best)
        bounds = sorted(set([0] + [k for k in kept if 0 < k < N] + [N]))
        segs = []
        for j in range(len(bounds) - 1):
            s, e = bounds[j], bounds[j + 1]
            if e - s < 2:
                continue
            O, P = self._fit_seg(bt_list[s:e])
            segs.append([O, P, s, e])
        merged = self._merge_segments(segs, bt_list)
        return [(O, P) for O, P, _, _ in merged]

    def _octave_anchor(self, R, ab, tempo, last_ms, bt_list=None):
        """Confidence-gated, DENSIFY-ONLY octave correction. Lacking global context (30s window),
        the tempo head HALVES locally during breaks (the beat 'chills' after a high-BPM stream)
        and half-time sections, dropping beats a human keeps by holding the established redline.
        We fold a redline whose tempo sits an octave too SLOW back UP toward the dominant stream
        tempo (anchor), but only when it's a break/half-time (low confidence OR the tempo head
        still holds mass at the anchor octave). A genuine slow change -- high confidence AND no
        tempo-head support for the fold -- is left alone. We never thin a DOUBLED redline: a
        doubled grid already covers every real beat (octave-doubling is don't-care) so folding it
        down can only lose coverage. Densifying keeps the original beats as a subset, so phase is
        preserved without any snapping."""
        ab = np.asarray(ab); tempo = np.asarray(tempo)
        R = sorted(R, key=lambda x: x[0])
        if len(R) < 2:
            return R
        offs = [o for o, _ in R]
        ends = offs[1:] + [last_ms]

        def span(i):
            s = max(0, int(offs[i] / FRAME_MS)); e = min(len(ab), int(ends[i] / FRAME_MS))
            return s, (s if e < s else e)

        # confidence = mean beat-activation AT the decoded beats of the segment (tempo-
        # invariant). A frame-mean is biased against slow sections (sparser peaks -> lower
        # mean), so genuine slow sections failed the gate and got folded up an octave.
        bt = np.asarray(bt_list, float) if bt_list is not None else None

        def conf(i):
            s, e = span(i)
            if bt is not None:
                sel = bt[(bt >= offs[i]) & (bt < ends[i])]
                if len(sel):
                    fr = np.clip(np.round(sel / FRAME_MS).astype(int), 1, len(ab) - 2)
                    return float(np.maximum(ab[fr - 1], np.maximum(ab[fr], ab[fr + 1])).mean())
            return float(ab[s:e].mean()) if e > s else 0.0

        def tmass(i, mpb):                       # avg tempo-head prob within +-2 BPM of mpb
            s, e = span(i)
            b = int(round(60000.0 / mpb)) - int(BPM_MIN)
            if e <= s or b < 2 or b >= tempo.shape[1] - 2:
                return 0.0
            return float(tempo[s:e, b - 2:b + 3].sum(1).mean())

        confs = [conf(i) for i in range(len(R))]
        durs = [ends[i] - offs[i] for i in range(len(R))]
        anchor = R[int(np.argmax([c * d for c, d in zip(confs, durs)]))][1]   # dominant stream tempo
        out = []
        for i, (off, mpb) in enumerate(R):
            new = mpb
            if mpb > 1.0 and anchor > 1.0:
                while new > anchor * self.anchor_ratio: new /= 2.0
                while new < anchor / self.anchor_ratio: new *= 2.0
            # DENSIFY-ONLY: fold a too-slow (halved) redline UP toward the anchor so the beats it
            # dropped during the break come back (new mpb is smaller = more beats, a SUPERSET of
            # the originals so phase is preserved for free). NEVER thin a doubled redline: a
            # doubled grid already covers every real beat, octave-doubling is don't-care, and
            # folding it down only risks losing coverage. A genuine slow change -- high confidence
            # AND no tempo-head mass at the faster octave -- is left alone.
            if new < mpb - 1e-6:
                # genuine slow section = strong beats AND ~zero tempo-head support at the
                # folded (faster) octave. Half-time keeps ~15-20% mass at the fast octave
                # (bimodal) -> folds; a silent break fails the conf gate -> folds; a real
                # tempo drop has strong beats + no fast-octave mass -> protected.
                genuine = confs[i] >= self.anchor_conf and tmass(i, new) < 0.10
                if not genuine:
                    mpb = new
            out.append((off, mpb))
        return out

    @staticmethod
    def _fold_ratio(r):
        """fold a tempo ratio toward ~1 by octaves -> distance from unison/octave (0 = same pulse)."""
        if r <= 0:
            return 1.0
        while r > 1.42:
            r /= 2.0
        while r < 0.71:
            r *= 2.0
        return abs(r - 1.0)

    def _prune_crumbs(self, out, bt_list, ab, last_ms):
        """Drop low-confidence SAME-PULSE crumb redlines. out=[(off,mpb,meter)] -> pruned. A non-
        first redline is dropped when its span-mean beat-activation < prune_conf AND its tempo folds
        to the confident previous KEPT grid (unison or octave, |fold| <= prune_ratio_tol) -- i.e. a
        redundant re-anchor or an octave-confused fade crumb. Empty (silent) spans are dropped. A
        low-conf span at a genuinely different tempo is a real soft change and is KEPT. Decoded-beat
        residual is deliberately NOT consulted: in fades the decoded beats are noise (the crumb fits
        THEM better -- that is why the solver emitted it -- but the clean human grid is better served
        by extending the stable confident neighbour)."""
        R = sorted(out, key=lambda r: r[0])
        if len(R) < 2:
            return out
        ab = np.asarray(ab)
        bt = np.asarray(bt_list, float)
        offs = [r[0] for r in R]
        ends = offs[1:] + [last_ms]
        kept = [R[0]]
        for i in range(1, len(R)):
            off, mpb, meter = R[i]
            sel = bt[(bt >= offs[i] - 1) & (bt < ends[i])]
            if len(sel):
                fr = np.clip(np.round(sel / FRAME_MS).astype(int), 1, len(ab) - 2)
                conf = float(np.maximum(ab[fr - 1], np.maximum(ab[fr], ab[fr + 1])).mean())
            else:
                conf = 0.0
            if conf >= self.prune_conf:
                kept.append(R[i]); continue
            if len(sel) == 0:
                continue                                              # silent span -> drop
            prev = kept[-1]
            if prev[1] > 0 and self._fold_ratio(mpb / prev[1]) <= self.prune_ratio_tol:
                continue                                              # same-pulse crumb -> drop
            kept.append(R[i])                                         # genuine soft change -> keep
        return kept

    # ============================= RUBATO SPECIALIST =============================
    @staticmethod
    def _close_runs(mask, close=2, minlen=4):
        """Binary-close gaps <= `close` between True beats, then drop True runs shorter than
        `minlen`. Keeps drift regions contiguous and ignores 1-2 beat detector flickers."""
        m = mask.copy()
        idx = np.where(m)[0]
        if len(idx) == 0:
            return m
        for a, b in zip(idx[:-1], idx[1:]):
            if 0 < b - a <= close + 1:
                m[a:b] = True
        out = m.copy(); s = None
        for k in range(len(m) + 1):
            if k < len(m) and m[k]:
                if s is None:
                    s = k
            else:
                if s is not None and k - s < minlen:
                    out[s:k] = False
                s = None
        return out

    def _drift_mask(self, bt):
        """LOCAL rubato detector. Median-filter the inter-beat intervals (removes per-beat decoder
        JITTER), then a beat is 'drift' only if its smoothed-tempo window is a STEEP, CLEAN, MONOTONIC
        ramp: a least-squares line over the window has high R^2 (it really is a ramp, not noise/wobble/
        model imprecision) AND spans > ~6% of the tempo (a genuine ritardando/accelerando, not the
        small sustained position error the model carries even on constant tempo). This deliberately
        targets the rubato that segment solvers fail on -- continuous accel/decel -- and leaves
        constant tempo, hard steps, wobble and decode glitches to the clean DP base. Returns a per-beat
        boolean mask (gap-closed + short-run filtered)."""
        from redline.npsignal import medfilt
        N = len(bt)
        drift = np.zeros(N, bool)
        if N < 12:
            return drift
        ibi = np.diff(np.asarray(bt, float))
        sm = medfilt(ibi, 5)
        medibi = float(np.median(sm)) or 1.0
        w = 5
        for k in range(len(ibi)):
            lo, hi = max(0, k - w), min(len(ibi), k + w + 1)
            seg = sm[lo:hi]
            if len(seg) < 8:
                continue
            x = np.arange(len(seg), dtype=float)
            a, b = np.polyfit(x, seg, 1)
            pred = a * x + b
            ss_tot = float(np.sum((seg - seg.mean()) ** 2)) + 1e-9
            r2 = 1.0 - float(np.sum((seg - pred) ** 2)) / ss_tot
            span = abs(a * (len(seg) - 1)) / medibi
            if span > 0.06 and r2 > 0.75:               # steep, clean monotonic ramp = real rubato
                drift[k] = True
                drift[min(N - 1, k + 1)] = True
        # require a SUSTAINED ramp (>= 8 beats): brief 4-5 beat ramp-looking blips are usually decode
        # glitches (octave-doubling smoothed into a fake ramp), and a real rubato passage worth
        # per-beat tracking lasts longer than that.
        return self._close_runs(drift, close=2, minlen=8)

    def _segment_rubato(self, curves, bt_list):
        """RUBATO SPECIALIST (manual --solver rubato): a grid that aligns with the beats no matter
        what (over-segmentation accepted), staying as clean as possible.

        Hybrid of a clean DP base and per-beat overlay, split by a LOCAL drift detector (_drift_mask):
          - CLEAN spans (constant / syncopated / hard tempo steps / decode glitches): timed by DP
            (minimax offset + musical BPM) -> the cleanest, most precise form.
          - DRIFT spans (genuine gradual tempo movement -- ritardando/accelerando/wobble): replaced
            with PER-BEAT redlines (exact gap each, 0 residual) -> the grid follows every beat.
        Because the detector is LOCAL and excludes jitter/steps/decode-doubling, a constant song
        stays ~entirely DP (clean, == DP precision) and only the truly floating regions go per-beat.
        The per-beat mask is stored so downbeat-phase snapping is applied to the clean DP spans
        but NOT to the per-beat ones (which must stay exactly on their beat)."""
        bt = np.asarray(bt_list, float)
        N = len(bt)
        us = self.unified_solver
        if N < 3:
            O, P = us._optimize_segment(bt) if N >= 2 else (float(bt[0]) if N else 0.0, 500.0)
            self._rubato_perbeat_mask = [False]
            return [(O, P)]
        # Whole-song DP base (the clean, precise fit) is computed ONCE and kept INTACT.
        # The per-beat overlay is NON-DESTRUCTIVE: only the beats inside a drift run are replaced, and
        # the DP grid is re-anchored in-phase right after each run. So a detector false-positive costs
        # only its own few beats locally -- it can never re-phase / break a whole clean section (the
        # destructive failure mode of re-running DP on each sub-span).
        boundaries = self._get_confirmed_boundaries(curves["change"], bt_list) if self.use_ibi_change else []
        base = sorted(us.process_beats_dp(bt_list, boundaries), key=lambda x: x[0])
        if not base:
            O, P = us._optimize_segment(bt)
            self._rubato_perbeat_mask = [False]
            return [(O, P)]
        drift = self._drift_mask(bt)
        runs = []
        s = None
        for k in range(N + 1):
            if k < N and drift[k]:
                if s is None:
                    s = k
            elif s is not None:
                runs.append((s, k)); s = None

        def active(t_ms):                                    # DP redline governing time t_ms
            a = base[0]
            for O, P in base:
                if O <= t_ms + 1.0:
                    a = (O, P)
                else:
                    break
            return a

        def in_drift(t_ms):
            for a, b in runs:
                if bt[a] - 1.0 <= t_ms < bt[min(b, N - 1)]:
                    return True
            return False

        R, mask = [], []
        for O, P in base:                                    # keep DP redlines outside drift runs
            if not in_drift(O):
                R.append((O, P)); mask.append(False)
        for a, b in runs:                                    # per-beat the drift beats
            for t in range(a, min(b, N - 1)):
                g = float(bt[t + 1] - bt[t])
                if g > 1.0:
                    R.append((float(bt[t]), g)); mask.append(True)
            if b < N:                                        # resume DP grid in-phase after the run
                O0, P0 = active(bt[min(b, N - 1)])
                if P0 > 1:
                    newO = O0 + round((bt[min(b, N - 1)] - O0) / P0) * P0
                    R.append((float(newO), float(P0))); mask.append(False)
        order = sorted(range(len(R)), key=lambda t: R[t][0])
        R = [R[t] for t in order]; mask = [mask[t] for t in order]
        self._rubato_perbeat_mask = mask
        return R

    def predict_from_curves(self, curves, return_beats=False):
        ab, ad, ao = curves["beat"], curves["down"], curves["off"]
        bt_list = self.decode_beats(curves)
        if len(bt_list) < 8:
            return ([] if not return_beats else ([], []))

        is_rubato = (self.solver == "rubato")
        if self.solver == "chg":
            R = self._segment_chg(curves, bt_list)
        elif self.solver == "rubato":
            R = self._segment_rubato(curves, bt_list)
        elif self.solver == "tempo":
            R = self._segment_tempo(curves, bt_list)
        elif self.solver == "chgval":
            R = self._segment_chgval(curves, bt_list)
        else:
            boundaries = []
            if self.use_ibi_change:
                boundaries = self._get_confirmed_boundaries(curves["change"], bt_list)
            # minimax + clean-BPM snapping is more robust than a least-squares offset over
            # DP's long merged segments.
            R = self.unified_solver.process_beats_dp(bt_list, boundaries)

        if not R:
            return ([] if not return_beats else ([], bt_list))

        # Guarantee sorted order so that phase alignment processing runs accurately
        R = sorted(R, key=lambda x: x[0])

        # --- CONFIDENCE-GATED OCTAVE ANCHOR ---
        # Hold the established stream tempo through break/half-time octave excursions (metal
        # "beat chills"); leaves genuine high-confidence tempo changes untouched.
        if self.octave_anchor and "tempo" in curves and not is_rubato:
            R = self._octave_anchor(R, curves["beat"], curves["tempo"], len(ao) * FRAME_MS,
                                    bt_list=bt_list)

        # --- 1. RUBATO FALLBACK FOR LIVE MUSIC ---
        # Since the DP solver limits max error to 15ms, the median residual will rarely exceed ~10ms.
        # Lowering the threshold to 8.0ms allows us to catch live drift properly.
        rubato_resid_th = self.rubato_th
        stride = self.rubato_stride
        t_arr = np.asarray(bt_list)
        R_rubato = []
        for i, (off, mpb) in enumerate([] if is_rubato else R):
            nxt = R[i + 1][0] if i + 1 < len(R) else len(ao) * FRAME_MS
            sel = t_arr[(t_arr >= off - 1.0) & (t_arr < nxt)]

            if self.use_rubato and len(sel) >= 2 * stride and mpb > 1:
                resid = np.abs(sel - (off + np.round((sel - off) / mpb) * mpb))
                frs = np.clip((sel / FRAME_MS).astype(int), 0, len(ab) - 1)
                strong = float(np.mean(ab[frs])) >= 0.6

                # only re-anchor where a constant fit genuinely drifts (gated). Coarser stride =
                # fewer redlines while still tracking drift -> recover beat<10 without spray.
                if float(np.median(resid)) > rubato_resid_th and strong:
                    for b_idx in range(0, len(sel) - stride, stride):
                        b0 = sel[b_idx]
                        b1 = sel[b_idx + stride]
                        if 240.0 < b1 - b0 < 16000.0:
                            R_rubato.append((float(b0), float((b1 - b0) / stride)))
                    continue
                    
            R_rubato.append((off, mpb))

        R = R if is_rubato else R_rubato

        # --- 2. METER ESTIMATION ---
        # vote meter per segment from the downbeat head (the rubato solver's greedy-grown segments
        # are now long enough to vote, so meter changes like 5/4 are captured the same as other solvers)
        meters = self._meter_from_downbeat(R, bt_list, ad, last_ms=len(ao) * FRAME_MS)

        # --- 3. REFINE PLACEMENT WITH CHANGE HEAD ---
        refined_mask = None
        if self.refine_placement and "change" in curves and not is_rubato:
            R, refined_mask = self._refine_placement_with_change_head(
                R, bt_list, curves["change"], conf_thresh=getattr(self, 'conf_thresh', 0.70)
            )

        # --- 4. SNAP TO DOWNBEAT PHASE ---
        # snap the clean (DP-base) redlines to the downbeat for rankability; the rubato per-beat
        # overlay lines must NOT be phase-shifted off their beat, so they pass through the
        # refined_mask=True path (snap left untouched).
        if is_rubato:
            pb_mask = (list(self._rubato_perbeat_mask) + [False] * len(R))[:len(R)]
            R = self._snap_downbeat_phase(R, meters, ad, last_ms=len(ao) * FRAME_MS, refined_mask=pb_mask)
        else:
            R = self._snap_downbeat_phase(R, meters, ad, last_ms=len(ao) * FRAME_MS, refined_mask=refined_mask)

        # --- 5. FIRST-ANCHOR CONVENTION ---
        # Slide the very first redline forward to the first true sustained downbeat (ignores quiet intros)
        if not is_rubato and len(R) > 0 and len(bt_list) > 0:
            off0, mpb0 = R[0]
            nxt0 = R[1][0] if len(R) > 1 else float('inf')
            
            bscore = [ab[min(len(ab) - 1, max(0, int(round(b / FRAME_MS))))] for b in bt_list]
            dscore = [ad[min(len(ad) - 1, max(0, int(round(b / FRAME_MS))))] for b in bt_list]
            
            idxs = [i for i, b in enumerate(bt_list) if b < nxt0]
            k0 = None
            
            for i in idxs:
                w = bscore[i:i + 8]
                if len(w) >= 4 and float(np.mean(w)) >= 0.5:
                    k0 = i
                    break
                    
            if k0 is not None:
                cand = idxs[idxs.index(k0):min(len(idxs), idxs.index(k0) + 4)]
                if cand:
                    best_down_idx = max(cand, key=lambda i: dscore[i])
                    new_off = off0 + round((bt_list[best_down_idx] - off0) / mpb0) * mpb0
                    
                    if new_off > off0:
                        R[0] = (new_off, mpb0)

        out = [(off, mpb, m) for (off, mpb), m in zip(R, meters)]

        # --- 6. CRUMB DEDUP ---
        # refine/first-anchor can slide two redlines onto (nearly) the same spot: a leftover
        # phantom/crumb line governing <2 beats before the real anchor. Solvers never emit
        # segments this short (min_seg_beats>=4), so any such pair is an artifact -> keep the
        # LATER line (the refined/real anchor). Rubato per-beat overlays are exempt.
        if not is_rubato and len(out) > 1:
            dedup = []
            for k, (off, mpb, m) in enumerate(out):
                if k + 1 < len(out) and mpb > 1 and (out[k + 1][0] - off) < 1.9 * mpb:
                    continue
                dedup.append((off, mpb, m))
            out = dedup

        # --- 6b. LOW-CONFIDENCE CRUMB PRUNE ---
        # The solver emits crumb / octave-halved redlines on fading outros and soft bridges (low
        # beat-activation, no hit objects there -> mappers don't place them). They are either
        # redundant (the previous confident grid already covers the beats) or harmful (a halved/
        # doubled crumb) -- removing them re-aligns the grid with the real beats. Drop a NON-FIRST
        # low-conf redline only when its tempo is the SAME PULSE as the confident previous grid
        # (unison or octave fold); a low-conf span at a genuinely different tempo is a real soft
        # change and is kept. Runs before the fit-gate overlay so live per-beat overlays are exempt.
        if self.prune_crumbs and not is_rubato and len(out) > 1:
            out = self._prune_crumbs(out, bt_list, ab, last_ms=len(ao) * FRAME_MS)

        # --- 7. FIT-FAILURE OVERLAY (live timing) ---
        # Wherever a constant grid provably cannot fit the decoded beats (live drummers
        # drifting a few BPM per phrase), lay per-beat redlines through the misfit runs and
        # re-anchor the base grid in-phase after. Gated at the segment level (a high
        # fraction of strong beats must misfit) so clean maps are untouched by construction.
        # CONSTANT-PREFERENCE GUARD: if ONE line (the longest-governing segment, extended
        # over the whole song) already fits >=const_cover of the decoded beats within 10ms,
        # the map is constant-like and dense overlays are SPAM by the mapper convention (the
        # GT mapper fit one line too; measured: 244-line fitgate carpet bought +2 p10 over
        # that single line). Suppresses fitgate/live-reanchor/notation; genuine tempo CHANGES
        # still pass through the ratio-gated red re-anchors.
        const_like = False
        if not is_rubato and len(out) > 0 and len(bt_list) >= 32:
            _R = sorted(out)
            _ln = [(_R[i + 1][0] if i + 1 < len(_R) else len(ao) * FRAME_MS) - _R[i][0]
                   for i in range(len(_R))]
            _o, _m, _ = _R[int(np.argmax(_ln))]
            if _m > 1:
                _bt = np.asarray(bt_list, float)
                _res = np.abs(_bt - (_o + np.round((_bt - _o) / _m) * _m))
                const_like = float((_res < 10.0).mean()) >= self.const_cover

        if self.fitgate and not is_rubato and not const_like and len(out) > 0 and len(bt_list) >= 12:
            out = self._fitgate_overlay(out, bt_list, ab)

        # --- 8. LIVE RE-ANCHOR (barline convention) ---
        # Human convention on live/drifting songs: re-anchor a redline at (nearly) EVERY
        # barline, not only where a constant grid provably fails. Gate is GLOBAL (fraction
        # of decoded beats >live_gate_resid off the final grid) so constant maps are
        # untouched; fired maps get one redline per voted downbeat (meter {3,4} vote on the
        # down head), offset/mpb snapped back to the existing grid wherever it already fits
        # (protects beat accuracy), skipping jittery-decode measures (median local-IBI
        # deviation > live_jit).
        red_ppm = self._red_peaks_per_min(curves.get("red")) if isinstance(curves, dict) else None
        head_says_live = red_ppm is None or red_ppm >= self.red_live_ppm
        if self.live_reanchor and not is_rubato and not const_like and head_says_live \
                and len(out) > 0 and len(bt_list) >= 64:
            out = self._live_reanchor(out, bt_list, ad, last_ms=len(ao) * FRAME_MS)

        # --- 9. MUSIC-START ANCHOR ---
        # Mappers anchor their FIRST redline at the barline where the mapped music kicks in
        # (post-intro): that barline usually sits on the already-detected beat lattice but
        # can land many beats after the first decoded beat (soft/fx intros). Add
        # ONE grid-exact redline at the measure-lattice point nearest the first sustained-
        # strong beat (never moves the grid -> p10 unchanged; recovers the mapper's anchor).
        if self.anchor_music and not is_rubato and len(out) > 0 and len(bt_list) >= 16:
            out = self._music_anchor(out, bt_list, ab)

        # --- 10. NOTATION-HEAD PLACEMENT (red head) ---
        # Models trained with --red_head emit curves["red"]: per-frame "a mapper writes a
        # redline HERE" (supervised on ALL GT redlines incl. convention re-anchors, unlike
        # the major-only change head). Peak-pick, snap each peak to the nearest decoded
        # beat; where the existing grid already fits, the emitted redline is GRID-EXACT
        # (offset+mpb from the grid -> pure notation, p10 untouched); where it misfits, the
        # peak RE-ANCHORS with a local mpb (accelerando/rubato). Older ckpts have no "red"
        # curve -> this step is inert.
        if self.red_place and not is_rubato and isinstance(curves, dict) and \
                curves.get("red") is not None and len(out) > 0 and len(bt_list) >= 8:
            out = self._red_place(out, bt_list, np.asarray(curves["red"], float), ab,
                                  last_ms=len(ao) * FRAME_MS, allow_notation=not const_like)

        if self.use_meter2 and not is_rubato:
            out = self._progressive_meter_parser(out, ad, FRAME_MS)

        # --- 11. NOTATION PARSIMONY (collapse) ---
        # const_like (step 7) stops the dense overlays from being ADDED, but the segmentation
        # that ran before it already emitted its tracking re-anchors, so a song a human wrote
        # with one redline still ships with many. If one refitted grid explains const_collapse
        # _cover of the decoded beats, notate it the way the mapper did: a single line.
        # Measured on the 366-map genre bench (held-out half): P +8.3, F1 +3.1, p10 +0.6,
        # and the drift tail (>=11 GT redlines) is untouched because it never passes the gate.
        collapsed = False
        # The notation head is the model's own verdict on whether a mapper wrote redlines
        # here for CONVENTION (section re-anchors) rather than tempo change. A song whose
        # beats fit one grid can still be a many-redline map for that reason, so a quiet
        # red head is required before collapsing.
        quiet_head = red_ppm is None or red_ppm < self.const_collapse_ppm
        if self.const_collapse and not is_rubato and quiet_head \
                and len(out) > 0 and len(bt_list) >= 8:
            out, collapsed = self._const_collapse(out, bt_list)

        # --- 12. DOWNBEAT ANCHOR (collapsed output only) ---
        # Only meaningful when the collapse just made ONE line carry the whole song: that
        # line's beat choice IS the notation, and mappers write it on a downbeat. On output
        # that kept its tracking redlines, moving the first one is measurably harmful
        # (cohort redline-F1 -3.5) -- the solver's own anchors are already where the
        # segmentation put them, so leave them alone.
        if self.anchor_downbeat and collapsed and len(out) > 0 and len(bt_list) >= 8:
            out = self._anchor_downbeat(out, bt_list, ad)

        return (out if not return_beats else (out, bt_list))

    def _global_grid_fit(self, bt_list, mpb0):
        """One (offset, mpb) grid refitted over ALL decoded beats, outlier-trimmed IRLS.
        Returns (offset, mpb, coverage) where coverage = fraction of beats within 10 ms."""
        b = np.asarray(bt_list, float)
        if len(b) < 8 or not (mpb0 > 1):
            return None
        mpb, off = float(mpb0), float(b[0])
        for _ in range(12):
            k = np.round((b - off) / mpb)
            r = b - (off + k * mpb)
            s = max(float(np.median(np.abs(r))) * 3.0, 8.0)
            w = np.abs(r) <= s
            if int(w.sum()) < 8:
                break
            A = np.vstack([np.ones(int(w.sum())), k[w]]).T
            sol, *_ = np.linalg.lstsq(A, b[w], rcond=None)
            off, mpb = float(sol[0]), float(sol[1])
            if not (mpb > 1):
                return None
        k = np.round((b - off) / mpb)
        res = np.abs(b - (off + k * mpb))
        return off, mpb, float((res < 10.0).mean()), float(np.median(res))

    def _const_collapse(self, out, bt_list):
        """Replace tracking re-anchors with the single line a mapper would write, when one
        refitted grid covers the song. Deliberately gated well above the drift regime: a
        song whose beats genuinely wander never reaches the coverage bar, so live/rubato
        maps keep their per-barline notation."""
        R = sorted(out, key=lambda x: x[0])
        f = self._global_grid_fit(bt_list, R[0][1])
        if f is None:
            return out, False
        # Coverage counts beats inside 10 ms, so a handful of mis-decoded beats can veto a
        # song whose tempo is plainly constant. The MEDIAN residual is robust to exactly
        # that: measured on the bench, constant-GT songs the coverage gate rejected sit at
        # 3.5 ms median while genuine drift sits at 42.7 ms.
        fires = f[2] >= self.const_collapse_cover or \
            (self.const_collapse_resid > 0 and f[3] <= self.const_collapse_resid)
        if fires:
            return [(f[0], f[1], int(R[0][2]))], True
        return out, False

    def _anchor_downbeat(self, out, bt_list, ad):
        """Write the first redline on the measure start the model hears, not on whichever
        beat the tracker locked onto first (mapper convention; GT anchors agree).

        The offset only ever moves by a whole number of beats along its own grid, so every
        grid line stays exactly where it was and beat fidelity is untouched -- what changes
        is purely which beat carries the redline. Snapping to a raw decoded beat instead
        would drag the whole grid by that beat's residual, which p10 (a 10 ms threshold)
        punishes hard."""
        R = sorted(out, key=lambda x: x[0])
        b = np.asarray(bt_list, float)
        off, mpb, meter = R[0]
        meter = int(meter) if int(meter) >= 2 else 4
        if not (mpb > 1):
            return out
        end = R[1][0] if len(R) > 1 else (float(b[-1]) + mpb if len(b) else off)
        sel = b[(b >= off - mpb * 0.5) & (b < end)]
        if len(sel) == 0:
            return out
        k = np.round((sel - off) / mpb).astype(int)
        fr = np.clip(np.round(sel / FRAME_MS).astype(int), 0, len(ad) - 1)
        act = np.asarray(ad, float)[fr]
        best, bestv = None, -1.0
        for p in range(meter):
            m = (k % meter) == p
            if not m.any():
                continue
            v = float(act[m].mean())
            if v > bestv:
                best, bestv = p, v
        if best is None:
            return out
        kc = k[(k % meter) == best]
        if not len(kc):
            return out
        R[0] = (off + int(kc.min()) * mpb, mpb, meter)
        return sorted(R, key=lambda x: x[0])

    def _red_peaks_per_min(self, red):
        """Notation-head peak density (peaks/min at red_th) — the model's own verdict on
        whether a song is convention-timed. None if the model has no red head."""
        if red is None:
            return None
        c = np.asarray(red, float)
        if len(c) < 100:
            return 0.0
        n, i = 0, 1
        while i < len(c) - 1:
            if c[i] >= self.red_th and c[i] >= c[i - 1] and c[i] >= c[i + 1]:
                n += 1; i += 15
            else:
                i += 1
        return n / (len(c) * FRAME_MS / 60000.0)

    # ------------------------------------------------------------------ notation placement
    def _red_place(self, out, bt_list, red, ab, last_ms, allow_notation=True):
        bt = np.asarray(bt_list, float)
        ab = np.asarray(ab, float)
        act = ab[np.clip(np.round(bt / FRAME_MS).astype(int), 0, len(ab) - 1)]
        lines = self._grid_lines_of(out, last_ms)
        if len(lines) == 0:
            return out
        med_ibi = float(np.median(np.diff(bt))) if len(bt) > 4 else 400.0
        gap = max(2, int(0.7 * med_ibi / FRAME_MS))     # allow per-beat peaks (accelerando)
        # peak-pick the red curve
        pk = []
        i = 1
        while i < len(red) - 1:
            if red[i] >= self.red_th and red[i] >= red[i - 1] and red[i] >= red[i + 1]:
                pk.append(i * FRAME_MS)
                i += gap
            else:
                i += 1
        if not pk:
            return out

        R = sorted((float(o), float(m), int(mt)) for o, m, mt in out)
        offs = np.array([r[0] for r in R])

        def governing(t):
            k = int(np.searchsorted(offs, t + 1)) - 1
            return R[max(0, k)]

        # snap peaks to beat indices first so each re-anchor can see the NEXT peak (its
        # true span) instead of a fixed lookahead that lags on ramps.
        pj = []
        for tp in pk:
            j = int(np.abs(bt - tp).argmin())
            if abs(bt[j] - tp) <= 150.0 and (not pj or j > pj[-1]):
                pj.append(j)

        # CLEAN-MAP GATE: notation-only (same-BPM grid-exact) lines carry ZERO timing
        # information -- on a fixed-BPM song every one of them is spam (GT: ~1 redline).
        # They are the mapper convention ONLY where timing is actually unstable, so allow
        # them solely on maps whose grid broadly misfits (same stat as the live gate).
        # Re-anchors (real tempo changes) pass their own gates regardless.
        resid_all = np.abs(bt[:, None] - lines[None, :]).min(1)
        # the NOTATION HEAD is the arbiter of whether this song is convention-timed:
        # on constant songs it stays quiet (~3 peaks/min) even when the beat decode
        # wobbles enough to trip grid-residual stats (no residual statistic separates
        # a noisy-decode constant track from real live: both show frac6~0.3). Genuine
        # live/convention maps fire 9-110 peaks/min.
        red_ppm = self._red_peaks_per_min(red)
        map_live = allow_notation and red_ppm >= self.red_live_ppm and \
            float((resid_all > self.live_gate_resid).mean()) >= self.live_gate_frac
        if map_live:
            # jitter veto (same as _live_reanchor): a jittery-but-constant decode can trip
            # the misfit fraction; smooth-drifting rubato does not scatter local IBIs.
            ibi_all = np.diff(bt)
            if len(ibi_all) >= 8:
                med_w = np.median(np.lib.stride_tricks.sliding_window_view(ibi_all, 5), axis=1)
                dev = np.abs(ibi_all[2:2 + len(med_w)] - med_w)
                if float(np.median(dev)) > self.live_jit:
                    map_live = False

        res = list(R)
        reanchored = []                                  # (offset, beat idx) of placed re-anchors
        last_off = None                                  # last redline at/before the peak (any source)
        for pi, j in enumerate(pj):
            t = float(bt[j])
            _, gmpb, gmeter = governing(t)
            gdist = float(np.abs(lines - t).min())
            # local window fit: distinguishes a truly fitting grid from an octave-alias
            # touch (e.g. a stale 167 grid grazing every other beat of an 84.5 section)
            lo, hi = max(0, j - 2), min(len(bt), j + 6)
            w_res = np.abs(bt[lo:hi, None] - lines[None, :]).min(1)
            window_fits = float(np.median(w_res)) < self.live_snap
            prev = max([r[0] for r in res if r[0] <= t + 1] or [-1e12])
            if last_off is not None:
                prev = max(prev, last_off)
            if gdist < self.live_snap and window_fits:
                if not map_live:
                    continue                             # clean map: no notation-only lines
                # notation-only line (same BPM, on-grid): mappers place these at SECTION
                # anchors, not per-beat -- require a min gap so an over-eager head on
                # structureless audio (synthetic clicks) can't carpet a constant grid.
                if (t - prev) < self.red_min_gap_beats * gmpb:
                    continue
                # keep clear of existing redlines (they already notate this spot)
                if np.abs(offs - t).min() < 1.9 * gmpb:
                    continue
                off = float(lines[np.abs(lines - t).argmin()])   # grid-exact notation redline
                mpb = gmpb
            else:
                # RE-ANCHOR guard: only move off-grid when the misfit is SUSTAINED around the
                # peak (accelerando/rubato). A lone off-grid beat is decode jitter -> placing
                # a redline on it would corrupt a good grid (metal p10 regression).
                if window_fits:
                    continue                                     # isolated jitter, skip
                # decode-noise gate: in fades/soft outros decoded beats are NOISE (see
                # crumb-prune) -- re-anchoring onto them replaces a good grid with garbage.
                # Require confident beats around the peak (same bar as fitgate_act).
                if float(np.mean(act[lo:hi])) < self.fitgate_act:
                    continue
                # span to the NEXT red peak (mapper's own next re-anchor) -> mpb tracks the
                # ramp instead of averaging 4 beats ahead (which lags on accelerandos)
                k = (pj[pi + 1] - j) if pi + 1 < len(pj) else 4
                k = max(1, min(4, k, len(bt) - 1 - j))
                if len(bt) - 1 - j < 1:
                    continue
                off = t
                mpb = (bt[j + k] - bt[j]) / k
                # tempo-RATIO gate, SPAN-AWARE (crumb-prune lesson: ratio, not residual):
                # a local mpb estimate from 1-4 beats is only ~±1% precise. A same-tempo
                # re-anchor is harmless when the NEXT red peak corrects it 3-4 beats later
                # (live convention: ~1.5-2% steps per measure), but catastrophic when it
                # governs a long span (0.1% off over 64s = ~50ms drift on a constant map).
                # Skip only when the tempo is unchanged within noise AND no nearby peak
                # bounds the error.
                k_next = (pj[pi + 1] - j) if pi + 1 < len(pj) else 10 ** 6
                if abs(mpb - gmpb) / gmpb < 0.015 and k_next > 6:
                    continue
                # re-anchors may sit NEAR a stale solver redline (its mpb is wrong through a
                # ramp) -- only skip true duplicates (<0.5 beats), not the 1.9-beat notation
                # radius, else each stale line casts a dead zone the ramp can't re-anchor in.
                if np.abs(offs - off).min() < 0.5 * gmpb:
                    continue
            if not (20 < mpb < 2000):
                continue
            res.append((off, float(mpb), int(gmeter)))
            if gdist >= self.live_snap or not window_fits:
                reanchored.append((off, j))
            last_off = off

        # CLOSE RE-ANCHOR RUNS: a re-anchor's mpb comes from 1-4 decoded beats (±1%), so it
        # must never govern past its run -- when the ORIGINAL grid fits again, snap back with
        # ONE structural line on that grid (the rubato-overlay in-phase re-anchor). Without
        # this the run's last line drifts the whole tail (~50ms/min at 0.1% error). Runs
        # regardless of map_live: it is a correctness line, not notation.
        if reanchored:
            # per-beat original-grid window fit (resid_all precomputed above)
            fit_ok = np.array([
                float(np.median(resid_all[max(0, q - 2):min(len(bt), q + 6)])) < self.live_snap
                for q in range(len(bt))])
            offs_res = np.array(sorted(r[0] for r in res))
            for off_r, j_r in reanchored:
                nxt = offs_res[offs_res > off_r + 1]
                nxt_off = float(nxt[0]) if len(nxt) else 1e18
                _, gmpb_r, gmeter_r = governing(off_r)
                if nxt_off - off_r < 8 * gmpb_r:
                    continue                      # another line bounds this re-anchor already
                for q in range(j_r + 1, len(bt)):
                    if bt[q] >= nxt_off:
                        break
                    if fit_ok[q]:
                        tq = float(lines[np.abs(lines - bt[q]).argmin()])
                        if 20 < gmpb_r < 2000 and abs(tq - off_r) > 0.5 * gmpb_r:
                            res.append((tq, float(gmpb_r), int(gmeter_r)))
                        break
        res.sort()
        ded = []
        for r in res:
            if ded and abs(r[0] - ded[-1][0]) < 30:
                ded[-1] = r
            else:
                ded.append(r)
        return ded

    # ------------------------------------------------------------------ live re-anchor
    @staticmethod
    def _grid_lines_of(R, last_ms):
        R = sorted(R, key=lambda r: r[0]); lines = []
        for i, (off, mpb, _m) in enumerate(R):
            nxt = R[i + 1][0] if i + 1 < len(R) else last_ms
            if mpb <= 1:
                continue
            t = off
            while t < nxt + 1:
                lines.append(t); t += mpb
        return np.asarray(sorted(set(lines)))

    def _live_reanchor(self, out, bt_list, ad, last_ms):
        bt = np.asarray(bt_list, float)
        lines = self._grid_lines_of(out, last_ms)
        if len(lines) == 0:
            return out
        resid = np.abs(bt[:, None] - lines[None, :]).min(1)
        if float((resid > self.live_gate_resid).mean()) < self.live_gate_frac:
            return out                                            # constant map: untouched
        # global jitter veto: rubato DRIFTS smoothly (local IBIs cluster), jittery decode on a
        # constant song scatters them. Median local-IBI deviation across the map separates the
        # two; without this a noisy constant map trips the gate and gets per-measure spam.
        ibi_all = np.diff(bt)
        if len(ibi_all) >= 8:
            med = np.median(np.lib.stride_tricks.sliding_window_view(ibi_all, 5), axis=1) \
                if len(ibi_all) >= 5 else np.full(len(ibi_all), np.median(ibi_all))
            dev = np.abs(ibi_all[2:2 + len(med)] - med) if len(ibi_all) >= 5 else np.abs(ibi_all - med)
            if float(np.median(dev)) > self.live_jit:
                return out                                        # jitter, not rubato

        fr = np.clip(np.round(bt / FRAME_MS).astype(int), 0, len(ad) - 1)
        d = np.asarray(ad)[fr]
        # meter-{3,4} downbeat vote per 16-beat window (compound/waltz maps need meter 3)
        dbs = []
        W = 16
        for s in range(0, len(bt), W):
            e = min(s + W, len(bt))
            best, bsc = None, -1.0
            for m in (3, 4):
                if e - s < m:
                    continue
                for p in range(m):
                    sc = float(d[s + p:e:m].mean())
                    if sc > bsc:
                        bsc, best = sc, (m, p)
            if best is not None:
                m, p = best
                dbs.extend((i, m) for i in range(s + p, e, m))

        depR = sorted(out, key=lambda r: r[0])
        ibi = np.diff(bt)

        def dep_mpb_at(t):
            mpb = depR[0][1]
            for o, m, _mt in depR:
                if o <= t + 1:
                    mpb = m
                else:
                    break
            return mpb

        res = [(float(o), float(m), int(mt)) for o, m, mt in depR]
        for k in range(len(dbs)):
            i, meter = dbs[k]
            j = dbs[k + 1][0] if k + 1 < len(dbs) else min(i + meter, len(bt) - 1)
            if j <= i:
                continue
            t = float(bt[i])
            gdist = float(np.abs(lines - t).min())
            seg = bt[i:j + 1]
            seg_res = np.abs(seg[:, None] - lines[None, :]).min(1)
            fits = float(np.median(seg_res)) < self.live_snap
            if not fits:
                # erratic local IBIs = decode jitter, not rubato -> trust the grid, skip
                lo, hi = max(0, i - 2), min(len(ibi), j + 2)
                w = ibi[lo:hi]
                if len(w) >= 3 and float(np.median(np.abs(w - np.median(w)))) > self.live_jit:
                    continue
            off = float(lines[np.abs(lines - t).argmin()]) if gdist < self.live_snap else t
            mpb = float(dep_mpb_at(t)) if fits else float((bt[j] - bt[i]) / (j - i))
            if 20 < mpb < 2000:
                res.append((off, mpb, meter))
        res.sort()
        ded = []
        for r in res:
            if ded and abs(r[0] - ded[-1][0]) < 30:
                ded[-1] = r
            else:
                ded.append(r)
        return ded

    # ------------------------------------------------------------------ music-start anchor
    def _music_anchor(self, out, bt_list, ab):
        bt = np.asarray(bt_list, float)
        ab = np.asarray(ab)
        fr = np.clip(np.round(bt / FRAME_MS).astype(int), 0, len(ab) - 1)
        acts = ab[fr]
        th = max(0.35, 0.5 * float(np.median(acts)))
        k0 = None
        for i in range(len(bt) - 8):
            if acts[i] >= th and float(acts[i:i + 8].mean()) >= th:
                k0 = i
                break
        if k0 is None:
            return out
        R = sorted(out, key=lambda r: r[0])
        # governing redline at the music start (usually the first)
        gov = R[0]
        for r in R:
            if r[0] <= bt[k0] + 1:
                gov = r
            else:
                break
        goff, gmpb, gmeter = gov
        if gmpb <= 1:
            return out
        span = gmpb * max(1, int(gmeter))
        t = goff + round((bt[k0] - goff) / span) * span          # measure-lattice, grid-exact
        # only add when the anchor is meaningfully past the governing redline and not near
        # an existing one (else the mapper anchor is already covered)
        if t - goff < 2 * span or min(abs(t - r[0]) for r in R) < 1.9 * gmpb:
            return out
        R.append((float(t), float(gmpb), int(gmeter)))
        return sorted(R, key=lambda r: r[0])

    def _fitgate_overlay(self, out, bt_list, ab):
        bt = np.asarray(bt_list, float)
        ab = np.asarray(ab)
        N = len(bt)
        R = sorted(out, key=lambda r: r[0])
        offs = np.array([r[0] for r in R])
        resid = np.full(N, 1e9)
        segid = np.full(N, -1)
        for i, (o, m, _) in enumerate(R):
            if m <= 1:
                continue
            nxt = offs[i + 1] if i + 1 < len(R) else 1e18
            sel = (bt >= o - 1.0) & (bt < nxt)
            resid[sel] = np.abs((bt[sel] - o) - np.round((bt[sel] - o) / m) * m)
            segid[sel] = i
        fr = np.clip(np.round(bt / FRAME_MS).astype(int), 1, len(ab) - 2)
        act = np.maximum(ab[fr - 1], np.maximum(ab[fr], ab[fr + 1]))
        miss = resid > self.fitgate_resid
        allowed = np.zeros(N, bool)                       # segment-level unfittability gate
        for i in range(len(R)):
            sel = segid == i
            strong = sel & (act >= self.fitgate_act)
            if strong.sum() >= 12 and miss[strong].mean() >= self.fitgate_frac:
                allowed[sel] = True
        bad = miss & (act >= self.fitgate_act) & allowed
        bad = self._close_runs(bad, close=2, minlen=self.fitgate_run)
        if not bad.any():
            return out
        runs = []
        s = None
        for k in range(N + 1):
            if k < N and bad[k]:
                if s is None:
                    s = k
            elif s is not None:
                runs.append((s, k)); s = None

        def active(t_ms):
            a = R[0]
            for r in R:
                if r[0] <= t_ms + 1.0:
                    a = r
                else:
                    break
            return a

        def in_run(t_ms):
            return any(bt[a] - 1.0 <= t_ms < bt[min(b, N - 1)] for a, b in runs)

        newR = [r for r in R if not in_run(r[0])]
        for a, b in runs:
            a = max(0, a - 1)                                 # phase-in one beat early
            e_end = min(b, N - 1)
            off0, mpb0, m0 = active(bt[a])
            if self.overlay_group and mpb0 > 1:
                # HUMAN-STYLE spacing: greedily grow constant-BPM segments (anchor on the first
                # beat, LS slope) while every covered beat stays within overlay_tol of the line;
                # prefer cuts on MEASURE boundaries (beat-count phase carried from the base grid)
                # so lines land on downbeats every 1-3 measures like a human timer.
                p_a = int(round((bt[a] - off0) / mpb0)) % max(1, m0)
                s0 = a
                while s0 < e_end:
                    best_e = s0 + 1
                    e = s0 + 1
                    while e <= e_end and e - s0 <= self.overlay_max_beats:
                        k = np.arange(1, e - s0 + 1, dtype=float)
                        d = bt[s0 + 1:e + 1] - bt[s0]
                        P = float((k * d).sum() / (k * k).sum())
                        if P <= 1 or np.abs(d - k * P).max() > self.overlay_tol:
                            break
                        best_e = e
                        e += 1
                    cut = best_e
                    if m0 > 1:                                # cut on a whole measure if possible
                        for c in range(best_e, s0, -1):
                            if (p_a + (c - a)) % m0 == 0:
                                cut = c
                                break
                        if cut <= s0:
                            cut = best_e
                    k = np.arange(1, cut - s0 + 1, dtype=float)
                    d = bt[s0 + 1:cut + 1] - bt[s0]
                    P = float((k * d).sum() / (k * k).sum()) if len(k) else mpb0
                    if P > 1:
                        newR.append((float(bt[s0]), P, m0))
                    s0 = cut
            else:                                             # per-beat overlay (max precision)
                for t in range(a, e_end):
                    g = float(bt[t + 1] - bt[t])
                    if g > 1.0:
                        newR.append((float(bt[t]), g, m0))
            if b < N:                                         # resume base grid in-phase
                offb, mpbb, mb = active(bt[min(b, N - 1)])
                if mpbb > 1:
                    na = offb + round((bt[min(b, N - 1)] - offb) / mpbb) * mpbb
                    newR.append((float(na), float(mpbb), mb))
        return sorted(set(newR), key=lambda r: r[0])
        
    def _refine_placement_with_change_head(self, R, bt_list, ach, search_window_ms=2000.0, conf_thresh=0.70):
        if len(R) <= 1:
            return R, [False] * len(R)
            
        import numpy as np
        ach_arr = np.asarray(ach)
        
        # Handle fallback if output features are raw logits rather than probability
        if ach_arr.max() > 1.1 or ach_arr.min() < -0.1:
            ach_arr = 1.0 / (1.0 + np.exp(-np.clip(ach_arr, -100, 100)))
            
        bt_arr = np.asarray(bt_list)
        
        R_refined = [R[0]]
        refined_mask = [False]
        
        for i in range(1, len(R)):
            off, mpb = R[i]

            # Define search window in frames
            frame_start = max(0, int((off - search_window_ms) / FRAME_MS))
            frame_end = min(len(ach_arr), int((off + search_window_ms) / FRAME_MS) + 1)
            
            new_off = off
            is_refined = False
            
            if frame_end > frame_start:
                window = ach_arr[frame_start:frame_end]
                peak_idx_in_window = int(np.argmax(window))
                peak_conf = float(window[peak_idx_in_window])
                
                if peak_conf >= conf_thresh:
                    peak_time = (frame_start + peak_idx_in_window) * FRAME_MS
                    
                    # Find nearest decoded beat to the peak time
                    closest_beat_idx = int(np.argmin(np.abs(bt_arr - peak_time)))
                    closest_beat_time = float(bt_arr[closest_beat_idx])
                    
                    # Snap the redline to this beat by shifting the phase grid.
                    # We shift by exact multiples of mpb to perfectly preserve the solver's fractional phase.
                    shift = round((closest_beat_time - off) / mpb)
                    cand_off = off + shift * mpb

                    # Ensure we don't cross boundaries into other segments
                    prev_off = R_refined[-1][0]
                    nxt_off = R[i+1][0] if i+1 < len(R) else np.inf

                    if prev_off < cand_off < nxt_off and abs(cand_off - off) <= search_window_ms + 1000.0:
                        new_off = cand_off
                        is_refined = True

            R_refined.append((new_off, mpb))
            refined_mask.append(is_refined)
            
        return R_refined, refined_mask
            

    def _get_confirmed_boundaries(self, ach, bt_list):
        from redline import npsignal as ss
        bt = np.asarray(bt_list, float)
        if len(bt) < 12: return []
        
        pk = ss.find_peaks(np.asarray(ach), height=0.30, distance=int(500 / FRAME_MS))[0] * FRAME_MS
        clean_beats = []
        
        def ibi(t0, t1):
            seg = bt[(bt >= t0) & (bt < t1)]
            return float(np.median(np.diff(seg))) if len(seg) >= 4 else 0.0
            
        for p in pk:
            b, a = ibi(p - 2500, p - 300), ibi(p + 300, p + 2500)
            if b > 1 and a > 1 and abs(math.log(a / b)) > self.ibi_th:
                new_off = float(bt[int(np.argmin(np.abs(bt - p)))])
                clean_beats.append(new_off)
                
        return sorted(list(set(clean_beats)))

    def _meter_from_downbeat(self, R, bt_list, ad, last_ms, prior4_margin=0.05):
        bt = np.asarray(bt_list, float)
        dscore = np.array([ad[min(len(ad) - 1, max(0, int(round(b / FRAME_MS))))] for b in bt])
        meters = []
        for i, (off, mpb) in enumerate(R):
            nxt = R[i + 1][0] if i + 1 < len(R) else last_ms
            sel = (bt >= off - 1.0) & (bt < nxt)
            d = dscore[sel]
            n = len(d)
            if n < 9 or mpb <= 1:
                meters.append(4); continue
            sc = {}
            for M in (3, 4, 5, 6, 7):
                best = -1.0
                for p in range(M):
                    on = d[p::M]
                    if len(on) < 3: continue
                    mask = np.ones(n, bool); mask[p::M] = False
                    off_b = d[mask]
                    best = max(best, float(on.mean() - off_b.mean()))
                sc[M] = best
            m4 = sc.get(4, -1.0)
            best_m = max(sc, key=sc.get)
            if best_m != 4 and sc[best_m] - m4 < prior4_margin: best_m = 4
            meters.append(best_m)
        return meters
    
    def _progressive_meter_parser(self, out, ad, FRAME_MS):
        """Post-processes stable BPM segments to inject progressive time signature changes 

        measure-by-measure based on downbeat activations 'ad'.
        """
        import numpy as np
        progressive_redlines = []
        
        for idx, (off, mpb, m) in enumerate(out):
            # Only parse meter changes on stable, fast BPM sections (e.g., between 120 and 400 BPM)
            if mpb > 500.0 or mpb < 150.0:
                progressive_redlines.append((off, mpb, m))
                continue
                
            # Determine the end bound of this stable tempo section
            end_time = out[idx+1][0] if idx + 1 < len(out) else len(ad) * FRAME_MS
            
            curr_time = off
            active_meter = m
            
            # Add the starting redline
            progressive_redlines.append((curr_time, mpb, active_meter))
            
            while curr_time + (mpb * 3) < end_time:
                # Evaluate downbeat probability for 3/4, 4/4, 5/4, 6/4 starting from curr_time
                candidates = [3, 4, 5, 6]
                scores = []
                
                for cand in candidates:
                    cand_time = curr_time + (mpb * cand)
                    frame_idx = int(round(cand_time / FRAME_MS))
                    
                    # Check a tight 3-frame window around the candidate beat
                    if frame_idx < len(ad):
                        window = ad[max(0, frame_idx-1) : min(len(ad), frame_idx+2)]
                        scores.append(float(np.max(window)))
                    else:
                        scores.append(-1.0)
                
                # Find the best time signature candidate
                best_idx = int(np.argmax(scores))
                best_meter = candidates[best_idx]
                
                # --- THE FIX: Apply the meter change at the START of this measure ---
                if best_meter != active_meter:
                    # If we already have a redline at the start of this measure, update its meter
                    if progressive_redlines and progressive_redlines[-1][0] == curr_time:
                        progressive_redlines[-1] = (curr_time, mpb, best_meter)
                    else:
                        progressive_redlines.append((curr_time, mpb, best_meter))
                    active_meter = best_meter
                
                # Move the pointer forward by the length of the determined measure
                curr_time = curr_time + (mpb * best_meter)
                
        return sorted(list(set(progressive_redlines)), key=lambda x: x[0])

    def _snap_downbeat_phase(self, R, meters, ad, last_ms, refined_mask=None):
        if refined_mask is None:
            refined_mask = [False] * len(R)
            
        out = []
        for i, (off, mpb) in enumerate(R):
            m = meters[i] if i < len(meters) else 4
            nxt = R[i + 1][0] if i + 1 < len(R) else last_ms
            
            # If this segment was refined via change head, trust its anchor completely
            if refined_mask[i]:
                out.append((off, mpb))
                continue
                
            if mpb <= 1 or m < 2:
                out.append((off, mpb)); continue
            allbeats = np.arange(off, nxt, mpb)
            if len(allbeats) < m + 2:
                out.append((off, mpb)); continue
            bf = np.clip((allbeats / FRAME_MS).astype(int), 0, len(ad) - 1)
            bs = ad[bf]
            best_ph, best_sc = 0, -1e9
            for ph in range(m):
                on = bs[ph::m]
                mask = np.ones(len(bs), bool); mask[ph::m] = False
                off_b = bs[mask]
                if len(on) < 2 or len(off_b) < 2: continue
                sc = float(on.mean() - off_b.mean())
                if sc > best_sc:
                    best_sc, best_ph = sc, ph
            shift_back = (m - best_ph) % m
            actual_off = off - (shift_back * mpb)
            out.append((actual_off, mpb))
        return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--osz", required=True)
    args = ap.parse_args()
    from redline.audio_io import load_osz_audio
    from redline.osu_format import gt_from_osz
    gt, title, artist, _ = gt_from_osz(args.osz)
    sig, sr = load_osz_audio(args.osz, None)

    t = TimingSolver()
    R = t.predict(sig, sr)
    med = float(np.median([r[1] for r in R])) if R else 0
    print(f"  Unified   : {len(R):3d} redlines  ({60000/med:.1f} BPM)" if med else f"  Unified   : (none)")
        
    print(f"  GT        : {len(gt):3d} redlines  ({60000/np.median([r[1] for r in gt]):.1f} BPM)")