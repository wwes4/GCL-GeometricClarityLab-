import numpy as np
from typing import Dict, Any, Tuple
import time
from core.clarity_ratio import clarity_ratio
from utils.radial_displacer import radial_displacer

class Observer:
    def __init__(self):
        pass

    def blend(self, data: np.ndarray, band_mean: float = 21.0) -> np.ndarray:
        # Guard: if signal is near-constant (saturated tanh output from generative
        # path), std ≈ 0 and normalisation explodes. Detect and handle gracefully.
        data_std = float(np.std(data))
        if data_std < 1e-4:
            # Signal is saturated — return a mild mean-centred version
            # without amplification so the observer can still read direction
            direction = float(np.mean(data))
            return np.full_like(data, np.clip(direction * 0.5, -1.0, 1.0))

        eq = data - np.mean(data)
        eq = eq / (data_std + 1e-8)

        # Scale boost by signal health — saturated signals get less boost
        # so band_mean=65 (Data observer) can't explode a dense field
        signal_health = min(1.0, data_std / 0.5)
        boost = (band_mean ** 2) / np.pi * signal_health
        return np.clip(eq * (1 + boost * 0.45), -1.0, 1.0)


class MultiObserver:
    """
    Radicalized asymmetry: one observer for Matter (structure), one for Wave
    (breathing), one for Data (detail).

    Spin integration:
      When the generative path saturates the waveform (tanh → flat), amplitude
      carries no directional information. The bipolar lattice's ring_spin_signal
      is the fallback — it's positional, not amplitude-based, so it survives
      saturation. When is_generative and signal is saturated, the spin signal
      replaces the question/negation multipliers as the directional input.
    """
    def __init__(self, num_observers: int = 3):
        self.num_observers = num_observers
        self.observers = [Observer() for _ in range(num_observers)]
        self.bands = [21.0, 10.0, 65.0]  # Matter / Wave / Data
        self.cumulative_perturb = 0.0
        self.last_consensus_time = time.time()

    def interact(self, data: np.ndarray, prompt: str = "", iterations: int = 10,
                 prop_result: Dict[str, Any] = None) -> Tuple[float, float]:
        if len(data) == 0:
            return 0.0, 0.0

        prompt_lower = prompt.lower()
        is_question = prompt_lower.endswith("?") or any(
            w in prompt_lower for w in ["does", "do", "is", "are", "what", "who", "how"]
        )
        has_negation = any(w in prompt_lower for w in ["not", "no", "never"])

        radial_status     = radial_displacer.get_status()
        convergence_boost = max(0.3, radial_status.get("global_clarity", 1.0) * 0.8)

        # Detect generative mode
        is_generative = False
        phys_w = wave_w = data_w = 1.0
        if prop_result is not None and prop_result.get("mode") == "generative":
            is_generative = True
            phys_w = prop_result.get("phys_pers", 1.0)
            wave_w = prop_result.get("wave_pers", 1.0)
            data_w = prop_result.get("data_pers", 1.0)
            total  = phys_w + wave_w + data_w + 1e-8
            phys_w = (phys_w / total) * 3
            wave_w = (wave_w / total) * 3
            data_w = (data_w / total) * 3

        # ── Spin signal — directional fallback when generative field saturates ─
        # When all three Ouroboros passes converge to persistence >= 0.99,
        # the tanh output is flat and amplitude carries no directional info.
        # The bipolar lattice ring_spin_signal is positional — it survives
        # saturation completely. Read it directly from the pass result rather
        # than trying to detect saturation from the waveform std (which the
        # FFT linkage step in main.py masks).
        spin_signal      = 0.0
        signal_saturated = (
            is_generative and
            prop_result.get("phys_pers", 0.0) >= 0.97 and
            prop_result.get("wave_pers", 0.0) >= 0.97 and
            prop_result.get("data_pers", 0.0) >= 0.97
        )
        if is_generative and signal_saturated:
            try:
                from utils.bipolar_lattice import bipolar_lattice as _bl
                spin_signal = _bl._ring_spin_signal()
            except Exception:
                pass

        perceptions   = []
        role_weights  = [phys_w, wave_w, data_w]
        for i, obs in enumerate(self.observers):
            band_mean = self.bands[i % len(self.bands)]
            perc      = obs.blend(data, band_mean=band_mean)
            amp       = float(np.mean(perc))

            if is_generative:
                amp *= role_weights[i % 3] * convergence_boost
            else:
                if i == 0:
                    amp *= 1.8 * convergence_boost
                elif i == 1:
                    amp *= 1.2 * convergence_boost
                else:
                    amp *= 0.8 * convergence_boost

            if signal_saturated and is_generative:
                # Amplitude is flat — use spin signal as the directional input.
                # Spin is signed: positive = vertical build (agree), negative =
                # horizontal recognize (compress). Observer parity preserved:
                # odd observers follow spin, even observers invert it — same
                # odd/even asymmetry as the rest of the system.
                spin_dir = spin_signal if i % 2 == 1 else -spin_signal
                amp     += spin_dir * 0.35 * convergence_boost
            else:
                # Normal path — question and negation multipliers
                if is_question:
                    amp *= 1.2 if i % 2 == 0 else 0.85
                if has_negation:
                    amp *= -1.2 if i % 2 == 1 else 0.9

            perceptions.append(amp)

        consensus = np.mean(perceptions)
        for _ in range(iterations):
            props     = [p * (1.05 if i % 2 == 0 else 0.95) for i, p in enumerate(perceptions)]
            consensus = float(np.mean(props))

        final_consensus = float(np.clip(consensus, -1.0, 1.0))
        perturb         = float(np.std(perceptions) * 1.0)

        self.last_consensus_time  = time.time()
        self.cumulative_perturb   = np.clip(self.cumulative_perturb + perturb * 0.6, -1.0, 1.0)

        return final_consensus, self.cumulative_perturb

    def get_status(self) -> Dict[str, Any]:
        return {
            "num_observers":      self.num_observers,
            "last_consensus_time": self.last_consensus_time,
            "cumulative_perturb": round(self.cumulative_perturb, 6)
        }
