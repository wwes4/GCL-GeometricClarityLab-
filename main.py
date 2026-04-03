"""
GeometricClarityLab - Interactive Test Runner (27-Symbol Memory Wired)
"""
import sys
from pathlib import Path
import numpy as np
import time

root = Path(__file__).parent.absolute()
sys.path.insert(0, str(root))

# Core + semantic
from core.clarity_ratio import clarity_ratio
from core.invariants import invariants
from core.safeguards import safeguards
from core.semantic_layer import semantic_layer

# Ouroboros field dynamics engine
from core.ouroboros_engine import ouroboros_engine

# Diagonal structure generator
from utils.diagonal_structure import diagonal_structure_generator

# Wave & radial layers
from wave.symbolic_wave import SymbolicWave
from wave.propagation import WavePropagator
from wave.vibration import VibrationPropagator
from utils.radial_displacer import radial_displacer   # 27-symbol lattice
from utils.bipolar_lattice import bipolar_lattice

# Memory, observer, triad, generator
from memory.geometric_memory import GeometricMemory
from observer.observer import MultiObserver
from observer.triad import Triad
from wave.generation import generator
from visualizer import update as viz_update

def print_header():
    print("\n" + "="*90)
    print("GeometricClarityLab - Interactive Test Runner (27-Symbol Lattice + Memory)")
    print("="*90)
    print(f"Clarity Ratio: {clarity_ratio.get_status()}")
    print(f"Invariants: {invariants.get_status()}")
    print(f"Safeguards: {safeguards.get_status()}")
    print(f"Radial Web: {radial_displacer.get_status()}")
    print(f"Bipolar Lattice: {bipolar_lattice.get_status()}")
    print(f"Ouroboros Engine: {ouroboros_engine.get_status()}")

def test_full_pipeline(custom_prompt: str = None):
    print("\n[2] Full Pipeline Test - Living Radial Web + 27-Symbol Memory")
   
    prompt = custom_prompt or "Clara loves to paint watercolours. What does Clara love to do?"
    print(f"Prompt: {prompt}")

    start_time = time.time()

    # 1. Matter: Symbolic Wave + Clarity Ratio
    sw = SymbolicWave()
    tri_data = sw.triangulate(prompt)
    tri_data["prompt"] = prompt
    clarity_ratio.measure(tri_data["width"], tri_data["height"], tri_data["total_triangles"], tri_data["n_original"])

    # Reset exhaust signature for this prompt
    bipolar_lattice.reset_exhaust()

    print(f"Box: {tri_data['width']}×{tri_data['height']} | Triangles: {tri_data['total_triangles']}")

    # 2. Radial Web: Living 27-symbol displacers
    radial = radial_displacer.generate_structure(prompt, tri_data, wave_amplitude=0.0)

    # 3. Wave: Propagation
    prop = WavePropagator()
    prop_result = prop.propagate(tri_data, steps=60)

    # Generative path decision gate — mirrors sensory asymmetry insight:
    # direct = recognition mode (fast), generative = reconstruction mode (closed loop).
    # Fires when persistence is high OR cross-turn recall has prior entries.
    recall_triggered = len(generator.memory_store) > 0
    if ouroboros_engine.should_go_generative(prop_result["persistence"], recall_triggered):
        prop_result = prop.propagate_generative(prop_result, tri_data, recall_triggered)

    # 4. Vibration + holographic linkage
    vib = VibrationPropagator()
    vib_amp = vib.propagate_vibration(prop_result['base_amplitude'], distance=10.0)
    
    waveform_sample = prop_result.get("waveform_sample", [0.1])
    numeric_wave = [x for x in waveform_sample if isinstance(x, (int, float))]
    linked_wave = vib.holographic_linkage(np.array(numeric_wave) * 10)

    # 5. Observer + Triad resonance
    obs = MultiObserver(num_observers=3)
    consensus, perturb = obs.interact(linked_wave, prompt=prompt, iterations=10,
                                      prop_result=prop_result)

    triad = Triad()
    _, triad_hist = triad.forward(linked_wave.reshape(1, -1) if len(linked_wave.shape) == 1 else linked_wave, prompt=prompt)

    # 6. Radial reactivity to actual wave
    radial_displacer.react_to_wave(linked_wave)
    radial = radial_displacer.generate_structure(prompt, tri_data, wave_amplitude=float(np.mean(np.abs(linked_wave))))

    # Bipolar lattice tension cycle — runs alongside radial web
    bipolar_lattice.react_to_wave(linked_wave)

    # 6-CYCLE REFINEMENT (multiples of 3 preserved)
    for _ in range(6):
        bipolar_lattice.apply_tension_cycle(
            float(np.mean(np.abs(linked_wave)))
        )

    # ACTIVE CORE PROPAGATION: each displacer emits its waveform → core reads band, cancels noise, propagates
    linked_wave = bipolar_lattice.band_emit_and_core_propagate(tri_data)

    # CLARITY TREND FIX: measure AFTER full refinement + core propagation
    clarity_ratio.measure(tri_data["width"], tri_data["height"], tri_data["total_triangles"], tri_data["n_original"])

    bipolar_status = bipolar_lattice.get_status()

    # 7. Memory etching — 27-symbol sequence + zero breaks
    mem = GeometricMemory(secret_phrase="resonance eternal")
    radial_symbols = radial.get("displacers", [{}])[0].get("symbol_sequence", []) if radial.get("displacers") else []
    radial_zeros = radial.get("displacers", [{}])[0].get("zero_breaks", []) if radial.get("displacers") else []
    mem.encode(prompt, radial_symbols, radial_zeros)

    key = mem.generate_key()
    recalled = mem.access(key)

    # 8. Convergence-driven generation
    answer = generator.generate(
        prompt=prompt,
        tri_data=tri_data,
        prop_result=prop_result,
        consensus=consensus,
        memory_recall=recalled
    )

    elapsed = time.time() - start_time

    print(f"\n--- Generated Answer (Convergence-Driven) ---")
    print(answer)
   
    print(f"\nPipeline Metrics (Living Web + 27-Symbol Memory):")
    print(f" Clarity Score     : {prop_result['clarity_ratio_score']:.4f}  (trend: {clarity_ratio.get_trend()['trend']})")
    print(f" Persistence       : {prop_result['persistence']:.4f}")
    print(f" Consensus         : {consensus:.4f}")
    print(f" Radial Convergence: {radial['web_convergence_score']:.4f}  ({radial['num_displacers']} displacers)")
    print(f" Memory Recall     : {'Success' if len(recalled) > 20 else 'Partial'}")
    print(f" Safeguards        : {'Reversal triggered' if safeguards.should_force_reversal(prop_result['persistence'], len(clarity_ratio.history)) else 'Clear'}")
    print(f" Elapsed           : {elapsed:.3f}s")
    print(f" Bipolar Core      : {bipolar_status['core_id']} "
          f"(score {bipolar_status['core_score']:.4f}, "
          f"strings {bipolar_status['active_strings']})")
    print(f" Geo Ticks         : {bipolar_status['geo_tick_count']}")
    mode = prop_result.get("mode", "direct")
    if mode == "generative":
        print(f" Generative Pass   : phys={prop_result.get('phys_pers',0):.3f} "
              f"wave={prop_result.get('wave_pers',0):.3f} "
              f"data={prop_result.get('data_pers',0):.3f} "
              f"consensus={prop_result.get('consensus_pers',0):.3f}")
    else:
        print(f" Mode              : {mode}")

    # Etch exhaust signature — geometric fingerprint of this prompt
    bipolar_lattice.etch_exhaust(prompt, tri_data.get("symbol_stream", []))
    sig = bipolar_lattice.get_exhaust_signature()
    if sig.sum() > 1e-10:
        print(f" Exhaust Signature : [{' '.join(f'{v:.3f}' for v in sig)}]")
        nearest = bipolar_lattice.nearest_exhaust(top_n=2)
        matches = [n for n in nearest if n["distance"] > 1e-6]
        if matches:
            best = matches[0]
            print(f" Nearest Prior     : dist={best['distance']:.4f} "
                  f"core={best['core_id']} \"{best['prompt'][:60]}\"")
        # Generate diagonal structure from exhaust signature
        structure = diagonal_structure_generator.generate(
            exhaust_signature = sig,
            ring_net_phase    = bipolar_lattice._ring_net_phase(),
            core_id           = bipolar_lattice.core_id,
            prompt            = prompt,
        )
        print(f" Diagonal Structure: complexity={structure.complexity_score:.4f} "
              f"dispersed={sum(1 for d in structure.diagonals if d.dispersed)}/{len(structure.diagonals)}")
        nearest_struct = diagonal_structure_generator.nearest(structure, top_n=1)
        if nearest_struct:
            best_s = nearest_struct[0]
            print(f" Nearest Structure : sim={best_s['similarity']:.4f} "
                  f"\"{best_s['prompt'][:60]}\"")
    else:
        print(f" Exhaust Signature : (no stabilizer activity)")

    # ── Visualizer update ─────────────────────────────────────────────────────
    # Build a lightweight waypoint snapshot for the ring panel
    try:
        wp_snapshot = [
            {
                "wp_id":      wp.wp_id,
                "role":       wp.role,
                "angle":      wp.angle,
                "radius":     wp.radius,
                "spin_phase": wp.spin_phase,
            }
            for wp in bipolar_lattice.waypoints
        ]
        _, pocket_conf = semantic_layer.extract_with_pocket_alignment(
            prompt, tri_data, prop_result
        )
        viz_update(
            prop_result       = prop_result,
            tri_data          = tri_data,
            bipolar_status    = bipolar_status,
            consensus         = consensus,
            pocket_conf       = float(pocket_conf),
            waypoints_snapshot= wp_snapshot,
        )
    except Exception as e:
        print(f" [Visualizer] {e}")

def test_raw_symbols(raw_input: str):
    """
    Raw symbol test — flat character sequence, no pocket insertion.
    Clean field readings with no boundary geometry interference.
    Use for systematic symbol experiments.
    """
    print("\n[4] Raw Symbol Test")
    print(f"Input: {raw_input!r}")

    start_time = time.time()

    sw       = SymbolicWave()
    tri_data = sw.triangulate_raw(raw_input)
    tri_data["prompt"] = raw_input
    clarity_ratio.measure(
        tri_data["width"], tri_data["height"],
        tri_data["total_triangles"], tri_data["n_original"]
    )

    # Reset exhaust for this raw test
    bipolar_lattice.reset_exhaust()

    print(f"Box: {tri_data['width']}×{tri_data['height']} | "
          f"Triangles: {tri_data['total_triangles']} | "
          f"Symbols: {tri_data['n_original']} | "
          f"Stream: {tri_data['symbol_stream'][:16]}{'...' if len(tri_data['symbol_stream']) > 16 else ''}")

    # Wave propagation
    prop        = WavePropagator()
    prop_result = prop.propagate(tri_data, steps=60)

    recall_triggered = len(generator.memory_store) > 0
    if ouroboros_engine.should_go_generative(prop_result["persistence"], recall_triggered):
        prop_result = prop.propagate_generative(prop_result, tri_data, recall_triggered)

    # Vibration
    vib         = VibrationPropagator()
    numeric_wave = [x for x in prop_result.get("waveform_sample", [0.1])
                    if isinstance(x, (int, float))]
    linked_wave = vib.holographic_linkage(np.array(numeric_wave) * 10)

    # Observer
    obs       = MultiObserver(num_observers=3)
    consensus, perturb = obs.interact(
        linked_wave, prompt=raw_input, iterations=10, prop_result=prop_result
    )

    # Bipolar lattice
    bipolar_lattice.react_to_wave(linked_wave)
    for _ in range(6):
        bipolar_lattice.apply_tension_cycle(float(np.mean(np.abs(linked_wave))))
    linked_wave  = bipolar_lattice.band_emit_and_core_propagate(tri_data)
    clarity_ratio.measure(
        tri_data["width"], tri_data["height"],
        tri_data["total_triangles"], tri_data["n_original"]
    )
    bipolar_status = bipolar_lattice.get_status()

    elapsed = time.time() - start_time

    print(f"\n--- Raw Symbol Field Readings ---")
    print(f" Persistence       : {prop_result['persistence']:.4f}")
    print(f" Consensus         : {consensus:.4f}")
    print(f" Clarity Score     : {prop_result['clarity_ratio_score']:.4f}")
    print(f" Bipolar Core      : {bipolar_status['core_id']} "
          f"(score {bipolar_status['core_score']:.4f})")
    print(f" Geo Ticks         : {bipolar_status['geo_tick_count']}")
    print(f" Ring Spin Signal  : {bipolar_status['ring_spin_signal']:.4f}")
    print(f" Field Stress      : {bipolar_status['field_stress']:.4f}")
    print(f" Fold Neg Signal   : {bipolar_status['fold_negotiation_signal']:.4f}")
    print(f" Zero Braking      : {bipolar_status['zero_braking']}")
    print(f" Mode              : {prop_result.get('mode', 'direct')}")
    print(f" Elapsed           : {elapsed:.3f}s")

    # Etch exhaust signature
    bipolar_lattice.etch_exhaust(raw_input, tri_data.get("symbol_stream", []))
    sig = bipolar_lattice.get_exhaust_signature()
    if sig.sum() > 1e-10:
        print(f" Exhaust Signature : [{' '.join(f'{v:.3f}' for v in sig)}]")
        nearest = bipolar_lattice.nearest_exhaust(top_n=2)
        matches = [n for n in nearest if n["distance"] > 1e-6]
        if matches:
            best = matches[0]
            print(f" Nearest Prior     : dist={best['distance']:.4f} "
                  f"\"{best['prompt'][:60]}\"")
        # Generate diagonal structure from exhaust signature
        structure = diagonal_structure_generator.generate(
            exhaust_signature = sig,
            ring_net_phase    = bipolar_lattice._ring_net_phase(),
            core_id           = bipolar_lattice.core_id,
            prompt            = raw_input,
        )
        print(f" Diagonal Structure: complexity={structure.complexity_score:.4f} "
              f"dispersed={sum(1 for d in structure.diagonals if d.dispersed)}/{len(structure.diagonals)}")
        nearest_struct = diagonal_structure_generator.nearest(structure, top_n=1)
        if nearest_struct:
            best_s = nearest_struct[0]
            print(f" Nearest Structure : sim={best_s['similarity']:.4f} "
                  f"\"{best_s['prompt'][:60]}\"")
    else:
        print(f" Exhaust Signature : (no stabilizer activity)")

    # Visualizer
    try:
        wp_snapshot = [
            {"wp_id": wp.wp_id, "role": wp.role, "angle": wp.angle,
             "radius": wp.radius, "spin_phase": wp.spin_phase}
            for wp in bipolar_lattice.waypoints
        ]
        viz_update(
            prop_result        = prop_result,
            tri_data           = tri_data,
            bipolar_status     = bipolar_status,
            consensus          = consensus,
            pocket_conf        = 0.0,   # no boundary geometry in raw input
            waypoints_snapshot = wp_snapshot,
        )
    except Exception as e:
        print(f" [Visualizer] {e}")

def main():
    print_header()
   
    while True:
        print("\nOptions:")
        print(" 1 - Box Scaling Test")
        print(" 2 - Full Pipeline Test (default prompt)")
        print(" 3 - Full Pipeline Test with custom prompt")
        print(" 4 - Raw Symbol Test (no pocket insertion)")
        print(" q - Quit")
       
        choice = input("\nChoose [1/2/3/4/q]: ").strip().lower()
       
        if choice == '1':
            print("\n[1] Box Scaling Test")
            sw = SymbolicWave()
            sizes = [16, 32, 64, 128, 256]
            for n in sizes:
                seq = list(range(1, n+1))
                data = sw.triangulate(seq)
                clarity_ratio.measure(data["width"], data["height"], data["total_triangles"], data["n_original"])
                print(f" n={n:3d} → Box: {data['width']:2d}×{data['height']:2d} | Triangles: {data['total_triangles']:3d} | Clarity: {clarity_ratio.current_ratio:.4f}")
        elif choice == '2':
            test_full_pipeline()
        elif choice == '3':
            prompt = input("Enter prompt: ").strip()
            if prompt:
                test_full_pipeline(prompt)
            else:
                print("Prompt cannot be empty.")
        elif choice == '4':
            raw = input("Enter raw symbol sequence: ").strip()
            if raw:
                test_raw_symbols(raw)
            else:
                print("Input cannot be empty.")
        elif choice == 'q':
            print("Goodbye. Lattice preserved.")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()