"""
Modular FEL Coulomb explosion simulation runner.

Usage:
    python run_template.py [config.json]

The config file controls all paths and simulation parameters.

Sweep mode: add a "sweep" section to the config to vary any explosion_sim
parameter independently. Each (param, value) pair becomes its own set of
simulations named {explosion_sim.name}_{param}_{value}. All sweep sims are
submitted to a single process pool together.

If config_sim.enabled is false, the config simulation is skipped and
frames are loaded from the directory specified in config_sim.frames_dir.
"""
import argparse
import itertools
import json
import os
import sys
import traceback
from multiprocessing import get_context

import explode_tools as et


_CONFIG_SIM_META_KEYS = {"enabled", "name", "pdb", "num_frames", "frames_dir"}
_EXPLOSION_SIM_META_KEYS = {"name"}


def _pool_init(ntomp):
    os.environ["OMP_NUM_THREADS"] = str(ntomp)


def _run_one_sim(args):
    sim, ionize, ntomp = args
    try:
        os.environ.setdefault("OMP_NUM_THREADS", str(ntomp))
        meta = sim.mdrun(ionize=ionize, num_cores=ntomp)
        return {"status": "ok", "meta": meta}
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "name": getattr(sim, "name", "unknown"),
            "simdir": getattr(sim, "path_simulation", None),
        }


def load_config(path):
    with open(path) as f:
        return json.load(f)


def run_config_simulation(name, pdb, gmx, mdp, ff, sim_params, frames_dir, num_frames):
    """Run the equilibration simulation and extract frames into frames_dir."""
    sim = et.simulation(name, gmx, mdp, ff, PDB=pdb)
    sim.set_parameters(**sim_params)
    sim.pdb2gmx(PDB=True)
    sim.grompp()
    sim.mdrun(ionize=False)
    sim.result.extract_frames(frames_dir, num_frames, as_pdb=False)
    return sim


def collect_frames(frames_dir, extension=".gro"):
    """Return sorted list of frame file paths from frames_dir."""
    if not os.path.isdir(frames_dir):
        return []
    return sorted(
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if f.endswith(extension)
    )


def _fmt(value):
    return f"{value:.3g}" if isinstance(value, float) else str(value)


def build_sweep_entries(es_cfg, sweep_cfg, sweep_mode="independent"):
    """
    Build (name, sim_params) pairs for all simulations to run.

    sweep_mode="independent": each parameter axis is varied on its own while
        all others stay at their base values. N1+N2+... entries total.
    sweep_mode="cartesian": all combinations of all axes are generated.
        N1*N2*... entries total.
    Without sweep: one entry with the base explosion_sim parameters.
    """
    base_name = es_cfg["name"]
    base_params = {k: v for k, v in es_cfg.items() if k not in _EXPLOSION_SIM_META_KEYS}

    if not sweep_cfg:
        return [(base_name, base_params)]

    if sweep_mode == "cartesian":
        keys = list(sweep_cfg.keys())
        entries = []
        for combo in itertools.product(*sweep_cfg.values()):
            params = {**base_params, **dict(zip(keys, combo))}
            name = base_name + "".join(f"_{k}_{_fmt(v)}" for k, v in zip(keys, combo))
            entries.append((name, params))
        return entries

    # independent (default)
    entries = []
    for param, values in sweep_cfg.items():
        for value in values:
            params = {**base_params, param: value}
            name = f"{base_name}_{param}_{_fmt(value)}"
            entries.append((name, params))
    return entries


def setup_explosion_sims(gmx, root, name, systems, mdp, ff, sim_params, atomic_models=None, top=None):
    """Create, configure, and grompp all explosion simulations for one parameter set."""
    handler = et.simulation_handler(gmx, root, name, systems, mdp, ff=ff)
    energy = sim_params.get("energy")
    for sim in handler.single_explosions:
        sim.set_parameters(**sim_params)
        if top is not None:
            sim.set_top(top)
        sim.grompp()
        if atomic_models is not None:
            sim.atomic_data_from_directory(atomic_models, energy=energy)
    return handler


def run_all_handlers(handlers, num_cores, ionize=True, ntomp=1, verbose=True):
    """
    Run all simulations across all handlers in a single shared process pool.
    Results are written back to each handler's single_explosions list.
    """
    tasks = [(sim, ionize, ntomp) for h in handlers for sim in h.single_explosions]
    results = []

    ctx = get_context("spawn")
    with ctx.Pool(
        processes=num_cores,
        initializer=_pool_init,
        initargs=(ntomp,),
        maxtasksperchild=1,
    ) as pool:
        for r in pool.imap_unordered(_run_one_sim, tasks):
            results.append(r)
            if verbose:
                if r["status"] == "ok":
                    print(f"[{r['meta']['name']}] OK (rc={r['meta']['returncode']})")
                else:
                    print(f"[{r.get('name', '?')}] ERROR: {r['error']}")

    for handler in handlers:
        for sim in handler.single_explosions:
            meta = next(
                (x["meta"] for x in results
                 if x["status"] == "ok" and x["meta"]["simdir"] == sim.path_simulation),
                None,
            )
            try:
                sim.result = et.result_handler(sim.path_simulation) if meta else None
            except Exception:
                sim.result = None

    return results


def main(config_path="config.json"):
    cfg = load_config(config_path)

    paths = cfg["paths"]
    gmx = paths["gmx_bin"]
    root = paths["root"]
    atomic_models = paths.get("atomic_models")
    mdp = paths["mdp"]
    ff = cfg["force_field"]

    os.chdir(root)

    # Phase 1: Config simulation (optional)
    cs_cfg = cfg.get("config_sim", {})
    frames_dir = os.path.join(root, cs_cfg.get("frames_dir", "frames"))

    config_sim = None
    if cs_cfg.get("enabled", False):
        sim_params = {k: v for k, v in cs_cfg.items() if k not in _CONFIG_SIM_META_KEYS}
        config_sim = run_config_simulation(
            name=cs_cfg["name"],
            pdb=cs_cfg["pdb"],
            gmx=gmx,
            mdp=mdp,
            ff=ff,
            sim_params=sim_params,
            frames_dir=frames_dir,
            num_frames=cs_cfg["num_frames"],
        )

    # Phase 2: Collect frames
    systems = collect_frames(frames_dir)
    if not systems:
        print(f"No .gro files found in {frames_dir}", file=sys.stderr)
        sys.exit(1)

    # Phase 3: Build parameter entries (base + sweep)
    es_cfg = cfg["explosion_sim"]
    sweep_cfg = cfg.get("sweep", {})
    sweep_mode = cfg.get("sweep_mode", "independent")
    entries = build_sweep_entries(es_cfg, sweep_cfg, sweep_mode)

    top = config_sim.TOP if config_sim is not None else None
    n_total = len(entries) * len(systems)
    print(f"Setting up {len(entries)} parameter set(s) × {len(systems)} frame(s) = {n_total} simulation(s)")

    # Phase 4: Setup all handlers
    handlers = []
    for sweep_name, sim_params in entries:
        handler = setup_explosion_sims(
            gmx=gmx,
            root=root,
            name=sweep_name,
            systems=systems,
            mdp=mdp,
            ff=ff,
            sim_params=sim_params,
            atomic_models=atomic_models,
            top=top,
        )
        handlers.append(handler)

    # Phase 5: Run everything in one pool
    run_cfg = cfg.get("run", {})
    results = run_all_handlers(
        handlers=handlers,
        num_cores=run_cfg.get("num_cores", 1),
        ionize=run_cfg.get("ionize", True),
        verbose=run_cfg.get("verbose", True),
    )
    return handlers, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FEL Coulomb explosion simulations.")
    parser.add_argument(
        "config",
        nargs="?",
        default="config.json",
        help="Path to JSON config file (default: config.json)",
    )
    args = parser.parse_args()
    main(args.config)
