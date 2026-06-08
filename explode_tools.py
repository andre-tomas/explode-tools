import os
import shutil
import subprocess
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context

import healpy as hp
import MDAnalysis as md
import numpy as np
from Bio.PDB import PDBParser, PDBIO, Select
from MDAnalysis.auxiliary import EDR
from scipy.spatial.transform import Rotation as R

from . import leveldict as ld


UN = hp.UNSEEN



class Detector:
    def __init__(self, plane_size,detector_distance,num_bins,sigma=0,axis=1,detector_eff=1.0):
        self.plane_size = plane_size
        self.distance = detector_distance
        self.num_bins = num_bins
        self.sigma = sigma
        self.axis = axis
        self.detector_eff = detector_eff
        
    def __repr__(self):
        return (f"DetectorParameters(plane_size={self.plane_size}, "
                f"detector_distance={self.distance}, "
                f"num_bins={self.num_bins}, "
                f"sigma={self.sigma}, "
                f"axis={self.axis}, "
                f"detector_eff={self.detector_eff})")
    
    


def bin_plane_xaxis(xyz, detector, mass=None, mass_filter=None):
    """
    Bin unit vectors onto a detector plane at x = +L or x = -L.

    Parameters:
    xyz (numpy.ndarray): (N, 3) array of [x, y, z] coordinates.
    detector (Detector): Detector object containing plane_size, detector_distance, num_bins, and detector_eff.
    mass (numpy.ndarray, optional): Array of masses corresponding to the xyz coordinates.
    mass_filter (tuple, optional): (min_mass, max_mass) to filter the xyz coordinates by mass.

    Returns:
    bins (numpy.ndarray): (num_bins, num_bins) histogram of binned hits.
    hit_coords (numpy.ndarray): (N_hits, 3) array of [x_plane, y_bin_center, z_bin_center] for each hit.
    """
    # Optionally filter by mass
    if mass is not None and mass_filter is not None:
        mass = np.array(mass)
        idx = (mass > mass_filter[0]) & (mass < mass_filter[1])
        xyz = xyz[idx]

    # Only keep vectors pointing toward the detector
    L = detector.distance
    if L > 0:
        xyz = xyz[xyz[:, 0] > 0]
    else:
        xyz = xyz[xyz[:, 0] < 0]

    if xyz.shape[0] == 0:
        return np.zeros((detector.num_bins, detector.num_bins), dtype=int), np.empty((0, 3))

    # Project to detector plane at x = L (or -L)
    k = L / xyz[:, 0]
    hits = xyz * k[:, np.newaxis]  # shape (N, 3)
    y = hits[:, 1]
    z = hits[:, 2]

    # Bin edges and centers
    edges = np.linspace(-detector.plane_size/2, detector.plane_size/2, detector.num_bins+1)
    centers = (edges[:-1] + edges[1:]) / 2

    # Digitize y and z to bin indices
    y_idx = np.digitize(y, edges) - 1
    z_idx = np.digitize(z, edges) - 1

    # Mask for hits inside the detector area
    mask = (y_idx >= 0) & (y_idx < detector.num_bins) & (z_idx >= 0) & (z_idx < detector.num_bins)
    y_idx = y_idx[mask]
    z_idx = z_idx[mask]

    # Detector efficiency: randomly keep a fraction of hits
    N = len(y_idx)
    M = int(detector.detector_eff * N)
    keep = np.random.choice(N, M, replace=False)
    y_idx = y_idx[keep]
    z_idx = z_idx[keep]

    # Fill histogram
    bins = np.zeros((detector.num_bins, detector.num_bins), dtype=int)
    np.add.at(bins, (y_idx, z_idx), 1)

    # Return coordinates of each binned hit (center of bin)
    hit_coords = np.stack([np.full_like(y_idx, L), centers[y_idx], centers[z_idx]], axis=1)

    return bins, hit_coords

def generate_plane_single(args):
    """Helper function to generate a single plane."""
    v, detector, mass, mass_filter, Rot, index = args
    v = Rot[index].apply(v)
    return bin_plane_xaxis(v, detector, mass, mass_filter)

def generate_planes(xyz, detector, num_images, Rot=None, mass_filter=[0, 1000],mass=None, num_workers=1):
    """
    Generate planes with optional parallelization control.

    Parameters:
    - src: Source file path
    - ID pdb id of protein
    - plane_size: Physical size of the plane
    - L: Scaling factor
    - bin_density: Density of bins in the plane
    - num_images: Number of planes to generate
    - Rot: Rotation matrices
    - detector_eff: Detector efficiency
    - axis: Axis for binning
    - mass_filter: Range of mass to filter
    - num_workers: Number of parallel workers (None for default)
    
    Returns:
    - planes: Generated planes
    """



    planes = np.ndarray(shape=(num_images, detector.num_bins, detector.num_bins))
    hit_coords = []
    data_len = len(xyz)
    
    if Rot is None:
        Rot = R.identity(num_images)
    
    if mass is None:
        mass_index = np.arange(data_len)
    else:
        mass_index = np.where((mass > mass_filter[0]) & (mass < mass_filter[1]))[0]
        

    # Prepare arguments for parallel processing
    args_list = []
    full_repeats = num_images // data_len
    remaining = num_images % data_len

    for k in range(full_repeats):
        for l, v in enumerate(xyz):
            index = k * data_len + l
            if index >= num_images:
                break
            args_list.append((v, detector, mass, mass_filter, Rot, index))

    for l in range(remaining):
        index = full_repeats * data_len + l
        if index >= num_images:
            break
        args_list.append((xyz[l], detector, mass, mass_filter, Rot, index))

    # Use ProcessPoolExecutor with a controlled number of workers
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(generate_plane_single, args_list))

    # Assign results back to the planes array
    for i, result in enumerate(results):
        planes[i, :, :],x = result
        hit_coords.append(x)

    return planes,xyz[:,mass_index],hit_coords

def invert_H_to_r(H, L, side='+'):
    hy = H[:, 1]
    hz = H[:, 2]
    norm = np.sqrt(1 + (hy / L)**2 + (hz / L)**2)
    if side == '+':
        r = np.stack([
            1 / norm,
            hy / (L * norm),
            hz / (L * norm)
        ], axis=1)
    elif side == '-':
        r = np.stack([
            -1 / norm,
            hy / (L * norm),
            hz / (L * norm)
        ], axis=1)
    else:
        raise ValueError("side must be '+' or '-'")
    return r


def healpix_hit_mask(a, L, side='+', nside=32, nest=False):
    """
    Boolean mask over HEALPix pixels that *could* be hit by rays to the square(s) at x=±L.
    side: '+', '-', or 'both'
    """
    if side not in ('+','-','both'):
        raise ValueError("side must be '+', '-', or 'both'")
    npix = hp.nside2npix(nside)
    ipix = np.arange(npix)

    theta, phi = hp.pix2ang(nside, ipix, nest=nest)
    st = np.sin(theta)
    x = np.cos(phi) * st     # NOTE: Healpy's x from (theta,phi) this way
    y = np.sin(phi) * st
    z = np.cos(theta)

    alpha = a / (2.0 * L)

    with np.errstate(divide='ignore', invalid='ignore'):
        cond_y = np.abs(y / x) <= alpha
        cond_z = np.abs(z / x) <= alpha

    toward_plus  = x > 0
    toward_minus = x < 0

    mask_plus  = toward_plus  & cond_y & cond_z
    mask_minus = toward_minus & cond_y & cond_z

    if side == '+':
        return mask_plus
    elif side == '-':
        return mask_minus
    else:  # 'both'
        return mask_plus | mask_minus


def build_healpix(r_vectors, Detector, side='both', nside=32, nest=False):
    """
    Bin rays (unit vectors) into a HEALPix map and set pixels *outside* the chosen plane(s)
    field-of-view to hp.UNSEEN. Zeros *inside* the region remain zero.

    side: '+', '-', or 'both' (controls only the geometric mask)
    """
    if side not in ('+','-','both'):
        raise ValueError("side must be '+', '-', or 'both'")

    npix = hp.nside2npix(nside)

    # Bin rays -> counts
    theta = np.arccos(r_vectors[:, 2])
    phi   = np.arctan2(r_vectors[:, 1], r_vectors[:, 0])
    pix   = hp.ang2pix(nside, theta, phi, nest=nest)
    counts = np.bincount(pix, minlength=npix).astype(float)

    # Apply geometric mask
    mask = healpix_hit_mask(Detector.plane_size, Detector.distance, side=side, nside=nside, nest=nest)

    hpx_map = np.full(npix, hp.UNSEEN, dtype=float)
    hpx_map[mask] = counts[mask]
    return hpx_map, mask


def build_healpix_full(r_vectors, nside=32, nest=False):
    """
    Bin rays (vectors) into a full-sky HEALPix map (whole 4π solid angle).
    Returns (hpx_map, mask) where mask is True for all pixels.

    r_vectors : (N,3) array-like of ray vectors (need not be unit length).
    nside     : healpy nside
    nest      : healpy nest ordering flag
    """
    r = np.asarray(r_vectors)
    npix = hp.nside2npix(nside)

    # handle empty input
    if r.size == 0 or r.shape[0] == 0:
        return np.zeros(npix, dtype=float), np.ones(npix, dtype=bool)

    # normalize, skip zero-length vectors
    norms = np.linalg.norm(r, axis=1)
    valid = norms > 0
    if not np.all(valid):
        r = r[valid]
        norms = norms[valid]

    r_unit = r / norms[:, None]

    # spherical angles (healpy expects theta = colatitude)
    theta = np.arccos(np.clip(r_unit[:, 2], -1.0, 1.0))
    phi = np.mod(np.arctan2(r_unit[:, 1], r_unit[:, 0]), 2*np.pi)

    pix = hp.ang2pix(nside, theta, phi, nest=nest)
    counts = np.bincount(pix, minlength=npix).astype(float)

    hpx_map = counts
    mask = np.ones(npix, dtype=bool)

    return hpx_map, mask

def rotate_map(hmap, R_scipy: R, interp=False):
    """
    Rotate a HEALPix map by a SciPy Rotation (active rotation).
    Preserves hp.UNSEEN mask.

    f_rot(n) = f(R^{-1} n)  (pullback)
    """
    nside = hp.npix2nside(hmap.size)
    npix  = hp.nside2npix(nside)

    # Boolean mask
    mask = (hmap == UN)

    # Target directions (one per output pixel)
    ipix = np.arange(npix)
    vx, vy, vz = hp.pix2vec(nside, ipix)
    v = np.column_stack((vx, vy, vz))                    # (npix, 3)

    # Pullback: for each target dir n, sample source at R^{-1} n
    v_src = R_scipy.inv().apply(v)                       # (npix, 3)

    # Back to angles
    th_src = np.arccos(np.clip(v_src[:, 2], -1.0, 1.0))
    ph_src = np.mod(np.arctan2(v_src[:, 1], v_src[:, 0]), 2*np.pi)

    # Rotate the MASK first (nearest neighbor is exact for masks)
    ip_src_nn = hp.ang2pix(nside, th_src, ph_src)
    mask_rot = mask[ip_src_nn]

    # Rotate the VALUES
    if interp:
        vals = hp.get_interp_val(hmap, th_src, ph_src)
        # kill any interpolated garbage coming from UNSEEN neighbors
        vals[mask_rot] = UN
    else:
        vals = hmap[ip_src_nn]  # NN remap; exact mask preservation

    return vals


def average_maps(maps, weights=None, require_all=True):
    """
    maps: list/tuple of maps (npix) or stacks shaped (nmap, npix)
    weights: same length as maps, defaults to 1
    require_all: if True, pixel must be valid in all inputs; else at least one.
    """
    UN = hp.UNSEEN
    maps = np.asarray(maps)
    if maps.ndim == 1: maps = maps[None, :]
    nmap, npix = maps.shape

    if weights is None:
        weights = np.ones(nmap)
    weights = np.asarray(weights).reshape(nmap, 1)

    valid = (maps != UN)                      # (nmap, npix) boolean
    if require_all:
        keep = valid.all(axis=0)              # intersection mask
    else:
        keep = valid.any(axis=0)              # union mask

    filled = np.where(valid, maps, 0.0)       # zero-fill the holes
    w_eff  = weights * valid                  # zero out weights where invalid

    num = (w_eff * filled).sum(axis=0)
    den = w_eff.sum(axis=0)

    out = np.full(npix, UN, dtype=float)
    ok = keep & (den > 0)
    out[ok] = num[ok] / den[ok]
    return out






def _pool_init(ntomp: int):
    # set once per worker
    os.environ["OMP_NUM_THREADS"] = str(ntomp)

def _run_one_sim(args):
    sim, ionize, ntomp = args
    try:
        # optional: also enforce threads inside worker
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



class GromacsProteinSelect(Select):
    def accept_atom(self, atom):
        return atom.get_parent().id[0] == " "  # Keep only ATOMs, skip HETATM

def clean_pdb_file(input_path, output_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("cleaned", input_path)

    io = PDBIO()
    io.set_structure(structure)
    io.save(output_path, select=GromacsProteinSelect())

    # Remove TER, MASTER, END lines, only keep ATOMs
    with open(output_path, "r") as f:
        lines = [line for line in f if line.startswith("ATOM")]

    lines.append("END\n")

    with open(output_path, "w") as f:
        f.writelines(lines)


class simulation_handler: 
    def __init__(self,GMX_path,ROOT_PATH,name,systems,MDP,charge_file=None,ff="charmm36-mar2019-Fe-S"):
        self.GMX = GMX_path
        self.systems = [os.path.abspath(system) for system in systems]
        self.name = name
        self.path = os.path.join(ROOT_PATH,name)
        os.makedirs(self.path, exist_ok=True)
        self.path = os.path.abspath(self.path)
        self.MDP = MDP
        self.ff = ff
        self.charge_file = charge_file
    
        self.sim_paths = [os.path.join(self.path,os.path.basename(system).split(".")[0]) for system in self.systems]
        for sim_path in self.sim_paths:
            os.makedirs(sim_path, exist_ok=True)

        self.single_explosions = []

        for system,sim_path in zip(self.systems,self.sim_paths):
            if not os.path.exists(system):
                raise ValueError(f"System path '{system}' does not exist.")
            if system.endswith(".gro"):
                GRO = system
                PDB = None
            elif system.endswith(".pdb"):
                PDB = system
                GRO = None
            else:
                raise ValueError(f"System path '{system}' must be a .gro or .pdb file.")
            
            sim = simulation(sim_path,GMX_path,self.MDP,self.ff,GRO=GRO,PDB=PDB,charge_file=self.charge_file)
            self.single_explosions.append(sim)


    def run_all(
        self,
        processes: int = 1,
        ionize: bool = True,
        ntomp: int = 1,
        backend: str = "spawn",
        chunksize: int = 1,
        maxtasksperchild: int = 1,
        verbose: bool = True,
    ):
        tasks = [(sim, ionize, ntomp) for sim in self.single_explosions]
        results = []

        ctx = get_context(backend)
        with ctx.Pool(
            processes=processes,
            initializer=_pool_init,          # <-- named fn, picklable
            initargs=(ntomp,),
            maxtasksperchild=maxtasksperchild,
        ) as pool:
            it = pool.imap_unordered(_run_one_sim, tasks, chunksize=chunksize)
            for r in it:
                results.append(r)
                if verbose:
                    if r["status"] == "ok":
                        print(f"[{r['meta']['name']}] OK (rc={r['meta']['returncode']})")
                    else:
                        print(f"[{r.get('name','?')}] ERROR: {r['error']}")

        # set sim.result in parent (workers' mutations don't propagate)
        for sim in self.single_explosions:
            meta = next((x["meta"] for x in results
                        if x["status"] == "ok" and x["meta"]["simdir"] == sim.path_simulation), None)
            try:
                sim.result = result_handler(sim.path_simulation) if meta else None
            except Exception:
                sim.result = None

        self.last_run_results = results
        return results


def run_many_sims_parallel(sims,NUM_CORES): 

    def run_sim(sim):
        return sim.mdrun(ionize=True)

    with ThreadPoolExecutor(max_workers=NUM_CORES) as executor:
        for meta, sim in zip(executor.map(run_sim, sims), sims):
            sim.result = result_handler(meta["simdir"]) if meta["ok"] else None




class simulation:
    """
    This class handles the simulation data for a single run.
    """
    def __init__(self,path_simulation,GMX,MDP,ff,PDB=None,GRO=None,TOP=None,path_to_atomic_data=None,charge_file=None):
        """
        Initialize the simulation handler with the path to the simulation data.
        """
        if PDB is None and GRO is None:
            raise ValueError("Either PDB or GRO file must be provided.")


        os.makedirs(path_simulation,exist_ok=True)
        self.path_simulation = os.path.abspath(path_simulation)
        self.clean_dir()
        self.path_to_atomic_data = path_to_atomic_data
        self.name = os.path.basename(self.path_simulation)
        self.label = None  # label is a unique identifier for the simulation, created later in the "create_minimal_labels" function.
        self.GMX = GMX # path to the GROMACS executable
        self.GROMPP = os.path.join(self.GMX,"grompp")
        self.MDRUN = os.path.join(self.GMX,"mdrun")
        self.PDB2GMX = os.path.join(self.GMX,"pdb2gmx")
        self.EDITCONF = os.path.join(self.GMX,"editconf")
        self.ff = ff

        self.result = None  # This will be set later when the simulation is run


        # Convert between GRO and PDB if necessary
        if GRO is not None and os.path.exists(GRO):
            if not GRO.endswith(".gro"):
                raise ValueError("GRO file must have .gro extension.")
            
            PDB = self.GRO_to_PDB(GRO)

        if PDB is not None and os.path.exists(PDB):
            if not PDB.endswith(".pdb"):
                raise ValueError("PDB file must have .pdb extension.")

            GRO = self.PDB_to_GRO(PDB)


        # Copy all files to simulation folder
        MDP_BASE = os.path.basename(MDP)
        if PDB is not None:
            PDB_BASE = os.path.basename(PDB)
        if GRO is not None:
            GRO_BASE = os.path.basename(GRO)
        if TOP is not None:
            TOP_BASE = os.path.basename(TOP)
            
        if charge_file is not None:
            CHARGE_BASE = os.path.basename(charge_file)


        self.MDP = os.path.join(self.path_simulation,MDP_BASE)
        # Only copy if source and destination are not the same file
        if os.path.abspath(MDP) != os.path.abspath(self.MDP):
            shutil.copy(MDP,self.MDP)
        

        if PDB is not None:
            self.PDB = os.path.join(self.path_simulation,PDB_BASE)
            # Only copy if source and destination are not the same file
            if os.path.abspath(PDB) != os.path.abspath(self.PDB):
                shutil.copy(PDB,self.PDB)
        else:
            self.PDB = None
            
        if GRO is not None:
            self.GRO = os.path.join(self.path_simulation, GRO_BASE)
            # Only copy if source and destination are not the same file
            if os.path.abspath(GRO) != os.path.abspath(self.GRO):
                shutil.copy(GRO, self.GRO)
        else:
            self.GRO = None


        
        
        if TOP is not None:
            self.TOP = os.path.join(self.path_simulation,TOP_BASE)
            shutil.copy(TOP,self.TOP)
        else: 
            self.TOP = None
            
        if charge_file is not None:
            self.charge_file = os.path.join(self.path_simulation,CHARGE_BASE)
            if os.path.abspath(charge_file) != os.path.abspath(self.charge_file):
                shutil.copy(charge_file,self.charge_file)

    def pdb2gmx(self,PDB=False,GRO=False,clean=True):
        if not (PDB or GRO):
            raise ValueError("Either PDB or GRO must be True to run pdb2gmx.")

        if PDB:
            input_file = self.PDB
        if GRO:
            input_file = self.GRO

        if clean:
            clean_pdb_file(self.PDB,self.PDB)

        TOP = os.path.join(self.path_simulation,f"{self.name}.top")
        GRO_OUT = os.path.join(self.path_simulation,f"{self.name}.gro")
        cmd = [self.PDB2GMX,"-f",input_file,"-p",TOP,
               "-o",GRO_OUT,"-i",os.path.join(self.path_simulation,"posre.itp"),
               "-water","tip3p","-ff",self.ff,"-ignh"]
        
        try: 
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, errors="replace")
            self.TOP = TOP
            self.GRO = GRO_OUT
            return result
        except subprocess.CalledProcessError as e:
            print(f"Error in pdb2gmx: {e}")
            print(f"Error: {e.stderr}")
            return e



    def grompp(self):
        """
        This function prepares the simulation input files for GROMACS using grompp.

        Parameters:
        self.path_simulation (str): The path to the directory where the simulation files are stored.
        self.name (str): The name of the simulation.
        self.MDP (str): The path to the MDP file.
        self.GRO (str): The path to the GRO file.
        self.TOP (str): The path to the TOP file.
        self.GROMPP (str): The path to the grompp executable.

        Returns:
        None. If grompp execution is successful, it will create a TPR file in the simulation directory.
        If grompp execution fails, it will print an error message and the error output.
        """
        self.TPR = os.path.join(self.path_simulation,f"{self.name}.tpr")
        cmd = [self.GROMPP, '-f', self.MDP, '-c',self.GRO,
            '-p', self.TOP, '-o', self.TPR, 
            "-po",os.path.join(self.path_simulation,"mdout.mdp"),'-maxwarn', '5']

        try: 
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, errors="replace")
        except subprocess.CalledProcessError as e:
            print(f"Error in grompp: {e}")
            print(f"Error: {e.stderr}")
            return

    def mdrun(self, ionize=False, num_cores=1):
        simdir = self.path_simulation
        self.traj = os.path.join(simdir, f"{self.name}.trr")
        self.out  = os.path.join(simdir, "out.gro")
        self.edr  = os.path.join(simdir, f"{self.name}.edr")
        self.log  = os.path.join(simdir, f"{self.name}.log")

        cmd = [
            self.MDRUN,
            "-s", self.TPR,
            "-o", self.traj,
            "-c", self.out,
            "-e", self.edr,
            "-g", self.log,
            "-v",
            "-nt",
        ]
        if ionize:
            cmd.append("1")
            cmd.append("-ionize")
        else:
            cmd.append(str(num_cores))

        # Better: stream stdout/err to files to avoid bloating memory
        stdout_path = os.path.join(simdir, "mdrun.stdout")
        stderr_path = os.path.join(simdir, "mdrun.stderr")
        with open(stdout_path, "w") as so, open(stderr_path, "w") as se:
            res = subprocess.run(cmd, cwd=simdir, stdout=so, stderr=se, text=True, errors="replace")

        ok = (res.returncode == 0)
        # Keep the old side-effect for single-process runs
        if ok:
            try:
                self.result = result_handler(simdir)
            except Exception as e:
                print("Result handler failed:")
                traceback.print_exc()
                raise

        # Return only JSON/pickle-safe data to the parent
        return {
            "ok": ok,
            "returncode": res.returncode,
            "simdir": simdir,
            "name": self.name,
            "traj": self.traj,
            "out": self.out,
            "edr": self.edr,
            "log": self.log,
            "stdout": stdout_path,
            "stderr": stderr_path,
        }
        
    def set_top(self,TOP):
        """
        Set the topology file for the simulation.
        """
        if not os.path.exists(TOP):
            raise ValueError(f"Topology file '{TOP}' does not exist.")
        
        if not TOP.endswith(".top"):
            raise ValueError("Topology file must have .top extension.")
        
        self.TOP = os.path.join(self.path_simulation,os.path.basename(TOP))
        shutil.copy(TOP,self.TOP)
        
        
    def GRO_to_PDB(self, GRO):        
        """
        Convert the GRO file to PDB format.
        using edifconf
        """
        if GRO is None:
            raise ValueError("No GRO file found.")
        
        cmd = [self.EDITCONF, '-f', GRO, '-o', os.path.join(self.path_simulation, f"{self.name}.pdb")]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, errors="replace")
            pdb_file = os.path.join(self.path_simulation, f"{self.name}.pdb")
            return pdb_file
        except subprocess.CalledProcessError as e:
            print(f"Error in GRO_to_PDB: {e}")
            print(f"Error: {e.stderr}")
            return e
    
    
    def PDB_to_GRO(self,PDB):
        """
        Convert the PDB file to GRO format.
        using editconf
        """
        if PDB is None:
            raise ValueError("No PDB file found.")
        
        cmd = [self.EDITCONF, '-f', PDB, '-o', os.path.join(self.path_simulation, f"{self.name}.gro")]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, errors="replace")
            gro_file = os.path.join(self.path_simulation, f"{self.name}.gro")
            return gro_file
        except subprocess.CalledProcessError as e:
            print(f"Error in PDB_to_GRO: {e}")
            print(f"Error: {e.stderr}")
            return e

        


    def clean_dir(self,verbose=False):
        directory = os.path.abspath(self.path_simulation)

        # Safety net: refuse to delete dangerous directories
        if directory in ("/", os.path.expanduser("~")):
            raise RuntimeError(f"Refusing to clean dangerous directory: {directory}")
        
        # Allowed extensions to remove
        kill_exts = {".gro", ".pdb", ".log", ".trr", ".edr", ".tpr", ".top"}

        for entry in os.listdir(directory):
            path = os.path.join(directory, entry)
            _, ext = os.path.splitext(entry)

            if ext.lower() in kill_exts:
                try:
                    if os.path.isfile(path) or os.path.islink(path):
                        if verbose:
                            print(f"Removing file: {path}")
                        os.remove(path)
                    elif os.path.isdir(path):
                        # unlikely since we target file extensions
                        if verbose:
                            print(f"Removing directory: {path}")
                        shutil.rmtree(path)
                except Exception as e:
                    print(f"Could not remove {path}: {e}")

    def atomic_data_from_directory(self,path_to_directory,energy=None):
        if energy is None:
            energy = self.energy

        # Collect all atomic data files from a directory
        if not os.path.exists(path_to_directory):
            raise ValueError(f"Path '{path_to_directory}' does not exist.")
        if not os.path.isdir(path_to_directory):
            raise ValueError(f"Path '{path_to_directory}' is not a directory.") 

        # All files in the directory that ends with .z01
        models = os.listdir(path_to_directory)
        models = [os.path.join(path_to_directory,model) for model in models if model.endswith(".z01")]

        atomic_data_path = os.path.join(self.path_simulation,"Atomic_data")
        self.path_to_atomic_data = atomic_data_path
        os.makedirs(self.path_to_atomic_data, exist_ok=True)

        for model in models:
            element = os.path.basename(model).split(".")[0]
            e_file = os.path.join(self.path_to_atomic_data,f"energy_levels_{element}.txt")
            t_file = os.path.join(self.path_to_atomic_data,f"rate_transitions_{element}.txt")
            generate_atomic_data(model,e_file,t_file,energy)  # energy is in eV, can be changed later

        
        
    def set_parameters(self,
                       nsteps=None,
                       time_step=None,
                       pulse_peak=None,
                       num_photons=None,
                       sigma=None,
                       FWHM=None,
                       focus=None,
                       energy=None,
                       charge_transfer=None,
                       autostop=None,
                       autostop_limit=None,
                       logging=None,
                       ionize=None,
                       gen_vel=None,
                       gen_temp=None,
                       log_frequency=None,
                       set_charges=None,
                       rc=None,
                       ):
        
        if FWHM is not None:
            sigma = FWHM/(2*np.sqrt(2*np.log(2)))

        not_none_names = [
            k for k, v in locals().items()
            if v is not None and k not in {"FWHM", "self"}
        ]

        not_none_values = [
            v for k, v in locals().items()
            if v is not None and k not in {"FWHM", "self","not_none_names"}
        ]

        for _name, _value in zip(not_none_names, not_none_values):
            setattr(self, _name, _value)
        if FWHM is not None:
            self.FWHM = FWHM

        var_mdp = {"nsteps":"nsteps",
                "time_step":"dt",
                "pulse_peak":"userreal1",
                "num_photons":"userreal2",
                "sigma":"userreal3",
                "focus":"userreal4",
                "energy":"userreal5",
                "charge_transfer":"userint2",
                "autostop":"userint3",
                "autostop_limit":"userreal6",
                "logging":"userint5",
                "ionize":"userint1",
                "gen_vel":"gen_vel",
                "gen_temp":"gen_temp",
                "set_charges":"userint9",
                
                }

        def change_line(split_line,line,parameter,value):
            if parameter in split_line:
                return f"{parameter}                   = {value}; \n"
            else:
                return line

        with open(self.MDP,"r") as f:
            lines = f.readlines()

        with open(self.MDP,"w") as f:
            for line in lines: 
                # skip lines that are comments
                if line.strip().startswith(";") or line.strip() == "":
                    f.write(line)
                    continue
                
                split_line = line.split()
                
                for name,value in zip(not_none_names,not_none_values):
                    if name == "log_frequency":
                        # log_frequency is a special case, it correspond to multiple parameters in the MDP file
                        if "nstxout" in split_line:
                            line = change_line(split_line,line,"nstxout",value)
                        if "nstfout" in split_line:
                            line = change_line(split_line,line,"nstfout",value)
                        if "nstvout" in split_line:
                            line = change_line(split_line,line,"nstvout",value)
                        if "nstenergy" in split_line:
                            line = change_line(split_line,line,"nstenergy",value)
                        if "nstlog" in split_line:
                            line = change_line(split_line,line,"nstlog",value)
                        if "xtc_precision" in split_line:
                            line = change_line(split_line,line,"xtc_precision",value)
                    elif name == "rc":
                        # rc is a special case, it correspond to multiple parameters in the MDP file
                        if "rlist" in split_line:
                            line = change_line(split_line,line,"rlist",value)
                        if "rcoulomb" in split_line:
                            line = change_line(split_line,line,"rcoulomb",value)
                        if "rvdw" in split_line:
                            line = change_line(split_line,line,"rvdw",value)
                    
                    else:
                        par = var_mdp[name]
                        line = change_line(split_line,line,par,value)

                f.write(line)
        


        
        

        

        

        
def write_energy_levels_to_file(path_to_model, path_to_outfile):
    """
    This function writes the energy levels and corresponding configurations to a file.

    Parameters:
    path_to_model (str): The path to the LevelDB file containing the energy level data.
    path_to_outfile (str): The path to the output file where the energy levels and configurations will be written.

    Returns:
    None. The function writes the energy levels and configurations to the specified output file.
    """
    db = ld.LevelDB(path_to_model)
    with open(path_to_outfile, "w") as f:
        for level in db.levels:
            # Calculate binding energy
            bd = level.iso_energy - level.energy
            if bd < 0:
                bd = 1e14

            # Get config and pad with zeros if not already 3 long
            config = level.config.copy()
            while len(config) < 3:
                config.append(0)

            # Write to file
            for x in config:
                f.write(f"{x} ")
            f.write(f"{bd}\n")

def write_transitions_to_file(path_to_model,filename,energy):
    db = ld.LevelDB(path_to_model)
    transitions = collect_transitions(db,energy)

    #print(f"Writing {len(transitions)} transitions to {filename}")

    with open(filename,"w") as f:
        for transition in transitions:
            initital_state = transition[0]
            final_states = transition[1]
            rates = transition[2]
            types = transition[3]

            # Write the initial state
            for x in initital_state:
                f.write(f"{x} ")
            f.write(";")

            for k in range(len(final_states)):
                for x in final_states[k]:
                    f.write(f"{x} ")
                f.write(f"{rates[k]} ")
                f.write(f"{types[k]} ;")

            f.write("\n")
        f.writelines("0 0 0 ; 0 0 1  0.000 0; " + "\n")

def phot_ion_crossection(db,energy,iso1,i1,iso2,i2):
    a0 = db.get_phot_ion(iso1,i1,iso2,i2).a0
    b0 = db.get_phot_ion(iso1,i1,iso2,i2).b0
    c0 = db.get_phot_ion(iso1,i1,iso2,i2).c0
    d0 = db.get_phot_ion(iso1,i1,iso2,i2).d0
    threshold_energy = db.get_level(iso1,i1).iso_energy
    sigma = a0*(energy+b0)*c0*(energy-threshold_energy)/(energy+b0)*d0

    return sigma


def phis_crossection(db,energy,transition):
    a0 = transition.a0
    a1 = transition.a1
    a2 = transition.a2
    a3 = transition.a3
    de = transition.de
    n = transition.n
    emin = transition.emin
    emax = transition.emax
    # check if energy is within emin and emax
    if not (emin <= energy <= emax):
        return 0.0

    b = energy/de
    pblog = a0 + a1*np.log(b) + a2*np.log(b)**2 + a3*np.log(b)**3  
    pb = np.exp(pblog)
    sigma = 1e-18*n*pb*(13.606/de)
    return sigma

# Fluoresence
def phxs_crossection(db,iso1,i1,iso2,i2):
    c = 29979245800
    e = 1.60217663*1e-19 
    m = 9.1093837*1e-31


    f = db.get_phxs(iso1,i1,iso2,i2).f
    wavelength = db.get_phxs(iso1,i1,iso2,i2).wavelength
    width = db.get_phxs(iso1,i1,iso2,i2).width

    wavelength = wavelength * 1e-10
    v = c / wavelength

    gi = db.get_level(iso1, i1).g
    gf = db.get_level(iso2, i2).g

    sigma = (8*np.pi**2*e**2*v**2/(m*c**3))*(gi/gf)*f
    return sigma

def phxs2_crossection(db,iso1,i1,iso2,i2):
    c = 29979245800 # [cm / s]
    e = 1.60217663*1e-19 
    m = 9.1093837*1e-31
    f = db.get_phxs(iso1,i1,iso2,i2).f

    sigma = (np.pi * e**2 * f)/(np.sqrt(2*np.pi)*m*c)

    return sigma


def auger_crossection(db,iso1,i1,iso2,i2):
    sigma = db.get_augx(iso1,i1,iso2,i2).A

    return sigma

def collect_transitions(db,energy):
    phis_type = 2
    phxs_type = 1
    augxs_type = 0


    phis_transitions = []
    # Type 2: photoionization
    for transition in db.phis:
        iso1 = transition.iso1
        i1 = transition.i1
        iso2 = transition.iso2
        i2 = transition.i2

        config_i = db.get_level(iso1,i1).config.copy()
        config_f = db.get_level(iso2,i2).config.copy()

        while len(config_i) < 3:
            config_i.append(0)
        while len(config_f) < 3:
            config_f.append(0)

        sigma = phis_crossection(db, energy,transition)

        phis_transitions.append((config_i, config_f, sigma,phis_type))

    # Fluoresence type 1
    phxs_transitions = []
    for transition in db.phxs:
        
        iso1 = transition.iso1
        i1 = transition.i1
        iso2 = transition.iso2
        i2 = transition.i2

        # inverse process
        config_i = db.get_level(iso2,i2).config.copy()
        config_f = db.get_level(iso1,i1).config.copy()

        while len(config_i) < 3:
            config_i.append(0)
        while len(config_f) < 3:
            config_f.append(0)

        sigma = phxs_crossection(db, iso1, i1, iso2, i2)

        phxs_transitions.append((config_i, config_f, sigma,augxs_type))
    # Auger type 0
    augxs_transitions = []
    try:
        for transition in db.ai:
            iso1 = transition.iso1
            i1 = transition.i1
            iso2 = transition.iso2
            i2 = transition.i2

            # process
            config_i = db.get_level(iso1,i1).config.copy()
            config_f = db.get_level(iso2,i2).config.copy()

            while len(config_i) < 3:
                config_i.append(0)
            while len(config_f) < 3:
                config_f.append(0)

            sigma = auger_crossection(db, iso1, i1, iso2, i2)
            augxs_transitions.append((config_i,config_f,sigma,augxs_type))
    except Exception:
        pass


    # Photoexcitation
    photoex_transitions = []
    for transition in db.phxs:
        iso1 = transition.iso1
        i1 = transition.i1
        iso2 = transition.iso2
        i2 = transition.i2

        config_i = db.get_level(iso1,i1).config.copy()
        config_f = db.get_level(iso2,i2).config.copy()

        while len(config_i) < 3:
            config_i.append(0)
        while len(config_f) < 3:
            config_f.append(0)

        sigma = phxs2_crossection(db, iso1, i1, iso2, i2)

        photoex_transitions.append((config_i, config_f, sigma,phxs_type))

    def group_transitions(transitions):
        grouped = defaultdict(lambda: ([], [], []))
        for initial, final, sigma,type in transitions:
            grouped[tuple(initial)][0].append(final)
            grouped[tuple(initial)][1].append(sigma)
            grouped[tuple(initial)][2].append(type)
        return [(list(init), finals, sigmas,types) for init, (finals, sigmas,types) in grouped.items()]

    all_transitions = phis_transitions + phxs_transitions + augxs_transitions + photoex_transitions #+ augxs_inverse_transitions

    transitions = group_transitions(all_transitions)

    return transitions
    
def generate_atomic_data(path_to_model,path_to_energy_file,path_to_transition_file,energy):
    write_energy_levels_to_file(path_to_model,path_to_energy_file)
    write_transitions_to_file(path_to_model,path_to_transition_file,energy)
            





    








       





class result_handler:
    def __init__(self, path):

        # Check if path exists and is a directory
        if not os.path.exists(path) or not os.path.isdir(path):
            raise ValueError(f"Path '{path}' does not exist or is not a directory.")
        
        # Path and directory name
        self.path = path
        self.name = os.path.basename(self.path)
        self.label = None # label is a unique identifier for the result, created later in the "create_minimal_labels" function.

        # Save RAW MD parameters 
        try:
            with open(os.path.join(self.path,"mdout.mdp"), 'r') as f:
                self.mdp = f.readlines()
        except FileNotFoundError:
            self.mdp = None


        files = os.listdir(self.path)

        # save trr file path
        trr_files = [f for f in files if f.endswith('.trr')]
        self.trr = trr_files[0] if trr_files else None
        self.trr = os.path.join(self.path, self.trr) if self.trr else None

        # save FINAL STEP gro file path
        gro_files = [f for f in files if f.endswith('.gro')]
        self.gro = gro_files[0] if gro_files else None
        self.gro = os.path.join(self.path, self.gro) if self.gro else None

        # save pdb file path
        pdb_files = [f for f in files if f.endswith('.pdb')]
        self.pdb = pdb_files[0] if pdb_files else None
        self.pdb = os.path.join(self.path, self.pdb) if self.pdb else None

        # Save top file path
        top_files = [f for f in files if f.endswith('.top')]
        self.top = top_files[0] if top_files else None
        self.top = os.path.join(self.path, self.top) if self.top else None

        # Save tpr file path
        tpr_files = [f for f in files if f.endswith('.tpr')]
        self.TPR = tpr_files[0] if tpr_files else None
        self.TPR = os.path.join(self.path, self.TPR) if self.TPR else None
        
        # save edr file path
        edr_files = [f for f in files if f.endswith('.edr')]
        self.edr = edr_files[0] if edr_files else None
        self.edr = os.path.join(self.path, self.edr) if self.edr else None


        # Check if both files exist
        if not self.gro or not os.path.isfile(self.gro):
            raise FileNotFoundError(f"No valid .gro file found in {self.path}")
        if not self.trr or not os.path.isfile(self.trr):
            raise FileNotFoundError(f"No valid .trr file found in {self.path}")

        # Get MDA data
        self.masses, self.unit_displacements, self.final_velocities = self.get_MDA_data()

        self.total_mass = sum(self.masses)
        self.num_atoms = len(self.masses)

        # Get charges
        charge_file_path = os.path.join(self.path, "simulation_output","charges.txt")
        if os.path.exists(charge_file_path):
            charge_file = np.loadtxt(charge_file_path)  # Load the charge data
            self.final_charges = charge_file[:, 1]
            self.mean_charge = self.final_charges.mean()  # Calculate the mean charge

        mean_charge_vs_time_file = os.path.join(self.path,"simulation_output/mean_charge_vs_time.txt")
        if os.path.exists(mean_charge_vs_time_file):
            mean_charge_vs_time = np.loadtxt(mean_charge_vs_time_file)
            self.mean_charge_vs_time = mean_charge_vs_time[:, 1]

        # Pulse profile
        pulse_profile_file = os.path.join(self.path,"simulation_output/pulse_profile.txt")
        if os.path.exists(pulse_profile_file):
            pulse_profile = np.loadtxt(pulse_profile_file)
            self.pulse_profile = pulse_profile[:, 1]
    
        # Get FEL parameters
        self.fel_params = self.get_FEL_parameters()

        self.parameters = {
            "path": self.path,
            "name": self.name,
            "gro_file": self.gro,
            "trr_file": self.trr,
            "pdb_file": self.pdb,
            "edr_file": self.edr,
            "total_mass": self.total_mass,
            "num_atoms": self.num_atoms,
            "mean_charge": self.mean_charge if hasattr(self, 'mean_charge') else None,
            "pulse_peak": self.fel_params.get("pulse_peak", None),
            "intensity": self.fel_params.get("intensity", None),
            "FWHM_sigma": self.fel_params.get("FWHM_sigma", None),
            "FWHM": self.fel_params.get("FWHM", None),
            "focus": self.fel_params.get("focus", None),
            "energy": self.fel_params.get("energy", None)
        }

    def __repr__(self):
        units = [None,None,None,None,None,None,"Dalton","atoms","e/atom","ps","photons","ps","ps","nm","ev"]
        # return the self.parameters with line breaks for better readability
        return "\n".join([f"{key}: {value} [{unit}]" for (key, value), unit in zip(self.parameters.items(),units)])


    def get_MDA_data(self):
        try:
            U = md.Universe(self.gro,self.trr)
        except Exception:
            U = md.Universe(self.pdb,self.trr)
        ag = U.atoms.select_atoms("all")
        idx = ag.indices
        # Save mass of all atoms
        mass_data = np.array(ag.masses.tolist())

        # First frame
        U.trajectory[0]
        pos_i = ag.positions.copy() 

        # Last frame
        U.trajectory[-1] 
        vel_f = ag.velocities.copy()
        pos_f = ag.positions.copy() 
        # Normalize displacements
        pos_data = np.array([(x / np.linalg.norm(x)) if np.linalg.norm(x) != 0 else x for x in (pos_f - pos_i)])

        return mass_data, pos_data, vel_f
    
    def get_FEL_parameters(self):
        """
        Extract FEL parameters from the mdp file.
        """
        if not self.mdp:
            raise ValueError("MDP file not found.")
        
        fel_params = {}
        for line in self.mdp:
            if line.startswith('userreal1'):
                key, value = line.split('=')
                fel_params["pulse_peak"] = float(value.strip())
            elif line.startswith('userreal2'):
                key, value = line.split('=')
                fel_params["intensity"] = float(value.strip())
            elif line.startswith('userreal3'):
                key, value = line.split('=')
                fel_params["FWHM_sigma"] = float(value.strip())
                fel_params["FWHM"] = fel_params["FWHM_sigma"] *2*np.sqrt(2 * np.log(2))  # Convert sigma to FWHM
            elif line.startswith('userreal4'):
                key, value = line.split('=')
                fel_params["focus"] = float(value.strip())
            elif line.startswith('userreal5'):
                key, value = line.split('=')
                fel_params["energy"] = float(value.strip())

        self.pulse_peak = fel_params["pulse_peak"]
        self.intensity = fel_params["intensity"]
        self.FWHM_sigma = fel_params["FWHM_sigma"]
        self.FWHM = fel_params["FWHM"]
        self.focus = fel_params["focus"]
        self.energy = fel_params["energy"]

        return fel_params
    
    def extract_frames(self,outdir, num_frames, start=0, end=None, as_pdb=False):

        tpr = self.TPR
        trr = self.trr
        # Keep nm so GROMACS reads PDB correctly if requested
        U = md.Universe(tpr or None, trr, convert_units=True) if tpr else md.Universe(None, trr, convert_units=True)
        total = len(U.trajectory)
        if end is None:
            end = total
        idxs = np.unique(np.linspace(start, end-1, num=min(num_frames, end-start), dtype=int))
        os.makedirs(outdir, exist_ok=True)

        U.dimensions = [10.0, 10.0, 10.0, 90.0, 90.0, 90.0]
        ext = "pdb" if as_pdb else "gro"
        for i, fr in enumerate(idxs, 1):
            U.trajectory[fr]
            path = os.path.join(outdir, f"frame_{i:06d}.{ext}")
            with md.Writer(path, multiframe=False) as W:
                W.write(U.atoms)
                
    def get_kinetic_potential_from_edr(self):
        if not self.edr:
            raise ValueError("EDR file not found.")
        
        U = md.Universe(self.gro,self.trr)  # Initialize the universe to read the EDR file
        edr_info = EDR.EDRReader(self.edr)  # Initialize the reader to read the header and get the number of steps
        U.trajectory.add_auxiliary("energy", edr_info, "Potential")
        epot = []
        ekin= []
        
        for ts in U.trajectory:
            ek = ts.aux.energy["Kinetic En."]
            ep = ts.aux.energy["Potential"]
    
            epot.append(ep)
            ekin.append(ek)
        
        
        return np.array(ekin), np.array(epot)
    
def collect_results_from_directory(path):
    """
    Collect results from a directory containing multiple result folders.
    """
    if not os.path.exists(path):
        raise ValueError(f"Path '{path}' does not exist.")
    if not os.path.isdir(path):
        raise ValueError(f"Path '{path}' is not a directory.")
    results = []
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path):
            try:
                result = result_handler(folder_path)
                results.append(result)
            except (FileNotFoundError, ValueError) as e:
                print(f"Skipping folder '{folder}': {e}")
    return results

# filters out results based on  parameters
# Given an upper and lower bound for intensity, FWHM_sigma,pulse_peak, focus, energy, num_atoms and total mass if no bounds are given, it will return all results
def filter_results(results, intensity_bounds=None, FWHM_sigma_bounds=None, pulse_peak_bounds=None, focus_bounds=None, energy_bounds=None, num_atoms_bounds=None, total_mass_bounds=None):
    """
    Filter results based on given bounds for various parameters.
    """
    filtered_results = []
    
    for result in results:
        if intensity_bounds and not (intensity_bounds[0] <= result.fel_params["intensity"] <= intensity_bounds[1]):
            continue
        if FWHM_sigma_bounds and not (FWHM_sigma_bounds[0] <= result.fel_params["FWHM_sigma"] <= FWHM_sigma_bounds[1]):
            continue
        if pulse_peak_bounds and not (pulse_peak_bounds[0] <= result.fel_params["pulse_peak"] <= pulse_peak_bounds[1]):
            continue
        if focus_bounds and not (focus_bounds[0] <= result.fel_params["focus"] <= focus_bounds[1]):
            continue
        if energy_bounds and not (energy_bounds[0] <= result.fel_params["energy"] <= energy_bounds[1]):
            continue
        if num_atoms_bounds and not (num_atoms_bounds[0] <= result.num_atoms <= num_atoms_bounds[1]):
            continue
        if total_mass_bounds and not (total_mass_bounds[0] <= result.total_mass <= total_mass_bounds[1]):
            continue
        
        filtered_results.append(result)
    
    return filtered_results


def create_minimal_labels(results):
    """
    Create a minimal set of labels for the results.
    Labels include only the FEL parameters that vary across the results.
    All values are formatted with 2 decimals.
    Duplicate labels are suffixed with _00000, _00001, etc.
    """
    param_names = ["pulse_peak", "intensity", "FWHM", "focus", "energy"]
    varying_params = []

    # Find which parameters vary
    for param in param_names:
        values = set(result.fel_params[param] for result in results)
        if len(values) > 1:
            varying_params.append(param)

    # Build identifiers
    identifiers = []
    for result in results:
        parts = []
        for key in varying_params:
            val = result.fel_params[key]
            val_str = f"{val:.2f}"
            parts.append(f"{key}={val_str}")
        identifier = "_".join(parts)
        identifiers.append(identifier)

    # Handle duplicates: add _00000, _00001, etc.
    counts = {}
    labels = []
    for identifier in identifiers:
        if identifier not in counts:
            counts[identifier] = 0
        else:
            counts[identifier] += 1
        label = f"{identifier}_{counts[identifier]:05d}"
        labels.append(label)

    # Set labels to results
    for result, label in zip(results, labels):
        result.label = label    

    return labels



def collect_unit_displacements(results):
    """
    Collect unit displacements from all results.
    """
    unit_displacements = []
    for result in results:
        unit_displacements.append(result.unit_displacements)
    return np.array(unit_displacements)

def collect_velocities(results):
    """
    Collect final velocities from all results.
    """
    velocities = []
    for result in results:
        velocities.append(result.final_velocities)
    return np.array(velocities)

def collect_mean_charges(results):
    """
    Collect mean charges from all results.
    """
    mean_charges = []
    for result in results:
        mean_charges.append(result.mean_charge)
    return np.array(mean_charges)

def collect_energies(results):
    """
    Collect energies from all results.
    """
    energies = []
    for result in results:
        energies.append(result.fel_params.get("energy", None))
    return np.array(energies)

def collect_FWHM(results):
    """
    Collect FEL FWHM values from all results.
    """
    fel_FWHM = []
    for result in results:
        fel_FWHM.append(result.fel_params.get("FWHM", None))
    return np.array(fel_FWHM)

def collect_pulse_peaks(results):
    """
    Collect FEL pulse peak values from all results.
    """
    pulse_peaks = []
    for result in results:
        pulse_peaks.append(result.fel_params.get("pulse_peak", None))
    return np.array(pulse_peaks)

def collect_intensities(results):
    """
    Collect FEL intensity values from all results.
    """
    intensities = []
    for result in results:
        intensities.append(result.fel_params.get("intensity", None))
    return np.array(intensities)

def collect_focuses(results):
    """
    Collect FEL focus values from all results.
    """
    focuses = []
    for result in results:
        focuses.append(result.fel_params.get("focus", None))
    return np.array(focuses)

def collect_num_atoms(results):
    """
    Collect number of atoms from all results.
    """
    num_atoms = []
    for result in results:
        num_atoms.append(result.num_atoms)
    return np.array(num_atoms)

def collect_total_masses(results):
    """
    Collect total masses from all results.
    """
    total_masses = []
    for result in results:
        total_masses.append(result.total_mass)
    return np.array(total_masses)

def collect_charges(results):
    """
    Collect mean charges from all results.
    """
    charges = []
    for result in results:
        charges.append(result.charges)
    return np.array(charges)








        



        