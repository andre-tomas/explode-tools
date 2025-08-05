import os
import numpy as np
import MDAnalysis as md
# This file is part of ExplodeTools.





class result_handler:
    def __init__(self, path):
        # Path and directory name
        self.path = path
        self.name = os.path.basename(self.path)

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


        # Get MDA data
        self.masses, self.unit_displacements, self.final_velocities = self.get_MDA_data()

        self.total_mass = sum(self.masses)
        self.num_atoms = len(self.masses)




    def get_MDA_data(self):
        U = md.Universe(self.gro,self.trr)
        ag = U.atoms.select_atoms("all")
        idx = ag.indices
        # Save mass of all atoms
        mass_data = ag.masses.tolist()

        # First frame
        universe.trajectory[0]
        pos_i = ag.positions.copy() 

        # Last frame
        universe.trajectory[-1] 
        vel_f = ag.velocities.copy()
        pos_f = ag.positions.copy() 
        # Normalize displacements
        pos_data = [(x / np.linalg.norm(x)) if np.linalg.norm(x) != 0 else x for x in (pos_f - pos_i)]  

        return mass_data, pos_data, vel_f





        



        