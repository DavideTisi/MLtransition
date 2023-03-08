import numpy as np
import equistore
from equistore.operations import mean_over_samples, slice
from rascaline import SphericalExpansion, SoapPowerSpectrum
import chemiscope
import ase.io as aseio
from scipy import optimize

import copy as copy
from ase.build.tools import sort as asesort

from rascal.models import Kernel, train_gap_model, compute_KNM
from rascal.representations import SphericalInvariants
from rascal.representations import SphericalExpansion as SEXP
from rascal.utils import from_dict, to_dict, CURFilter, dump_obj, load_obj
from rascal.neighbourlist.structure_manager import (
        mask_center_atoms_by_species, mask_center_atoms_by_id)

# utils from equistore-examples
#from utils.rotations import rotation_matrix, xyz_to_spherical, spherical_to_xyz, wigner_d_real
#from utils.clebsh_gordan import ClebschGordanReal



import sys
alpha = float(sys.argv[1])


def mk_descriptor(frame, calculator=calculator):
    descriptor = calculator.compute([frame])
    descriptor = descriptor.keys_to_properties(['species_neighbor',]).keys_to_samples('species_center')
    return mean_over_samples(descriptor, samples_names = ['center'])


def mk_frame(pos, abcdef, template=beta):
    frame = template.copy()
    frame.positions = pos.reshape(-1,3)
    # keep the cell orthorhombic
    frame.cell = abcdef[:3]
    # frame.cell = [[abcdef[0],0,0],[abcdef[3],abcdef[1],0],[abcdef[4],abcdef[5],abcdef[2]]]
    return frame


# use "aligned" structures
align_beta, rot_gamma = aseio.read("../data/beta-gamma_aligned_sorted.extxyz",":")#aseio.read("/tmp/beta-gamma.extxyz",":")
beta, gamma = align_beta, rot_gamma

hypers = {
    "cutoff": 7,
    "max_radial": 9,
    "max_angular": 5,
    "atomic_gaussian_width": 1.0,
    "radial_basis": {"Gto": {}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.5}},
    "center_atom_weight": 1.0
}

print('letto tutto')
print('alpha',alpha)

calculator = SphericalExpansion(**hypers)

descriptor = calculator.compute([beta, gamma, align_beta, rot_gamma])


descriptor = descriptor.keys_to_properties(['species_neighbor']).keys_to_samples('species_center')

mean_descriptor = mean_over_samples(descriptor, samples_names = ['center'])


desc_beta = slice(mean_descriptor, samples=equistore.Labels(["structure"], np.array([[0]], np.int32)))
desc_gamma = slice(mean_descriptor, samples=equistore.Labels(["structure"], np.array([[1]], np.int32)))





PBEsolpot=load_obj('/home/tisi/LiPS/LiPS_michelemagic/model-PBEsol-menocoesiveenergy-rcut5_sigma0.3-4000sparseenv-1500traindata.json')
energiesPBEsol = []
forcesPBEsol = []

soapPBEsol = PBEsolpot.get_representation_calculator()

n_eval = 0
intermediates = []
def struc_diff(pos_cell, target_descriptors,alpha=1,ref_energy=-7.863632805093914):
    global n_eval
    pos = pos_cell[:-6]
    cell = pos_cell[-6:]  
    frame = mk_frame(pos, cell)
    desc = mk_descriptor(frame)
    frame.wrap(eps=1e-10)
    m = soapPBEsol.transform(frame)
    energy=0
    energy = PBEsolpot.predict(m)[0]
    diff = 0
    for i in range(len(desc.keys)):
        diff += np.sum((desc.block(i).values - target_descriptors.block(i).values)**2)
    diff += alpha*(energy-ref_energy)**2
    if n_eval%100 ==0:
        print(n_eval,cell, diff,energy)
        if n_eval%100 == 0:
            frame.info["diff"] = diff
            frame.info["energy"] = energy
            frame.positions -= frame.positions[12]
            frame.wrap(eps=1e-8)
            intermediates.append(frame)
    n_eval += 1 
    return diff



gamma_diff = struc_diff(np.concatenate([gamma.positions.flatten(), gamma.cell.diagonal(), [0,0,0]]), desc_beta,alpha=1)

x0 = np.concatenate([gamma.positions.flatten(), gamma.cell.diagonal(), [0,0,0]])


for i in range(12):
    find_struc = optimize.minimize(struc_diff, args=(desc_beta,alpha,-7.863632805093914), x0=x0, method="Nelder-Mead", options={"maxfev":6000, "initial_simplex":x0+0.1*np.random.normal(size=(len(x0)+1,len(x0)))})
    x0 = np.concatenate([intermediates[-1].positions.flatten(), intermediates[-1].cell.diagonal(), [0,0,0]])
    aseio.write(f'./traj_minimizzation-script_aligned_sorted_long_a{alpha}.extxyz', [gamma]+intermediates+[beta])

aseio.write(f'./traj_minimizzation-script_aligned_sorted_long_a{alpha}.extxyz', [gamma]+intermediates+[beta])
