import numpy as np
import argparse
import json
import equistore
from equistore.operations import mean_over_samples, slice
from rascaline import SphericalExpansion, SoapPowerSpectrum
#import chemiscope
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

import sys

def mk_diff(A,B):
    blocks = []
    for key,block in A:
        values = block.values - B[key].values
        blocks.append(equistore.TensorBlock(values=values,
                                            samples=block.samples,
                                           properties=block.properties,
                                           components=block.components))
    return equistore.TensorMap(A.keys,blocks)

def mk_dot(A,B):
    A1 = A.components_to_properties('spherical_harmonics_m').keys_to_properties('spherical_harmonics_l')
    B1 = B.components_to_properties('spherical_harmonics_m').keys_to_properties('spherical_harmonics_l')
    return np.dot(A1[0].values.reshape(1,-1),B1[0].values.reshape(-1)).reshape(-1)


n_eval = 0
def main(paramfile):
    with open(paramfile) as fd:
        parameters = json.load(fd)
    
    hypers = parameters['hypers']
    alpha = parameters['alpha']
    inputfile = parameters['inputfile']
    outputdir = parameters['outputdir']
    model = parameters['models']





    # use "aligned" structures
    align_beta, rot_gamma = aseio.read("./data/beta-gamma_aligned_sorted.extxyz",":")
    beta, gamma = align_beta, rot_gamma
    
    print('letto tutto')
    print('alpha',alpha)
    
    calculator = SphericalExpansion(**hypers)
    #calculator = SoapPowerSpectrum(**hypers)
    
    def mk_descriptor(frame, calculator=calculator):
        descriptor = calculator.compute([frame])
        descriptor = descriptor.keys_to_properties(['species_neighbor',]).keys_to_samples('species_center')
        descriptor = slice(descriptor, samples=equistore.Labels(["species_center"], np.array([[15],[16]], dtype=np.int32)),
            properties=equistore.Labels(["species_neighbor"], np.array([[15],[16]], dtype=np.int32)))
        return mean_over_samples(descriptor, samples_names = ['center'])
    
    
    def mk_frame(pos, abcdef, template=beta):
        frame = template.copy()
        frame.positions = pos.reshape(-1,3)
        # keep the cell orthorhombic
        frame.cell = abcdef[:3]
        # frame.cell = [[abcdef[0],0,0],[abcdef[3],abcdef[1],0],[abcdef[4],abcdef[5],abcdef[2]]]
        return frame
    
    descriptor = calculator.compute([beta, gamma, align_beta, rot_gamma])
    
    
    descriptor = descriptor.keys_to_properties(['species_neighbor']).keys_to_samples('species_center')
    descriptor = slice(descriptor, samples=equistore.Labels(["species_center"], np.array([[15],[16]], dtype=np.int32)),
            properties=equistore.Labels(["species_neighbor"], np.array([[15],[16]], dtype=np.int32)))
    
    mean_descriptor = mean_over_samples(descriptor, samples_names = ['center'])
    
    
    desc_beta = slice(mean_descriptor, samples=equistore.Labels(["structure"], np.array([[0]], np.int32)))
    desc_gamma = slice(mean_descriptor, samples=equistore.Labels(["structure"], np.array([[1]], np.int32)))
    print('desc_beta',desc_beta)
    print('desc_gamma',desc_gamma)
    
    diff_betagamma = mk_diff(desc_beta,desc_gamma)
    print('diff_betagamma',diff_betagamma)
    print(diff_betagamma[0])
    betagamma2 = mk_dot(diff_betagamma,diff_betagamma)
    
    
    
    #PBEsolpot=load_obj('/scratch/tisi/LiPS/Michele_magic/model-PBEsol-menocoesiveenergy-rcut5_sigma0.3-4000sparseenv-1500traindata.json')
    PBEsolpot=load_obj(model)
    energiesPBEsol = []
    forcesPBEsol = []
    
    soapPBEsol = PBEsolpot.get_representation_calculator()
    
    intermediates = []
    def collective_variable(pos_cell, target_descriptors,init_m_target,alpha_energy=1,ref_energy=-7.863632805093914):
        global n_eval
        pos = pos_cell[:-6]
        cell = pos_cell[-6:]  
        frame = mk_frame(pos, cell)
        desc = mk_descriptor(frame)
        eta_m_target = mk_diff(desc,target_descriptors)
        s = mk_dot(eta_m_target,init_m_target)
        frame.wrap(eps=1e-10)
        m = soapPBEsol.transform(frame)
        energy=0
        energy = PBEsolpot.predict(m)[0]
        diff = s**2
        energy_contrib = alpha_energy*(energy-ref_energy)
        diff += energy_contrib
        if n_eval%100 ==0:
            print(n_eval,cell, diff,energy)
            if n_eval%100 == 0:
                frame.info["totdiff"] = diff
                frame.info["diff"] = diff-energy_contrib
                frame.info["energy"] = energy
                frame.positions -= frame.positions[12]
                frame.wrap(eps=1e-8)
                intermediates.append(frame)
        n_eval += 1 
        return diff
    
    
    
    print(1./betagamma2)
    x0 = np.concatenate([gamma.positions.flatten(), gamma.cell.diagonal(), [0,0,0]])
    start = 0
    #print(np.linspace(0,1,int(1/delta)+1)[1:])
    #for factor in np.linspace(0,1,int(1/delta)+1)[1:]:
    #    start=len(intermediates)-start
    for i in range(8):
        find_struc = optimize.minimize(collective_variable, args=(desc_beta,equistore.operations.multiply(diff_betagamma,1./betagamma2),alpha,-7.8636328052103295), x0=x0, method="Nelder-Mead", options={"maxfev":12000, "initial_simplex":x0+0.01*np.random.normal(size=(len(x0)+1,len(x0)))})
        x0 = np.concatenate([intermediates[-1].positions.flatten(), intermediates[-1].cell.diagonal(), [0,0,0]])
        aseio.write(outputdir+f'/traj_minimizzation-script_aligned_sorted_long_a{alpha}_target{factor}.extxyz', intermediates[start:])
    
    aseio.write(outputdir+f'/traj_minimizzation-script_aligned_sorted_long_a{alpha}_total.extxyz', [gamma]+intermediates+[beta])




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"""
        This tool make the pulling of the transition from a phase to the other

        Usage:
             python {sys.argv[0]}  parameters.json 
        """
    )

    parser.add_argument(
        "parameters",
        type=str,
        help="file containing the parameters in JSON format",
    )
    args = parser.parse_args()

    main(args.parameters)
