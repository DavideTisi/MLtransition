import numpy as np
import argparse
import equistore
from equistore.operations import mean_over_samples, slice
from rascaline import SphericalExpansion, SoapPowerSpectrum
import ase.io as aseio
from scipy import optimize

import copy as copy
from ase.build.tools import sort as asesort

from rascal.models import Kernel, train_gap_model, compute_KNM
from rascal.representations import SphericalInvariants
from rascal.representations import SphericalExpansion as SEXP
from rascal.utils import from_dict, to_dict, CURFilter, dump_obj, load_obj
import sys
import json




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
    #align_beta, rot_gamma = aseio.read("../data/beta-gamma_aligned_sorted.extxyz",":")#aseio.read("/tmp/beta-gamma.extxyz",":")
    align_beta, rot_gamma = aseio.read(inputfile , ":")
    beta, gamma = align_beta, rot_gamma
    print('letto tutto')
    print('alpha',alpha)
    
    calculator = SphericalExpansion(**hypers)
    
    descriptor = calculator.compute([beta, gamma, align_beta, rot_gamma])
    
    
    descriptor = descriptor.keys_to_properties(['species_neighbor']).keys_to_samples('species_center')
    
    mean_descriptor = mean_over_samples(descriptor, samples_names = ['center'])
    
    
    desc_beta = slice(mean_descriptor, samples=equistore.Labels(["structure"], np.array([[0]], np.int32)))
    desc_gamma = slice(mean_descriptor, samples=equistore.Labels(["structure"], np.array([[1]], np.int32)))
    
    
    
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
    
    
    #PBEsolpot=load_obj('./model-PBEsol-menocoesiveenergy-rcut5_sigma0.3-4000sparseenv-1500traindata.json')
    PBEsolpot=load_obj(model)
    energiesPBEsol = []
    forcesPBEsol = []
    
    soapPBEsol = PBEsolpot.get_representation_calculator()
    
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
        aseio.write(outputdir+f'/traj_minimizzation-script_aligned_sorted_long_a{alpha}.extxyz', [gamma]+intermediates+[beta])
    
    aseio.write(outputdir+f'/traj_minimizzation-script_aligned_sorted_long_a{alpha}.extxyz', [gamma]+intermediates+[beta])


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

