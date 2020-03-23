'''
    Test for energy (FTED).
    Huanchen Zhai, Mar 22, 2020
'''
import numpy as np
import sys
import os
import pickle
sys.path.append("..")
import fted_fast as ffted
import fted
from pyscf import gto

BOHR = 0.52917721092  # Angstroms

def prepare(mol, ref_file, symm):
    ref = ffted.gc_ensemble_eigs(*ffted.run_scf(mol, symm), symm)
    pickle.dump(ref, open(ref_file % symm, "wb"))

def test_energy(ref_file, symm, mu0, beta):
    ref = pickle.load(open(ref_file % symm, "rb"))
    nelec = mol.nelectron
    if symm is 'UHF':
        nelec = nelec // 2, nelec - nelec // 2
    mu = ffted.solve_mu(ref, mu0, beta, nelec, symm=symm)
    ne, _ = ffted.elec_number(ref, mu, beta, symm=symm)
    ener = ffted.energy(ref, mu, beta, symm=symm)
    print('%s MU = %20.16f NA = %20.16f NB = %20.16f E = %20.16f' % (symm, mu,
        *((ne / 2, ne / 2) if symm is 'RHF' else ne), ener))
    return nelec, mu, ener

if __name__ == '__main__':
    N = 4
    symm = 'RHF'
    # prepare data
    for r in np.arange(1.4, 4.0, 0.4):
        print('R = %.1f' % r)
        R = r * BOHR
        mol = gto.M(atom = [['H', (i * R, 0, 0)] for i in range(N)],
            basis = 'sto6g', verbose = 0, symmetry = 'D2h')
        ref_file = 'H%d-%.1f-%%s.tmp' % (N, r)
        prepare(mol=mol, ref_file=ref_file, symm=symm)
    # test energy
    beta = 1
    for r in np.arange(1.4, 4.0, 0.4):
        print('R = %.1f ' % r, end='')
        ref_file = 'H%d-%.1f-%%s.tmp' % (N, r)
        test_energy(ref_file=ref_file, symm=symm, mu0=0, beta=beta)
    # test 1pdm
    for r in np.arange(1.4, 4.0, 0.4):
        print('R = %.1f' % r)
        R = r * BOHR
        mol = gto.M(atom = [['H', (i * R, 0, 0)] for i in range(N)],
            basis = 'sto6g', verbose = 0, symmetry = 'D2h')
        ref_file = 'H%d-%.1f-%%s.tmp' % (N, r)
        nelec, mu, _ = test_energy(ref_file=ref_file, symm=symm, mu0=0, beta=beta)
        h1e, g2e, norb = ffted.run_scf(mol, symm=symm)
        pdm1, _, _ = fted.rdm12s_fted(h1e, g2e, norb, nelec, beta, mu=mu, symm=symm)
        dm_file = 'H%d-%.1f-%%s-dm.tmp' % (N, r)
        pickle.dump(pdm1, open(dm_file % symm, "wb"))
