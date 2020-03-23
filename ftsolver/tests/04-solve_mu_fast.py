'''
test the calculation of electron number and the gradients.
dependency: pyscf
Chong Sun, Feb. 26, 2020

Revised:
    Eigenvalues cached.
    Huanchen Zhai, Mar 22, 2020
'''
import numpy as np
import sys
sys.path.append("..")
import fted_fast as ffted
from pyscf import gto

mol = gto.M(
    atom = [['H', (0, 0, 0)],
            ['H', (3, 0, 0)]],
    basis = 'sto6g',
    verbose = 0,
)

def test_symm(symm, mu0, beta):
    ref = ffted.gc_ensemble_eigs(*ffted.run_scf(mol, symm), symm)
    nelec = mol.nelectron
    if symm is 'UHF':
        nelec = nelec // 2, nelec - nelec // 2
    mu = ffted.solve_mu(ref, mu0, beta, nelec, symm=symm)
    ne, _ = ffted.elec_number(ref, mu, beta, symm=symm)
    print('%s MU = %20.16f NA = %20.16f NB = %20.16f' % (symm, mu,
        *((ne / 2, ne / 2) if symm is 'RHF' else ne)))

if __name__ == '__main__':
    test_symm(symm='RHF', mu0=0, beta=1)
    test_symm(symm='UHF', mu0=0, beta=1)
