'''
test the calculation of electron number and the gradients.
dependency: pyscf
Chong Sun, Feb. 26, 2020

Revised:
    Eigenvalues cached.
    Huanchen Zhai, Mar 22, 2020

'''
import sys
sys.path.append("..")
import fted_fast as ffted
from pyscf import gto

mol = gto.M(
    atom = [['H', (0, 0, 0)],
            ['H', (3, 0, 0)],
            ['H', (0, 3, 0)],
            ['H', (3, 3, 0)]],
    basis = 'sto6g',
    verbose = 0,
)

def test_symm(symm, mu0, beta):
    ref = ffted.gc_ensemble_eigs(*ffted.run_scf(mol, symm), symm)
    ne, dne = ffted.elec_number(ref, mu0, beta, symm=symm)
    if symm is 'RHF':
        print('RHF F  = %20.16f G  = %20.16f' % (ne / 2, dne / 2))
    else:
        print('UHF FA = %20.16f GA = %20.16f' % (ne[0], dne[0]))
        print('UHF FB = %20.16f GB = %20.16f' % (ne[1], dne[1]))

if __name__ == '__main__':
    test_symm(symm='RHF', mu0=0, beta=1)
    test_symm(symm='UHF', mu0=0, beta=1)
