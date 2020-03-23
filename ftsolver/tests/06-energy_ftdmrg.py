'''
    Test for FTDMRG.
    Huanchen Zhai, Mar 22, 2020
'''
import numpy as np
import sys
import os
import pickle
sys.path[:0] = ["..", "../../../pyblock", "../../../pyblock/build"]
import ftdmrg
from pyscf import gto

BOHR = 0.52917721092  # Angstroms

# for middle to large system, please use real scratch
pscratch = '.'
bond_dim = 100

if not os.path.isdir(pscratch):
    os.mkdir(pscratch)

def prepare(mol, scratch, symm, m):
    n_sites = mol.nao
    nelec = n_sites if symm is 'RHF' else (n_sites // 2, (n_sites + 1) // 2)
    ft = ftdmrg.FTDMRG(pg='d2h', n_sites=N, n_elec=nelec, scratch=scratch % symm, su2=symm is 'RHF', verbose=1)
    ft.write_fcidump(mol)
    ft.generate_initial_mps(m)
    pickle.dump(ft, open(scratch % symm + "/ftdmrg.tmp", "wb"))

def test_energy(scratch, symm, mu0, beta, step, m):
    ft = pickle.load(open(scratch % symm + "/ftdmrg.tmp", "rb"))
    nelec = mol.nelectron
    if symm is 'UHF':
        nelec = nelec // 2, nelec - nelec // 2
    mu = ft.optimize_mu(beta, step, mu0, m)
    ener = ft.energy(beta, step, mu, m)
    ne = ft.get_particle_number()
    print('%s MU = %20.16f NA = %20.16f NB = %20.16f E = %20.16f' % (symm, mu,
        *((ne / 2, ne / 2) if symm is 'RHF' else ne), ener))

def test_1pdm(scratch, symm):
    ft = pickle.load(open(scratch % symm + "/ftdmrg.tmp", "rb"))
    dm = ft.get_one_pdm(recover_orb_order=True)
    pickle.dump(dm, open(scratch % symm + "/dm.tmp", "wb"))

if __name__ == '__main__':
    N = 4
    symm = 'UHF'
    # prepare initial MPS
    for r in np.arange(1.4, 4.0, 0.4):
        print('PREPARING R = %.1f' % r)
        R = r * BOHR
        mol = gto.M(atom = [['H', (i * R, 0, 0)] for i in range(N)],
            basis='sto6g', verbose=0, symmetry='D2h')
        scratch = pscratch + '/FTDMRG-H%d-%.1f-%%s' % (N, r)
        prepare(mol=mol, scratch=scratch, symm=symm, m=bond_dim)
    # test
    beta = 1
    for r in np.arange(1.4, 4.0, 0.4):
        print('TESTING R = %.1f ' % r)
        scratch = pscratch + '/FTDMRG-H%d-%.1f-%%s' % (N, r)
        test_energy(scratch=scratch, symm=symm, mu0=0, beta=beta, step=0.25, m=bond_dim)
    # test 1pdm
    for r in np.arange(1.4, 4.0, 0.4):
        print('TESTING 1PDM R = %.1f ' % r)
        scratch = pscratch + '/FTDMRG-H%d-%.1f-%%s' % (N, r)
        test_1pdm(scratch=scratch, symm=symm)
