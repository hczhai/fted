
#
# FT-DMRG using pyscf and pyblock
#
# Author: Huanchen Zhai, Mar 22, 2020
#

import numpy as np
import scipy
import pickle
import time
import os
import copy

from pyscf import scf, ao2mo, symm
from pyblock.qchem import *
from pyblock.qchem.ancilla import *
from pyblock.qchem.operator import OpNames
from pyblock.qchem.fcidump import write_fcidump
from pyblock.qchem.thermal import FreeEnergy
from pyblock.algorithm import ExpoApply, Compress, Expect, DMRG

class FTDMRGError(Exception):
    pass

class FTDMRG:
    """
    Finite-temperature DMRG for molecules.
    """

    def __init__(self, pg, n_sites, n_elec, scratch='.', su2=True, omp_threads=4, memory=20000, verbose=2):
        """
        Args:
            pg: str
                Point group symmetry of the molecule.
            n_sites: int
                Number of physical sites.
            n_elec: int or (int, int)
                Number of electrons.
            scratch: str
                Directory for writing temporary files.
            su2: bool
                RHF (True) or UHF (False) orbitals.
            omp_threads: int
                Number of openmp threads used in FTDMRG.
            memory: int
                Upper limit of stack memory (in MB).
            verbose: 0 or 1 or 2
                Verbosity.
        """
        
        self.n_sites = n_sites
        if su2:
            assert isinstance(n_elec, int)
            self.n_elec = n_elec
        else:
            self.n_alpha, self.n_beta = n_elec

        if not os.path.isdir(scratch):
            os.mkdir(scratch)

        self.verbose = verbose

        if self.verbose < 2:
            from pyblock.algorithm import time_evolution, expectation, compress, dmrg
            time_evolution.pprint = lambda *args, **kwargs: None
            expectation.pprint = lambda *args, **kwargs: None
            compress.pprint = lambda *args, **kwargs: None
            dmrg.pprint = lambda *args, **kwargs: None

        self.scratch = scratch
        self.fcidump = self.scratch + '/' + 'FCIDUMP.%d' % self.n_sites
        self.init_mps = self.scratch + '/' + 'mps.initial.%d' % self.n_sites
        self.final_mps = self.scratch + '/' + 'mps.final.%d' % self.n_sites
        self.final_gs_mps = self.scratch + '/' + 'mps.final.gs.%d' % self.n_sites
        self.ridx = None

        self.opts = dict(
            fcidump=self.fcidump,
            pg=pg.lower(),
            su2=su2,
            output_level=-1,
            memory=memory,
            omp_threads=omp_threads,
            mkl_threads=1,
            nelec=self.n_sites * 2
        )

        self.simplifier = Simplifier(AllRules(su2=su2))

    
    def __getstate__(self):
        if self.opts['su2']:
            return (self.opts, self.n_sites, self.n_elec, self.scratch, self.verbose, self.ridx)
        else:
            return (self.opts, self.n_sites, self.n_alpha, self.n_beta, self.scratch, self.verbose, self.ridx)
    
    def __setstate__(self, state):
        if state[0]['su2']:
            self.opts, self.n_sites, self.n_elec, self.scratch, self.verbose, self.ridx = state
        else:
            self.opts, self.n_sites, self.n_alpha, self.n_beta, self.scratch, self.verbose, self.ridx = state
        self.simplifier = Simplifier(AllRules(su2=state[0]['su2']))
        self.fcidump = self.scratch + '/' + 'FCIDUMP.%d' % self.n_sites
        self.init_mps = self.scratch + '/' + 'mps.initial.%d' % self.n_sites
        self.final_mps = self.scratch + '/' + 'mps.final.%d' % self.n_sites
        self.final_gs_mps = self.scratch + '/' + 'mps.final.gs.%d' % self.n_sites

        if self.verbose < 2:
            from pyblock.algorithm import time_evolution, expectation, compress, dmrg
            time_evolution.pprint = lambda *args, **kwargs: None
            expectation.pprint = lambda *args, **kwargs: None
            compress.pprint = lambda *args, **kwargs: None
            dmrg.pprint = lambda *args, **kwargs: None

    def scan(self, beta, beta_step, mu, bond_dims):
        """
        Perform time evolution step-by-step for fixed mu and beta.

        Args:
            beta: float
                Inverse temperature.
            beta_step: float
                Step length of beta.
            mu: float
                Chemical potential.
            bond_dims: list(int) or int
                Bond dimension.
        """

        if self.verbose >= 2:
            print('>>> START scan <<<')
        t = time.perf_counter()

        self.opts["page"] = DMRGDataPage(save_dir=self.scratch, n_frames=4 if self.opts["su2"] else 7)
        
        with BlockHamiltonian.get(**self.opts) as hamil:
            
            fe_hamil = FreeEnergy(hamil)

            mpo_info = MPOInfo(hamil)
            empo_info = MPOInfo(hamil)
            if self.opts["su2"]:
                nmpo_info = LocalMPOInfo(hamil, OpNames.N)
                nnmpo_info = SquareMPOInfo(hamil, OpNames.N, OpNames.NN)
            else:
                na_mpo_info = LocalMPOInfo(hamil, OpNames.N, site_index=(0, ))
                nb_mpo_info = LocalMPOInfo(hamil, OpNames.N, site_index=(1, ))
                nna_mpo_info = SquareMPOInfo(hamil, OpNames.N, OpNames.NN, site_index=(0, ))
                nnb_mpo_info = SquareMPOInfo(hamil, OpNames.N, OpNames.NN, site_index=(1, ))
                nab_mpo_info = ProdMPOInfo(hamil, OpNames.N, OpNames.N, OpNames.NUD, site_index_a=(0, ), site_index_b=(1, ))
            
            mps, mps_info, forward = pickle.load(open(self.init_mps, 'rb'))
            
            tctr  = DMRGContractor(mps_info,   mpo_info, self.simplifier)
            ectr  = DMRGContractor(mps_info,  empo_info, self.simplifier)
            if self.opts["su2"]:
                nctr  = DMRGContractor(mps_info,  nmpo_info, self.simplifier)
                nnctr = DMRGContractor(mps_info, nnmpo_info, self.simplifier)
            else:
                nactr  = DMRGContractor(mps_info,  na_mpo_info, self.simplifier)
                nbctr  = DMRGContractor(mps_info,  nb_mpo_info, self.simplifier)
                nnactr = DMRGContractor(mps_info, nna_mpo_info, self.simplifier)
                nnbctr = DMRGContractor(mps_info, nnb_mpo_info, self.simplifier)
                nabctr = DMRGContractor(mps_info, nab_mpo_info, self.simplifier)
            
            tctr.page.activate({'_BASE'})
            fe_hamil.set_free_energy(mu)
            mpo = MPO(hamil)
            
            ectr.page.activate({'_BASE'})
            fe_hamil.set_energy()
            empo = MPO(hamil)
            
            if self.opts["su2"]:
                nctr.page.activate({'_BASE'})
                nmpo = LocalMPO(hamil, OpNames.N)
                nnctr.page.activate({'_BASE'})
                nnmpo = SquareMPO(hamil, OpNames.N, OpNames.NN)
            else:
                nactr.page.activate({'_BASE'})
                na_mpo  = LocalMPO(hamil, OpNames.N, site_index=(0, ))
                nbctr.page.activate({'_BASE'})
                nb_mpo  = LocalMPO(hamil, OpNames.N, site_index=(1, ))
                nnactr.page.activate({'_BASE'})
                nna_mpo = SquareMPO(hamil, OpNames.N, OpNames.NN, site_index=(0, ))
                nnbctr.page.activate({'_BASE'})
                nnb_mpo = SquareMPO(hamil, OpNames.N, OpNames.NN, site_index=(1, ))
                nabctr.page.activate({'_BASE'})
                nab_mpo = ProdMPO(hamil, OpNames.N, OpNames.N, OpNames.NUD, site_index_a=(0, ), site_index_b=(1, ))

            # Initial state
            normsq  = 1.0
            fener = Expect( mpo, mps, mps, mps.form, None, contractor=tctr).solve(bond_dim=bond_dims) / normsq
            ener  = Expect(empo, mps, mps, mps.form, None, contractor=ectr).solve(bond_dim=bond_dims) / normsq
            if self.verbose >= 1:
                print('Beta = %15.8f FEnergy = %25.15f Energy = %25.15f Norm^2 = %25.15f Error = %25.15f' % (0.0, fener, ener, normsq, 0))
            if self.opts["su2"]:
                partn   = Expect( nmpo, mps, mps, mps.form, None, contractor= nctr).solve(bond_dim=bond_dims) / normsq
                partnsq = Expect(nnmpo, mps, mps, mps.form, None, contractor=nnctr).solve(bond_dim=bond_dims) / normsq
                dpartn  = 0.0 * (partnsq - partn * partn)
                if self.verbose >= 1:
                    print('  ParticleN = %25.15f N^2 = %25.15f DN/DMu = %25.15f' % (partn, partnsq, dpartn))
            else:
                partna  = Expect( na_mpo, mps, mps, mps.form, None, contractor= nactr).solve(bond_dim=bond_dims) / normsq
                partnb  = Expect( nb_mpo, mps, mps, mps.form, None, contractor= nbctr).solve(bond_dim=bond_dims) / normsq
                partnna = Expect(nna_mpo, mps, mps, mps.form, None, contractor=nnactr).solve(bond_dim=bond_dims) / normsq
                partnnb = Expect(nnb_mpo, mps, mps, mps.form, None, contractor=nnbctr).solve(bond_dim=bond_dims) / normsq
                partnab = Expect(nab_mpo, mps, mps, mps.form, None, contractor=nabctr).solve(bond_dim=bond_dims) / normsq
                dpartna = 0.0 * (partnna + partnab - partna * partna - partna * partnb)
                dpartnb = 0.0 * (partnnb + partnab - partnb * partnb - partna * partnb)
                if self.verbose >= 1:
                    print(('  ParticleNA = %25.15f ParticleNB = %25.15f NA^2 = %25.15f NB^2 = %25.15f '
                        + 'NA*NB = %25.15f DNA/DMu = %25.15f DNB/DMu = %25.15f') % (partna, partnb, partnna, partnnb, partnab, dpartna, dpartnb))

            # Time evolution
            te = ExpoApply(mpo, mps, bond_dims=bond_dims, beta=beta_step / 2, contractor=tctr, canonical_form=mps.form)
            n_steps = int(round(beta / beta_step) + 0.1)
            assert abs(beta - n_steps * beta_step) < 1E-8
            current_beta = 0.0

            for it in range(n_steps):

                te.solve(n_sweeps=2, forward=forward, current_beta=current_beta, iprint=False)
                current_beta += beta_step / 2
                
                mps0 = te.mps
                normsq = te.normsqs[-1]
                error = te.errors[-1]
                fener = te.energies[-1]
                
                ener = Expect( empo, mps0, mps0, mps0.form, None, contractor= ectr).solve(bond_dim=bond_dims) / normsq
                if self.verbose >= 1:
                    print('Beta = %15.8f FEnergy = %25.15f Energy = %25.15f Norm^2 = %25.15f Error = %25.15f'
                        % (current_beta * 2, fener, ener, normsq, error))
                if self.opts["su2"]:
                    partn   = Expect( nmpo, mps0, mps0, mps0.form, None, contractor= nctr).solve(bond_dim=bond_dims) / normsq
                    partnsq = Expect(nnmpo, mps0, mps0, mps0.form, None, contractor=nnctr).solve(bond_dim=bond_dims) / normsq
                    dpartn  = 2 * current_beta * (partnsq - partn * partn)
                    if self.verbose >= 1:
                        print('  ParticleN = %25.15f N^2 = %25.15f DN/DMu = %25.15f' % (partn, partnsq, dpartn))
                else:
                    partna  = Expect( na_mpo, mps0, mps0, mps0.form, None, contractor= nactr).solve(bond_dim=bond_dims) / normsq
                    partnb  = Expect( nb_mpo, mps0, mps0, mps0.form, None, contractor= nbctr).solve(bond_dim=bond_dims) / normsq
                    partnna = Expect(nna_mpo, mps0, mps0, mps0.form, None, contractor=nnactr).solve(bond_dim=bond_dims) / normsq
                    partnnb = Expect(nnb_mpo, mps0, mps0, mps0.form, None, contractor=nnbctr).solve(bond_dim=bond_dims) / normsq
                    partnab = Expect(nab_mpo, mps0, mps0, mps0.form, None, contractor=nabctr).solve(bond_dim=bond_dims) / normsq
                    dpartna = 2 * current_beta * (partnna + partnab - partna * partna - partna * partnb)
                    dpartnb = 2 * current_beta * (partnnb + partnab - partnb * partnb - partna * partnb)
                    if self.verbose >= 1:
                        print(('  ParticleNA = %25.15f ParticleNB = %25.15f NA^2 = %25.15f NB^2 = %25.15f '
                            + 'NA*NB = %25.15f DNA/DMu = %25.15f DNB/DMu = %25.15f') % (partna, partnb, partnna, partnnb, partnab, dpartna, dpartnb))

        pickle.dump((mps0, mps_info, te.forward, normsq, bond_dims), open(self.final_mps, "wb" ))

        if self.verbose >= 2:
            print('>>> COMPLETE scan | Time = %.2f <<<' % (time.perf_counter() - t))

    def get_one_pdm(self, recover_orb_order=True):
        """
        Return one particle density matrix with shape (nspin=2, norb, norb) for final MPS.
        
        This only works when a final MPS is already in scratch directory. Use `self.energy` to generate final MPS.
        """
        
        if self.verbose >= 2:
            print('>>> START one-pdm <<<')
        t = time.perf_counter()
        
        self.opts["page"] = DMRGDataPage(save_dir=self.scratch, n_frames=1)
        
        with BlockHamiltonian.get(**self.opts) as hamil:
            
            pmpo_info = PDM1MPOInfo(hamil)
            
            mps0, mps_info, forward, normsq, bond_dims = pickle.load(open(self.final_mps, 'rb'))
            
            pctr = DMRGContractor(mps_info, pmpo_info, Simplifier(PDM1Rules(su2=self.opts["su2"])))
            pctr.page.activate({'_BASE'})
            pmpo = PDM1MPO(hamil)
            
            ex = Expect(pmpo, mps0, mps0, mps0.form, None, contractor=pctr)
            ex.solve(forward=forward, bond_dim=bond_dims)
            
            if self.opts["su2"]:
                dm = ex.get_1pdm_spatial(normsq=normsq)
            else:
                dm = ex.get_1pdm(normsq=normsq)
        
        if self.verbose >= 2:
            print('>>> COMPLETE one-pdm | Time = %.2f <<<' % (time.perf_counter() - t))

        if recover_orb_order:
            dm[:, :] = dm[self.ridx, :][:, self.ridx]

        if self.opts["su2"]:
            return np.concatenate([dm[None, :, :], dm[None, :, :]], axis=0) / 2
        else:
            return np.concatenate([dm[None, :, :, 0, 0], dm[None, :, :, 1, 1]], axis=0)

    def get_particle_number(self):
        """
        Return expectation value of particle number (total (for RHF) or (alpha, beta) (for UHF)) for final MPS.

        This only works when a final MPS is already in scratch directory. Use `self.energy` to generate final MPS.
        """
        
        if self.verbose >= 2:
            print('>>> START particle number <<<')
        t = time.perf_counter()
        
        self.opts["page"] = DMRGDataPage(save_dir=self.scratch, n_frames=1 if self.opts["su2"] else 2)
        
        with BlockHamiltonian.get(**self.opts) as hamil:
            
            if self.opts["su2"]:
                nmpo_info = LocalMPOInfo(hamil, OpNames.N)
            else:
                na_mpo_info = LocalMPOInfo(hamil, OpNames.N, site_index=(0, ))
                nb_mpo_info = LocalMPOInfo(hamil, OpNames.N, site_index=(1, ))
            
            mps0, mps_info, _, normsq, bond_dims = pickle.load(open(self.final_mps, 'rb'))

            if self.opts["su2"]:
                nctr = DMRGContractor(mps_info, nmpo_info, self.simplifier)
                nctr.page.activate({'_BASE'})
                nmpo = LocalMPO(hamil, OpNames.N)
            else:
                nactr = DMRGContractor(mps_info, na_mpo_info, self.simplifier)
                nbctr = DMRGContractor(mps_info, nb_mpo_info, self.simplifier)
                nactr.page.activate({'_BASE'})
                na_mpo  = LocalMPO(hamil, OpNames.N, site_index=(0, ))
                nbctr.page.activate({'_BASE'})
                nb_mpo  = LocalMPO(hamil, OpNames.N, site_index=(1, ))
            
            if self.opts["su2"]:
                partn = Expect(nmpo, mps0, mps0, mps0.form, None, contractor=nctr).solve(bond_dim=bond_dims) / normsq
            else:
                partna = Expect(na_mpo, mps0, mps0, mps0.form, None, contractor= nactr).solve(bond_dim=bond_dims) / normsq
                partnb = Expect(nb_mpo, mps0, mps0, mps0.form, None, contractor= nbctr).solve(bond_dim=bond_dims) / normsq

        if self.verbose >= 2:
            print('>>> COMPLETE particle number | Time = %.2f <<<' % (time.perf_counter() - t))
        
        if self.opts["su2"]:
            return partn
        else:
            return partna, partnb

    def ground_state_energy(self, bond_dims, noise):
        """
        Perform ground-state DMRG calculation.
        """
        
        from pyblock.qchem import MPS as GSMPS, MPO as GSMPO, MPOInfo as GSMPOInfo, LineCoupling as GSLineCoupling
        
        if self.verbose >= 2:
            print('>>> START dmrg <<<')
        t = time.perf_counter()
        
        self.opts["page"] = DMRGDataPage(save_dir=self.scratch, n_frames=1)
        opts = self.opts.copy()
        opts["nelec"] = self.n_elec if opts["su2"] else self.n_alpha + self.n_beta
        
        with BlockHamiltonian.get(**opts) as hamil:

            mpo_info  = GSMPOInfo(hamil)
            
            lcp = GSLineCoupling(hamil.n_sites, hamil.site_basis, hamil.empty, hamil.target)
            lcp.set_bond_dimension(bond_dims[0] if isinstance(bond_dims, list) else bond_dims)
            mps = GSMPS(lcp, center=0, dot=2)
            mps.canonicalize(random=True)
            
            mps_info = MPSInfo(lcp)
            ctr = DMRGContractor(mps_info,  mpo_info, self.simplifier, davidson_tol=1E-9)
            
            ctr.page.activate({'_BASE'})
            mpo = GSMPO(hamil)
            
            dmrg = DMRG(mpo, mps, bond_dims=bond_dims, noise=noise, contractor=ctr)
            ener = dmrg.solve(100, 1E-9)
            mps0 = dmrg.mps
            normsq = 1.0
        
        pickle.dump((mps0, mps_info, dmrg.forward, normsq, bond_dims), open(self.final_gs_mps, "wb" ))

        if self.verbose >= 2:
            print('>>> COMPLETE dmrg | Time = %.2f <<<' % (time.perf_counter() - t))
        
        return ener
        
    def energy(self, beta, beta_step, mu, bond_dims):
        """
        Perform time evolution and evaluate energy at last step.
        Note that constant energy term is not included in the output.

        Args:
            beta: float
                Inverse temperature.
            beta_step: float
                Step length of beta.
            mu: float
                Chemical potential.
            bond_dims: list(int) or int
                Bond dimension.
        """
        
        if self.verbose >= 2:
            print('>>> START energy <<<')
        t = time.perf_counter()
        
        self.opts["page"] = DMRGDataPage(save_dir=self.scratch, n_frames=2)
        
        with BlockHamiltonian.get(**self.opts) as hamil:
            
            fe_hamil = FreeEnergy(hamil)

            mpo_info  = MPOInfo(hamil)
            empo_info = MPOInfo(hamil)
            
            mps, mps_info, forward = pickle.load(open(self.init_mps, 'rb'))
            
            ectr = DMRGContractor(mps_info, empo_info, self.simplifier)
            tctr = DMRGContractor(mps_info,  mpo_info, self.simplifier)
            
            tctr.page.activate({'_BASE'})
            fe_hamil.set_free_energy(mu)
            mpo = MPO(hamil)
            
            ectr.page.activate({'_BASE'})
            fe_hamil.set_energy()
            empo = MPO(hamil)
                
            te = ExpoApply(mpo, mps, bond_dims=bond_dims, beta=beta_step / 2, contractor=tctr, canonical_form=mps.form)
                
            n_steps = int(round(beta / beta_step) + 0.1)
            assert abs(beta - n_steps * beta_step) < 1E-8

            te.solve(n_sweeps=2 * n_steps, forward=forward, current_beta=0, iprint=True)

            mps0 = te.mps
            normsq = te.normsqs[-1]
            ener = Expect(empo, mps0, mps0, mps0.form, None, contractor=ectr).solve(bond_dim=bond_dims) / normsq
        
        pickle.dump((mps0, mps_info, te.forward, normsq, bond_dims), open(self.final_mps, "wb" ))

        if self.verbose >= 2:
            print('>>> COMPLETE energy | Time = %.2f <<<' % (time.perf_counter() - t))
        
        return ener
    
    def optimize_mu(self, beta, beta_step, mu0, bond_dims, tol=1E-6, maxiter=10):
        """
        Find mu for expected number of electrons.

        Args:
            beta: float
                Inverse temperature.
            beta_step: float
                Step length of beta.
            mu0: float
                Initial guess for chemical potential.
            bond_dims: list(int) or int
                Bond dimension.
        """
        
        if self.verbose >= 2:
            print('>>> START optimize mu <<<')
        t = time.perf_counter()
        
        self.opts["page"] = DMRGDataPage(save_dir=self.scratch, n_frames=3 if self.opts["su2"] else 6)
        
        with BlockHamiltonian.get(**self.opts) as hamil:
            
            fe_hamil = FreeEnergy(hamil)

            if self.opts["su2"]:
                mpo_info = MPOInfo(hamil)
                nmpo_info = LocalMPOInfo(hamil, OpNames.N)
                nnmpo_info = SquareMPOInfo(hamil, OpNames.N, OpNames.NN)
            else:
                mpo_info = MPOInfo(hamil)
                na_mpo_info = LocalMPOInfo(hamil, OpNames.N, site_index=(0, ))
                nb_mpo_info = LocalMPOInfo(hamil, OpNames.N, site_index=(1, ))
                nna_mpo_info = SquareMPOInfo(hamil, OpNames.N, OpNames.NN, site_index=(0, ))
                nnb_mpo_info = SquareMPOInfo(hamil, OpNames.N, OpNames.NN, site_index=(1, ))
                nab_mpo_info = ProdMPOInfo(hamil, OpNames.N, OpNames.N, OpNames.NUD, site_index_a=(0, ), site_index_b=(1, ))
            
            _, mps_info, _ = pickle.load(open(self.init_mps, 'rb'))
            
            if self.opts["su2"]:
                nnctr = DMRGContractor(mps_info, nnmpo_info, self.simplifier)
                nctr  = DMRGContractor(mps_info,  nmpo_info, self.simplifier)
                tctr  = DMRGContractor(mps_info,   mpo_info, self.simplifier)
            else:
                tctr   = DMRGContractor(mps_info,     mpo_info, self.simplifier)
                nactr  = DMRGContractor(mps_info,  na_mpo_info, self.simplifier)
                nbctr  = DMRGContractor(mps_info,  nb_mpo_info, self.simplifier)
                nnactr = DMRGContractor(mps_info, nna_mpo_info, self.simplifier)
                nnbctr = DMRGContractor(mps_info, nnb_mpo_info, self.simplifier)
                nabctr = DMRGContractor(mps_info, nab_mpo_info, self.simplifier)
        
            mu_cache = {}

            def solve(mu):
                
                mps, mps_info, forward = pickle.load(open(self.init_mps, 'rb'))
                
                if self.opts["su2"]:
                    nnctr.mps_info = mps_info
                    nctr.mps_info = mps_info
                    tctr.mps_info = mps_info
                    nnctr.page.release()
                    nnctr.page.initialize()
                    nctr.page.release()
                    nctr.page.initialize()
                    tctr.page.release()
                    tctr.page.initialize()
                else:
                    for ctr in [tctr, nactr, nbctr, nnactr, nnbctr, nabctr]:
                        ctr.mps_info = mps_info
                        ctr.page.release()
                        ctr.page.initialize()
                
                tctr.page.activate({'_BASE'})
                fe_hamil.set_free_energy(mu)
                mpo = MPO(hamil)
                
                if self.opts["su2"]:
                    nctr.page.activate({'_BASE'})
                    nmpo = LocalMPO(hamil, OpNames.N)
                    nnctr.page.activate({'_BASE'})
                    nnmpo = SquareMPO(hamil, OpNames.N, OpNames.NN)
                else:
                    nactr.page.activate({'_BASE'})
                    na_mpo  = LocalMPO(hamil, OpNames.N, site_index=(0, ))
                    nbctr.page.activate({'_BASE'})
                    nb_mpo  = LocalMPO(hamil, OpNames.N, site_index=(1, ))
                    nnactr.page.activate({'_BASE'})
                    nna_mpo = SquareMPO(hamil, OpNames.N, OpNames.NN, site_index=(0, ))
                    nnbctr.page.activate({'_BASE'})
                    nnb_mpo = SquareMPO(hamil, OpNames.N, OpNames.NN, site_index=(1, ))
                    nabctr.page.activate({'_BASE'})
                    nab_mpo = ProdMPO(hamil, OpNames.N, OpNames.N, OpNames.NUD, site_index_a=(0, ), site_index_b=(1, ))
                
                te = ExpoApply(mpo, mps, bond_dims=bond_dims, beta=beta_step / 2, contractor=tctr, canonical_form=mps.form)
                
                n_steps = int(round(beta / beta_step) + 0.1)
                assert abs(beta - n_steps * beta_step) < 1E-8
                
                te.solve(n_sweeps=2 * n_steps, forward=forward, current_beta=0, iprint=True)
                
                mps0 = te.mps
                normsq = te.normsqs[-1]
                error = te.errors[-1]
                fener = te.energies[-1]
                if self.opts["su2"]:
                    partn   = Expect( nmpo, mps0, mps0, mps0.form, None, contractor= nctr).solve(bond_dim=bond_dims) / normsq
                    partnsq = Expect(nnmpo, mps0, mps0, mps0.form, None, contractor=nnctr).solve(bond_dim=bond_dims) / normsq
                    dpartn = 2 * beta * (partnsq - partn * partn)
                    diff = partn - self.n_elec
                    if self.verbose >= 1:
                        print('! MU = %20.16f N = %20.16f DN = %20.16f' % (mu, partn, dpartn))
                    return diff ** 2, 2 * diff * dpartn
                else:
                    partna  = Expect( na_mpo, mps0, mps0, mps0.form, None, contractor= nactr).solve(bond_dim=bond_dims) / normsq
                    partnb  = Expect( nb_mpo, mps0, mps0, mps0.form, None, contractor= nbctr).solve(bond_dim=bond_dims) / normsq
                    partnna = Expect(nna_mpo, mps0, mps0, mps0.form, None, contractor=nnactr).solve(bond_dim=bond_dims) / normsq
                    partnnb = Expect(nnb_mpo, mps0, mps0, mps0.form, None, contractor=nnbctr).solve(bond_dim=bond_dims) / normsq
                    partnab = Expect(nab_mpo, mps0, mps0, mps0.form, None, contractor=nabctr).solve(bond_dim=bond_dims) / normsq
                    dpartna = 2 * beta * (partnna + partnab - partna * partna - partna * partnb)
                    dpartnb = 2 * beta * (partnnb + partnab - partnb * partnb - partna * partnb)
                    diff = partna - self.n_alpha, partnb - self.n_beta
                    if self.verbose >= 1:
                        print('! MU = %20.16f NA = %20.16f NB = %20.16f DNA = %20.16f DNB = %20.16f' % (mu, partna, partnb, dpartna, dpartnb))
                    return diff[0] ** 2 + diff[1] ** 2, 2 * diff[0] * dpartna + 2 * diff[1] * dpartnb
            
            def f(mu):
                mu = mu[0]
                if mu not in mu_cache:
                    mu_cache[mu] = solve(mu)
                return mu_cache[mu][0]
            
            def g(mu):
                mu = mu[0]
                if mu not in mu_cache:
                    mu_cache[mu] = solve(mu)
                return mu_cache[mu][1]

            opt_mu = scipy.optimize.minimize(f, mu0, method="CG", jac=g,
                                             options={'disp': False, 'gtol': tol, 'maxiter': maxiter})
            if not opt_mu.success:
                print('optimization not success!!')

        if self.verbose >= 2:
            print('>>> COMPLETE optimize mu | Time = %.2f <<<' % (time.perf_counter() - t))

        return opt_mu.x[0]

    def generate_initial_mps(self, bond_dims):
        """Generate initial MPS at thermal limit, and decompress it into larger bond dimension."""
        
        if self.verbose >= 2:
            print('>>> START generate initial mps <<<')
        t = time.perf_counter()
        
        self.opts["page"] = DMRGDataPage(save_dir=self.scratch, n_frames=1)
        
        with BlockHamiltonian.get(**self.opts) as hamil:
    
            assert hamil.n_electrons == hamil.n_sites * 2

            # Line Coupling
            lcp_thermal = LineCoupling(hamil.n_sites, hamil.site_basis, hamil.empty, hamil.target)
            lcp_thermal.set_thermal_limit()
            lcp = LineCoupling(hamil.n_sites, hamil.site_basis, hamil.empty, hamil.target)
            lcp.set_bond_dimension(bond_dims[0] if isinstance(bond_dims, list) else bond_dims)

            # MPSInfo
            mps_info_thermal = MPSInfo(lcp_thermal)
            
            mps_info = MPSInfo(lcp)
            mps_info_d = { '_BRA': mps_info, '_KET': mps_info_thermal }
            
            mpo = MPO(hamil)
            mpo_info = MPOInfo(hamil)

            # Identity MPO
            impo = IdentityMPO(hamil)
            impo_info = IdentityMPOInfo(hamil)

            # Identity compression (fit mps_thermal to mps)
            ictr = DMRGContractor(mps_info_d, impo_info, Simplifier(NoTransposeRules(su2=self.opts["su2"])))
            
            # Thermal limit MPS
            mps_thermal = MPS(lcp_thermal, center=0, dot=2, iprint=self.verbose >= 2)
            mps_thermal.fill_thermal_limit()
            mps_thermal.canonicalize()

            m = bond_dims[0] if isinstance(bond_dims, list) else bond_dims
                
            # Random MPS (for fitting thermal limit MPS)
            mps = MPS(lcp, center=0, dot=2, iprint=self.verbose >= 2)
            mps.canonicalize(random=True)
                
            # MPSInfo
            mps_info = MPSInfo(lcp)
            ictr.mps_info.update({ '_BRA': mps_info, '_KET': mps_info_thermal })
            
            cps = Compress(impo, mps, mps_thermal, bond_dims=m, ket_bond_dim=10, contractor=ictr, noise=[1E-4]* 2 + [0])
            norm = cps.solve(forward=True, n_sweeps=50, tol=1E-9)
            mps0 = cps.mps
            assert abs(norm - 1) <= 1E-9
            
            pickle.dump((mps0, mps_info, cps.forward), open(self.init_mps, "wb" ))
        
        if self.verbose >= 2:
            print('>>> COMPLETE generate initial mps | Time = %.2f <<<' % (time.perf_counter() - t))
    
    def write_fcidump_general(self, n_mo, n_elec, h1e, g2e):
        """Write orbitals to FCIDUMP file from integrals with no assumed symmetry."""
        assert h1e.shape == (n_mo, n_mo)
        assert g2e.shape == (n_mo, n_mo, n_mo, n_mo)
        write_fcidump(self.fcidump, h1e, g2e, n_mo, n_elec, nuc=0.0, ms=0, orbsym=None, tol=1e-13)
    
    def write_fcidump(self, mol):
        """Write orbitals to FCIDUMP file for the molecule."""
        if self.opts["su2"]:
            self.write_fcidump_su2(mol)
        else:
            self.write_fcidump_sz(mol)

    def write_fcidump_su2(self, mol):
        """Write RHF orbitals FCIDUMP for the molecule."""
        
        n_elec = mol.nelectron
        assert self.n_elec == n_elec
        assert mol.symmetry.lower() == self.opts['pg']
        m = scf.RHF(mol)
        m.kernel()
        
        mo_coeff = m.mo_coeff
        n_ao = mo_coeff.shape[0]
        n_mo = mo_coeff.shape[1]

        pg_reorder = True

        if self.opts['pg'] == 'd2h':
            fcidump_sym = ["Ag", "B3u", "B2u", "B1g", "B1u", "B2g", "B3g", "Au"]
            optimal_reorder = ["Ag", "B1u", "B3u", "B2g", "B2u", "B3g", "B1g", "Au"]
        elif self.opts['pg'] == 'c1':
            fcidump_sym = ["A"]
            optimal_reorder = ["A"]
        else:
            raise FTDMRGError("Point group %d not supported yet!" % self.opts['pg'])
        orb_sym_str = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo_coeff)
        orb_sym = np.array([fcidump_sym.index(i) + 1 for i in orb_sym_str])

        # sort the orbitals by symmetry for more efficient DMRG
        if pg_reorder:
            idx = np.argsort([optimal_reorder.index(i) for i in orb_sym_str])
            orb_sym = orb_sym[idx]
            mo_coeff = mo_coeff[:, idx]
            self.ridx = np.argsort(idx)
        else:
            # keep track of how orbitals are reordered
            self.ridx = np.array(list(range(n_mo), dtype=int))

        h1e = mo_coeff.T @ m.get_hcore() @ mo_coeff
        g2e = ao2mo.kernel(mol, mo_coeff)
        ecore = mol.energy_nuc()

        if self.verbose >= 2:
            print('E(SCF) = ', m.energy_tot(), ecore)

        ecore = 0.0
        write_fcidump(self.fcidump, h1e, g2e, n_mo, n_elec, nuc=ecore, ms=0, orbsym=orb_sym, tol=1e-13)
    
    def write_fcidump_sz(self, mol):
        """Write UHF orbitals FCIDUMP for the molecule."""
        
        n_elec = mol.nelectron
        assert self.n_alpha + self.n_beta == n_elec
        assert mol.symmetry.lower() == self.opts['pg']
        m = scf.UHF(mol)
        m.kernel()
        
        mo_coeff_a, mo_coeff_b = m.mo_coeff[0], m.mo_coeff[1]
        n_ao = mo_coeff_a.shape[0]
        n_mo = mo_coeff_b.shape[1]

        pg_reorder = True

        if self.opts['pg'] == 'd2h':
            fcidump_sym = ["Ag", "B3u", "B2u", "B1g", "B1u", "B2g", "B3g", "Au"]
            optimal_reorder = ["Ag", "B1u", "B3u", "B2g", "B2u", "B3g", "B1g", "Au"]
        elif self.opts['pg'] == 'c1':
            fcidump_sym = ["A"]
            optimal_reorder = ["A"]
        else:
            raise FTDMRGError("Point group %d not supported yet!" % self.opts['pg'])

        orb_sym_str_a = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo_coeff_a)
        orb_sym_str_b = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo_coeff_b)
        orb_sym_a = np.array([fcidump_sym.index(i) + 1 for i in orb_sym_str_a])
        orb_sym_b = np.array([fcidump_sym.index(i) + 1 for i in orb_sym_str_b])

        # sort the orbitals by symmetry for more efficient DMRG
        if pg_reorder:
            idx_a = np.argsort([optimal_reorder.index(i) for i in orb_sym_str_a])
            orb_sym_a = orb_sym_a[idx_a]
            mo_coeff_a = mo_coeff_a[:, idx_a]
            idx_b = np.argsort([optimal_reorder.index(i) for i in orb_sym_str_b])
            orb_sym_b = orb_sym_b[idx_b]
            mo_coeff_b = mo_coeff_b[:, idx_b]
            assert np.allclose(idx_a, idx_b)
            assert np.allclose(orb_sym_a, orb_sym_b)
            self.ridx = np.argsort(idx_a)
        else:
            # keep track of how orbitals are reordered
            self.ridx = np.array(list(range(n_mo), dtype=int))

        h1ea = mo_coeff_a.T @ m.get_hcore() @ mo_coeff_a
        h1eb = mo_coeff_b.T @ m.get_hcore() @ mo_coeff_b
        g2eaa = ao2mo.restore(8, ao2mo.kernel(mol, mo_coeff_a), n_mo)
        g2ebb = ao2mo.restore(8, ao2mo.kernel(mol, mo_coeff_b), n_mo)
        g2eab = ao2mo.kernel(mol, [mo_coeff_a, mo_coeff_a, mo_coeff_b, mo_coeff_b])
        ecore = mol.energy_nuc()

        if self.verbose >= 2:
            print('E(SCF) = ', m.energy_tot(), ecore)

        ecore = 0.0
        write_fcidump(self.fcidump, (h1ea, h1eb), (g2eaa, g2eab, g2ebb), n_mo, n_elec, nuc=ecore, ms=0, orbsym=orb_sym_a, tol=1e-13)
