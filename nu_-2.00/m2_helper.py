# Basic functionality
import numpy as np
import scipy.linalg as lin
import math
from itertools import product
from h5 import *
import pickle

# triqs+CTHYB DMFT functionality
from triqs.gf import *
from triqs.operators import *
from triqs_cthyb import Solver
import triqs
import triqs.utility.mpi as mpi
from triqs.operators.util import *

# DFTtools interface
from triqs_dft_tools.converters.hk import *
from triqs_dft_tools.sumk_dft import *

# Plotting functionality
import matplotlib.pyplot as plt
from triqs.plot.mpl_interface import oplot

#
import os
import sys

sig0 = np.eye(2)
sigx = np.array([[0,1],[1,0]])
sigy = np.array([[0,-1j],[1j,0]])
sigz = np.array([[1,0],[0,-1]])

U = 57.95
W1 = 44.03
W3 = 50.20
V = 48.33
J = 16.38

def sample_BZ_direct(sample_len):
    K = 1.703 # 1/Angstrom
    theta = 1.05 # deg
    k = 2*K*np.sin(theta/2 * np.pi/180)
    # Moire reciprocal lattice vectors
    bM1, bM2 = k*np.array([np.sqrt(3)/2,3/2]), k*np.array([-np.sqrt(3)/2,3/2]) 
    b0,b1 = 1/(sample_len)*(bM1-bM2)/3, 1/(sample_len)*(2*bM1+bM2)/3
    b0,b1 = 1/(sample_len)*(bM1+bM2)/3, 1/(sample_len)*(2*bM2-bM1)/3
    weights = []
    ks = []
    shift=0
    
    for nn in range(sample_len):
        for mm in range(sample_len+1 - nn):
            if mm!=0:
                ks.append((mm+shift)*b0+((nn)+shift)*b1)
                ks.append((-mm+shift)*b0+(-(nn)+shift)*b1)
                if mm==sample_len:
                    weights.append(1/3)
                    weights.append(1/3)
                elif mm+nn==sample_len:
                    weights.append(1/2)
                    weights.append(1/2)
                else:
                    weights.append(1)
                    weights.append(1)
    pi = np.pi
    R = np.array([[np.cos(pi/3), np.sin(pi/3)],[-np.sin(pi/3), np.cos(pi/3)]])
    k1s = [R@k for k in ks]
    k2s = [R@k for k in k1s]
    weights = weights*3
    
    ks = ks+k1s+k2s    
    ks.append(np.array([0,0]))
    weights.append(1)
    
    da = lin.norm(np.cross(b0,b1))/lin.norm(np.cross(bM1,bM2))
    weights = da*np.array(weights)

    return ks, weights

def sample_BZ_kp(sample_len):
    K = 1.703 # 1/Angstrom
    theta = 1.05 # deg
    k = 2*K*np.sin(theta/2 * np.pi/180)
    # Moire reciprocal lattice vectors
    bM1, bM2 = k*np.array([np.sqrt(3)/2,3/2]), k*np.array([-np.sqrt(3)/2,3/2]) 
    b0,b1 = 1/(sample_len)*bM1, 1/(sample_len)*bM1
    ks = []
    shift = 0
    for nn in range(sample_len):
        for mm in range(sample_len+1 - nn):
            if mm!=0:
                ks.append((mm+shift)*b0+((nn)+shift)*b1)
                ks.append((-mm+shift)*b0+(-(nn)+shift)*b1)
    
    pi = np.pi
    R = np.array([[np.cos(pi/3), np.sin(pi/3)],[-np.sin(pi/3), np.cos(pi/3)]])
    k1s = [R@k for k in ks]
    k2s = [R@k for k in k1s]
    
    ks = ks+k1s+k2s    
    ks.append(np.array([0,0]))
    weights=np.zeros(len(ks))
    
    da = lin.norm(np.cross(b0,b1))/lin.norm(np.cross(bM1,bM2))
    weights = da*np.array(weights)

    return ks, weights

def SBhamiltonian(fixed_parameters={}):
    K = 1.703 # 1/A
    theta = 1.05 # deg
    k_theta = 2*K*np.sin(theta/2 * np.pi/180) # 1/A
    aM = 2*np.pi/(3*k_theta)*2 # A
    lam = 0.3375*aM # Damping factor for hybridization term
    
    # Fixed parameters:
    M = 3.697
    vstar = -4303
    vstarp = 1622
    gamma = -24.75
    
    # 4x4 blocks represent the Gamma_3 c, Gamma1+Gamma2 c, and f electrons in order.
    # Each 4x4 block is made up of 2x2 blocks representing valley K and K'in order.
    # The 4x4 blocks are built with np.kron(sig_nu, sig_mu), where 
    # the first (second) Pauli matrix is in valley (orbital) space.
    H33 = np.zeros((4,4))       # Gamma_3
    HMM = M*np.kron(sig0, sigx) # Gamma_1 + Gamma_2
    Hff = np.zeros((4,4))       # f
    H3M = lambda kx, ky: vstar*(kx*np.kron(sigz,sig0) + 1j*ky*np.kron(sig0, sigz))
    H3f = lambda kx, ky: np.exp(-lam**2*(kx**2+ky**2)/2)*( gamma*np.kron(sig0,sig0) + vstarp*(kx*np.kron(sigz,sigx)+ky*np.kron(sig0,sigy)) )
    HMf = np.zeros((4,4))
    H = lambda kx, ky: np.block([[H33, H3M(kx,ky), H3f(kx,ky)],
                                 [H3M(kx,ky).conj().T, HMM, HMf],
                                 [H3f(kx,ky).conj().T, HMf.conj().T, Hff]])
    return H

def get_nus(dm):
    """"
    returns the fillings of Gamma_3, Gamma_1 + Gamma_2, and f electrons in order
    """
    dm = (dm['up']+ dm['down'])
    nu_c3 = np.trace(dm[:4,:4])
    nu_c1p2 = np.trace(dm[4:-4,4:-4])
    nu_f = np.trace(dm[-4:,-4:])
    return nu_c3-4, nu_c1p2-4, nu_f-4

def mean_field_terms(dm, spin):
    H33 = 1j*np.zeros((4,4))       
    HMM = 1j*np.zeros((4,4))  
    Hff = 1j*np.zeros((4,4))       
    H3M = 1j*np.zeros((4,4))  
    H3f = 1j*np.zeros((4,4))  
    HMf = 1j*np.zeros((4,4))

    nu_c3, nu_c1p2, nu_f = get_nus(dm)
    dm_spin = (dm[spin]) 
    Of = dm_spin[-4:,-4:] - 1/2*np.eye(4)
    Ocp = dm_spin[:4,:4] - 1/2*np.eye(4)
    Ocpp = dm_spin[4:-4,4:-4]- 1/2*np.eye(4)
    Ofcp = dm_spin[-4:,:4]
    Ofcpp = dm_spin[-4:,4:-4]
    
    # Hartree W and V terms and residual U Hartree shift
    Hff += (-3.5*U + nu_c3*W1 + nu_c1p2*W3)*np.eye(4)
    H33 += nu_f*W1*np.eye(4) + (nu_c3 + nu_c1p2)*V*np.eye(4)
    HMM += nu_f*W3*np.eye(4) + (nu_c3 + nu_c1p2)*V*np.eye(4)
    
    # W Fock term
    H3f += -W1*Ofcp.T
    HMf += -W3*Ofcpp.T
    
    # J Hartree term
    J_Hartree_factor = -J*(np.kron(sig0, sig0)-np.kron(sigx,sigx))
    Hff += J_Hartree_factor*(Ocpp.T)
    HMM += J_Hartree_factor*(Of.T)
    # Does not include terms connecting spin sectors, Assumption: there are no spin--off-diagonal terms in the density marix
    
    # J Fock term
    Ofcpp_tot = dm['up'][-4:,4:-4]+dm['down'][-4:,4:-4]
    V3 = (Ofcpp_tot[1,1] + Ofcpp_tot[2,2])
    V4 = (Ofcpp_tot[0,0] + Ofcpp_tot[3,3]) 
    HMf += J*np.diag([V4,V3,V3,V4])
    
    H = np.block([[H33, H3M, H3f],
                [H3M.conj().T, HMM, HMf],
                [H3f.conj().T, HMf.conj().T, Hff]])
    return H

def path_BZ(num_ks):
    K = 1.703 # 1/Angstrom
    theta = 1.05 # deg
    k = 2*K*np.sin(theta/2 * np.pi/180)
    # Moire reciprocal lattice vectors
    bM1, bM2 = k*np.array([np.sqrt(3)/2,3/2]), k*np.array([-np.sqrt(3)/2,3/2])

    num_points = num_ks
    K_point = (2*bM1-bM2)/3
    M_point = (bM1 - bM2)/2
    Gamma_point = np.zeros(2)
    dk = lin.norm(K_point - Gamma_point)/num_points
    ks = np.linspace(K_point, Gamma_point, num_points)[:-1]
    ks2 = np.linspace(Gamma_point, M_point, int(lin.norm(M_point - Gamma_point)/dk))[:-1]
    ks3 = np.linspace( M_point,K_point, int(lin.norm(M_point - K_point)/dk))[:-1]
    BZ_path = np.vstack([ks,ks2,ks3])
    
    return BZ_path