from CNP_helper import *

from pathlib import Path


import argparse
parser=argparse.ArgumentParser()
parser.add_argument("beta", help="inverse temperature", type=float)
parser.add_argument("loops", help="number of DMFT loops", type=int)
parser.add_argument("log_n_cycles", help='number of measurements per core = 10^log_n_markov', type=int)
parser.add_argument("n_cores", help='number of cores', type=int)
parser.add_argument("polarizer", help="field that polarizes to the starting condition", type=float)
parser.add_argument("--mix", help="0.51-1.0 default:0.9", type = float, default=0.9)
parser.add_argument("--filename", help="override default filename", type = str)
parser.add_argument("--path", help="override default pathname", type = str)
parser.add_argument("--phsymm", help="enforce ph symmetry?", type =str)
parser.add_argument("--sample_len", help="set sampling resolution of k-mesh", type=int, default=9)

args=parser.parse_args()
beta = args.beta
log_n_cycles = args.log_n_cycles
loops = args.loops
polarizer = args.polarizer
n_cores=args.n_cores
mix=args.mix
sample_len=args.sample_len
path=args.path
nu=-0.000

path="DataU2/beta_{:.2f}/".format(beta)
if args.path:
    path=args.path
if mpi.is_master_node():
    Path(path).mkdir(parents=True, exist_ok=True)
filename="nu_{:.3f}".format(nu)
if args.filename:
    filename=args.filename
filename=path+filename

###################### Check nu, beta, and pick filename for set of system params
nu = -0.0
broken_symm = 'K-IVC'
Boltzmann=8.617333262e-5*1000
T=1/(Boltzmann*beta)
p={}
phsymm=False
if args.phsymm:
    if args.phsymm=="True":
        phsymm=True
    elif args.phsymm=='False':
        phsymm=False
    else:
        raise ValueError("phsymm can only be 'True' or 'False'")
################### 

constrain=False
BZ_sampling, weights = sample_BZ_direct(sample_len)
n_iw = 1025
prec_mu = 0.001
p["length_cycle"] = 1000
p["n_warmup_cycles"] = 5000
p["perform_tail_fit"] = True
p["fit_max_moment"] = 4
p["n_cycles"] = (10**log_n_cycles)
p['fit_min_n'] = int(160*T**(-0.9577746167096538))
p["fit_max_n"] = int(209*T**(-0.767031728744396))
p["imag_threshold"] = 1e-10


#################### Choose the right symmetry-breaking here:
if broken_symm == 'K-IVC':
    dm_init = {}  # K-IVC
    dm_init['up'] = lin.block_diag(np.diag([1/2,1/2,1/2,1/2]),np.diag([1/2,1/2,1/2,1/2])+0.05*np.kron(sigy,sigy), + np.diag([1/2,1/2,1/2,1/2])-1/2*np.kron(sigy,sigy))
    dm_init['down'] = lin.block_diag(np.diag([1/2,1/2,1/2,1/2]),np.diag([1/2,1/2,1/2,1/2])+0.05*np.kron(sigy,sigy), + np.diag([1/2,1/2,1/2,1/2])-1/2*np.kron(sigy,sigy))
    deg_shells = [[['up_0', 'up_1'],['down_0', 'down_1']]]
else:
    raise ValueError("Broken symmetry can only by K-IVC right now.")
    
def symm_ph(Sigma):
    for name, Sig in Sigma:
        if mpi.is_master_node():
            print('enforcing ph symmetry assuming k-ivc symmetry')
        # center = (Sig[0,0] + Sig[1,1])(0).real/2
        Sig.real[0,0] = 3.5*U + (Sig[0,0].real - Sig[1,1].real)/2
        Sig.real[1,1] = 3.5*U - (Sig[0,0].real - Sig[1,1].real)/2


def set_converter(H):
    ## Write a converter for dft_tools
    n_k = len(BZ_sampling)
    density_required = 12+nu 
    n_shells = 3
    n_valleys = 2

    if mpi.is_master_node():
        with open(filename, "w") as A:
            A.write(str(n_k)+"\n"+str(density_required)+"\n"+str(n_shells)+"\n")
            # describe all shells
            for ish in range(n_shells):
                A.write("1 "+str(ish)+" 0 "+str(2*n_valleys)+"\n")
            # describe the correlated shells
            A.write("1\n")
            A.write("1 2 0 "+str(2*n_valleys)+" 0 0\n")
            A.write(str(n_valleys)+" 2"*n_valleys+"\n")

        for ik in range(n_k):
            with open(filename, "a") as A:
                for row in H(*BZ_sampling[ik]):
                    A.write("\n")
                    for hopping in row:
                        A.write(str(hopping.real)+" ")
                A.write("\n")
                for row in H(*BZ_sampling[ik]):
                    A.write("\n")
                    for hopping in row:
                        A.write(str(hopping.imag)+" ")
                A.write("\n")

    Converter = HkConverter(filename = filename)
    Converter.convert_dft_input()

    ## Fix the BZ weights for the converter
    if mpi.is_master_node():
        with HDFArchive(filename+'.h5','a') as f:
            f['dft_input']['bz_weights'] = weights           
            
# Initial setup 
H0 = SBhamiltonian()

H = lambda kx, ky: H0(kx, ky)
set_converter(H)
SK = SumkDFT(hdf_file=filename+'.h5',use_dft_blocks=True)

## Check if previous runs exist
previous_runs = 0
previous_present = False
if mpi.is_master_node():
    if os.path.isfile(filename+'.h5'):
        with HDFArchive(filename+'.h5','a') as f:
            if 'dmft_output' in f:
                ar = f['dmft_output']
                if 'iterations' in ar:
                    previous_present = True
                    previous_runs = ar['iterations']
            else:
                f.create_group('dmft_output')
previous_runs = mpi.bcast(previous_runs)
previous_present = mpi.bcast(previous_present)

# TODO Save run parameters
if mpi.is_master_node():
    with HDFArchive(filename+'.h5', 'a') as ar:
        ar['dmft_output']['params_%i-%i'%(previous_runs+1, loops+previous_runs)] = (loops, p['n_cycles'], n_cores, mix, constrain, polarizer)


for iteration_number in range(1,loops+1):
    if mpi.is_master_node(): print("Iteration = ", iteration_number)

    if not previous_present and iteration_number==1:
        dm = dm_init
        if mpi.is_master_node():
            with HDFArchive(filename+'.h5', 'a') as ar:
                ar['dmft_output']['dm-0'] = dm
     
    if previous_present and iteration_number==1:
        dm = 0
        if mpi.is_master_node():
            with HDFArchive(filename+'.h5', 'r') as ar:
                iterations = ar['dmft_output']['iterations']
                dm = ar['dmft_output']['dm-%i'%iterations]
        
    dm = mpi.bcast(dm)
    Hmf = {}
    
    Hmf['up'] = mean_field_terms(dm, spin='up')
    Hmf['down'] = mean_field_terms(dm, spin='down')
    if constrain:
        Hmf['up'] = mean_field_terms(dm_init, spin='up')
        Hmf['down'] = mean_field_terms(dm_init, spin='down')
        
    H_pol = {}
    H_pol['up'] = -polarizer*(dm_init['up'] - lin.block_diag(1/2*np.eye(8), (1/2+nu/8)*np.eye(4) ))
    H_pol['down'] = -polarizer*(dm_init['down'] - lin.block_diag(1/2*np.eye(8), (1/2+nu/8)*np.eye(4) ))

    H = lambda kx, ky: H0(kx, ky) + Hmf['up'] + H_pol['up']
    set_converter(H)
    if mpi.is_master_node():
        # add spin down part
        with HDFArchive(filename+'.h5','a') as f:
            f['dft_input']['SP'] = 1
            n_orb = f['dft_input']['n_orbitals']
            f['dft_input']['n_orbitals'] = np.concatenate((n_orb, n_orb), axis = 1)
            proj_mat = f['dft_input']['proj_mat']
            f['dft_input']['proj_mat'] = np.concatenate((proj_mat, proj_mat), axis = 1)
            hopping1 = f['dft_input']['hopping']
            hopping2 = hopping1.copy()
            for ik in range(len(BZ_sampling)):
                hopping2[ik,0,:,:] = H0(*BZ_sampling[ik]) + Hmf['down']+ H_pol['down']
            f['dft_input']['hopping'] = np.concatenate((hopping1, hopping2), axis = 1)
            
    SK = SumkDFT(hdf_file=filename+'.h5',use_dft_blocks=True)
    SK.calculate_diagonalization_matrix(write_to_blockstructure=True)
    
    n_orb = SK.corr_shells[0]['dim']
    spin_names = ["up","down"]
    orb_names = [i for i in range(n_orb)]
    gf_struct = SK.gf_struct_solver_list[0]
    Umat, Upmat = U_matrix_kanamori(n_orb=n_orb, U_int=U, J_hund=0)
    h_int = h_int_density(spin_names, orb_names, map_operator_structure=SK.sumk_to_solver[0], U=Umat, Uprime=Upmat)
    S = Solver(beta=beta, gf_struct=gf_struct)
    

    if not previous_present and iteration_number==1:
        tt = SK.block_structure.effective_transformation_solver[0]
        for name, Sig in S.Sigma_iw:
            if 'up' in name:
                temp = -tt[name]@Hmf['up'][-4:,-4:]@(tt[name].conjugate().T)
                Sig << np.diag([np.average(np.diag(temp))]*len(temp))
            elif 'down' in name:
                temp = -tt[name]@Hmf['down'][-4:,-4:]@(tt[name].conjugate().T)
                Sig << np.diag([np.average(np.diag(temp))]*len(temp))
            else:
                raise("problem with self-energy initialization")

        if mpi.is_master_node():
            with HDFArchive(filename+'.h5', 'a') as ar:
                ar['dmft_output']['Sigma_iw-0'] = S.Sigma_iw
                
    if mpi.is_master_node():
        with HDFArchive(filename+'.h5', 'r') as ar:
            S.Sigma_iw = ar['dmft_output']['Sigma_iw-%i'%(iteration_number+previous_runs-1)]
    S.Sigma_iw << mpi.bcast(S.Sigma_iw)
    
    SK.deg_shells = deg_shells
    if mpi.is_master_node():
        print(("The degenerate shells are ", SK.deg_shells))
    if iteration_number>1 or previous_present:    
        SK.symm_deg_gf(S.Sigma_iw,ish=0)
        if phsymm:
            symm_ph(S.Sigma_iw)
        
    SK.set_Sigma([ S.Sigma_iw ])                            # put Sigma into the SumK class
    chemical_potential = SK.calc_mu(precision = prec_mu , delta = 10)  # find the chemical potential for given density
    S.G_iw << SK.extract_G_loc()[0]                         # calc the local Green function
                                                                
    if mpi.is_master_node():
        mpi.report("Total charge of Gloc : %.6f"%S.G_iw.total_density())
        
    # Calculate new G0_iw to input into the solver:
    S.G0_iw << S.Sigma_iw + inverse(S.G_iw)
    S.G0_iw << inverse(S.G0_iw)
    
    # Solve the impurity problem:
    S.solve(h_int=h_int, **p)

    if mpi.is_master_node():
        mpi.report("Total charge of impurity problem : %.6f"%S.G_iw.total_density())

    Sigma_symm = S.Sigma_iw.copy()
    SK.symm_deg_gf(Sigma_symm,ish=0)
    if phsymm:
        symm_ph(Sigma_symm)
    SK.set_Sigma([Sigma_symm])
    dm = {}
    dm['up'] = 1j*np.zeros((12,12))
    dm['down'] = 1j*np.zeros((12,12))
    for ik in range(len(BZ_sampling)):
        w = weights[ik]
        G = SK.lattice_gf(ik)
        for name, g in G:
            dm[name] += w*g.density()
    for name, dens in dm.items():
        if mpi.is_master_node():
            mpi.report("Symmetrizing density. The largest imaginary component in the diagonal of the density matrix is {}".format(np.max(np.abs(np.diag(dm[name].imag)))))
        dm[name] = 1/2*(dm[name] + dm[name].conjugate().T)
    dm['up'] = 1/2*(dm['up'] + dm['down'])
    dm['down'] = dm['up'].copy()

    if mpi.is_master_node():
        with HDFArchive(filename+'.h5','r') as ar:
            mpi.report("Mixing Sigma and G with factor %s"%mix)
            S.Sigma_iw << mix * S.Sigma_iw + (1.0-mix) * ar['dmft_output']['Sigma_iw-%i'%(iteration_number+previous_runs-1)]
            if previous_present or iteration_number>1:
                S.G_iw << mix * S.G_iw + (1.0-mix) * ar['dmft_output']['G_iw-%i'%(iteration_number+previous_runs-1)]
            for name, mat in dm.items():
                dm[name]=mix*mat + (1.0-mix)*ar['dmft_output']['dm-%i'%(iteration_number+previous_runs-1)][name]
    S.G_iw << mpi.bcast(S.G_iw)
    S.Sigma_iw << mpi.bcast(S.Sigma_iw)
    dm = mpi.bcast(dm)
    
    if mpi.is_master_node():
        with HDFArchive(filename+'.h5', 'a') as ar:
            ar['dmft_output']['iterations'] = iteration_number + previous_runs  
            ar['dmft_output']['G_0-%i'%(iteration_number + previous_runs)] = S.G0_iw
            ar['dmft_output']['G_tau-%i'%(iteration_number + previous_runs)] = S.G_tau
            ar['dmft_output']['G_iw-%i'%(iteration_number + previous_runs)] = S.G_iw
            ar['dmft_output']['Sigma_iw-%i'%(iteration_number + previous_runs)] = S.Sigma_iw
            ar['dmft_output']['mu-%i'%(iteration_number + previous_runs)] = chemical_potential
            ar['dmft_output']['dm-%i'%(iteration_number + previous_runs)] = dm
            
            if iteration_number==1:
                ar['dmft_output'].create_group('args-{:03d}-{:03d}'.format(previous_runs+1,previous_runs+loops))
                print(ar['dmft_output'].keys())
                for k, v in vars(args).items():
                    if v!=None:
                        ar['dmft_output']['args-{:03d}-{:03d}'.format(previous_runs+1,previous_runs+loops)][k] = v
        print('saved sigma, G etc')
    
    SK.save(['chemical_potential','dc_imp','dc_energ'])
