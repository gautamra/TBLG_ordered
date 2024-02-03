from m2_helper import *

import argparse
parser=argparse.ArgumentParser()
parser.add_argument("beta", help="inverse temperature", type=float)
parser.add_argument("loops", help="number of DMFT loops", type=int)
parser.add_argument("log_n", help="10^log_n number of QMC cycles", type=int)
parser.add_argument("polarizer", help="field that polarizes to the starting condition", type=float)
parser.add_argument("--mix", help="0.51-1.0 default:0.9", type = float)
parser.add_argument("--symm", help="K-IVC or IVC default:K-IVC", type = str)
parser.add_argument("--restart_from", help="neglect all runs after", type = int)
args=parser.parse_args()
beta = args.beta
log_n = args.log_n
loops = args.loops
polarizer = args.polarizer

###################### Check nu, beta, and pick filename for set of system params
nu = -2.0
broken_symm = 'K-IVC'
Boltzmann=8.617333262e-5*1000
T=1/(Boltzmann*beta)
p={}
p['fit_min_n'] = int(160*T**(-0.9577746167096538))
p["fit_max_n"] = int(209*T**(-0.767031728744396))
p["imag_threshold"] = 1e-10
filename = 'Data/beta_{:.2f}/nu_{:.2f}'.format(beta, nu)
################### 

# Not to be touched
mix = 0.9
if args.mix:
    mix = args.mix
    # filename = filename+"mix_{:.2f}".format(mix)
if args.symm:
    broken_symm=args.symm
    filename = 'Data/'+broken_symm+'/'+filename[5:]
p["n_cycles"] = (10**log_n)//(160*4)
constrain=False
sample_len = 9
BZ_sampling, weights = sample_BZ_direct(sample_len)
n_iw = 1025
prec_mu = 0.001
if polarizer>0:
    prec_mu = prec_mu*10
p["length_cycle"] = 1000
p["n_warmup_cycles"] = 5000
p["perform_tail_fit"] = True
p["fit_max_moment"] = 4

#################### Choose the right symmetry-breaking here:
if broken_symm == 'K-IVC':
    dm_init = {}  # K-IVC
    dm_init['up'] = lin.block_diag(np.diag([1/2,1/2,1/2,1/2]),np.diag([1/2,1/2,1/2,1/2])+0.05*np.kron(sigy,sigy), + np.diag([1/2,1/2,1/2,1/2])-1/2*np.kron(sigy,sigy))
    dm_init['down'] = lin.block_diag(np.diag([1/2,1/2,1/2,1/2]),np.diag([1/2,1/2,1/2,1/2]), + np.diag([0,0,0,0]))
    deg_shells = [[['up_0', 'up_1'],['down_0', 'down_1', 'down_2', 'down_3']]]
elif broken_symm == 'IVC':
    dm_init = {}  # IVC just standard IVC calculation
    dm_init['up'] = lin.block_diag(np.diag([1/2,1/2,1/2,1/2]),np.diag([1/2,1/2,1/2,1/2])+0.05*np.kron(sigx,sigx), + np.diag([1/2,1/2,1/2,1/2])-1/2*np.kron(sigx,sigx))
    dm_init['down'] = lin.block_diag(np.diag([1/2,1/2,1/2,1/2]),np.diag([1/2,1/2,1/2,1/2]), + np.diag([0,0,0,0]))
    deg_shells = [[['up_0', 'up_1'],['down_0', 'down_1', 'down_2', 'down_3']]]
elif broken_symm == 'IVC2':
    dm_init = {}  # IVC just standard IVC calculation
    dm_init['up'] = lin.block_diag(np.diag([1/2,1/2,1/2,1/2]),np.diag([1/2,1/2,1/2,1/2])+0.05*np.kron(sigx,sigx), + np.diag([1/2,1/2,1/2,1/2])-1/2*np.kron(sigx,sigx))
    dm_init['down'] = lin.block_diag(np.diag([1/2,1/2,1/2,1/2]),np.diag([1/2,1/2,1/2,1/2]), + np.diag([0,0,0,0]))
    deg_shells = [[['up_0', 'up_1'],['down_0', 'down_1', 'down_2', 'down_3']]]
elif broken_symm == 'IVC3':
    dm_init = {}  # IVC3=IVC except disallow VP explicitly
    dm_init['up'] = lin.block_diag(np.diag([1/2,1/2,1/2,1/2]),np.diag([1/2,1/2,1/2,1/2])+0.05*np.kron(sigx,sigx), + np.diag([1/2,1/2,1/2,1/2])-1/2*np.kron(sigx,sigx))
    dm_init['down'] = lin.block_diag(np.diag([1/2,1/2,1/2,1/2]),np.diag([1/2,1/2,1/2,1/2]), + np.diag([0,0,0,0]))
    deg_shells = [[['up_0', 'up_1'],['down_0', 'down_1', 'down_2', 'down_3']]]
elif broken_symm == 'VP':
    dm_init = {}  # VP
    dm_init['up'] = lin.block_diag(np.diag([1/2,1/2,1/2,1/2]),np.diag([1/2,1/2,1/2,1/2]), + np.diag([1,1,0,0]))
    dm_init['down'] = lin.block_diag(np.diag([1/2,1/2,1/2,1/2]),np.diag([1/2,1/2,1/2,1/2]), + np.diag([0,0,0,0]))
    deg_shells = [[['up_0', 'up_1'],['up_2','up_3'],['down_0', 'down_1', 'down_2', 'down_3']]]
else:
    raise ValueError("Broken symmetry can only by VP, K-IVC or IVC right now.")
    
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
mu_ph=12
if broken_symm=='IVC2':
    H = lambda kx, ky: H0(kx, ky) + np.diag([0]*4+[mu_ph]*4+[-mu_ph]*4)
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
                    if args.restart_from:
                        previous_runs=args.restart_from
            else:
                f.create_group('dmft_output')
previous_runs = mpi.bcast(previous_runs)
previous_present = mpi.bcast(previous_present)

# TODO Save run parameters
if mpi.is_master_node():
    with HDFArchive(filename+'.h5', 'a') as ar:
        ar['dmft_output']['params_%i-%i'%(previous_runs+1, loops+previous_runs)] = (loops, p['n_cycles'], mix, constrain, polarizer)


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
                dm = ar['dmft_output']['dm-%i'%previous_runs]
        
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
                Sig << np.diag([np.average(np.diag(temp))]*len(temp)) + (nu*(W1+W3)/2)
            elif 'down' in name:
                temp = -tt[name]@Hmf['down'][-4:,-4:]@(tt[name].conjugate().T)
                Sig << np.diag([np.average(np.diag(temp))]*len(temp)) + (nu*(W1+W3)/2)
            else:
                raise("problem with self-energy initialization")
        chemical_potential=(nu*(W1+W3)/2)
        if mpi.is_master_node():
            with HDFArchive(filename+'.h5', 'a') as f:
                f['dmft_output']['mu-0'] = chemical_potential

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
        
    SK.set_Sigma([ S.Sigma_iw ])                            # put Sigma into the SumK class
    chemical_potential=0
    if mpi.is_master_node():
        with HDFArchive(filename+'.h5', 'r') as f:
            chemical_potential = f['dmft_output']['mu-%i'%(previous_runs+iteration_number-1)]
    chemical_potential = mpi.bcast(chemical_potential)
    SK.set_mu(chemical_potential)
    chemical_potential = SK.calc_mu(precision = prec_mu , delta = 10)  # find the chemical potential for given density
    if chemical_potential==None:
        if mpi.is_master_node():
            with HDFArchive(filename+'.h5', 'r') as f:
                chemical_potential = f['dmft_output']['mu-%i'%(previous_runs+iteration_number-1)]
        chemical_potential = mpi.bcast(chemical_potential)
        SK.set_mu(chemical_potential)
        chemical_potential = SK.calc_mu(precision = prec_mu*10 , delta = 10)
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
        if broken_symm=='IVC3':
            ccf = np.trace(dm['up'][-4:,-4:])/4
            cc1 = np.trace(dm['up'][:4,:4])/4
            cc2 = np.trace(dm['up'][4:-4,4:-4])/4
            for i in range(4):
                dm['up'][-4+i,-4+i]=ccf
                dm['up'][i,i]=cc1
                dm['up'][4+i,4+i]=cc2

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
        print('saved sigma, G etc')
    
    SK.save(['chemical_potential','dc_imp','dc_energ'])