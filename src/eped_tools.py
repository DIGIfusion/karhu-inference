import torch
import numpy as np
from scipy.constants import e, mu_0
from scipy.interpolate import interp1d, CubicSpline

from utils import get_circumference, shape_function, get_polar_from_rz, mtanh_hel

TANH_NORMALIZER = (np.tanh(1) - np.tanh(-1))*1.155
TANH_NORMALIZER_TI = np.tanh(1) - np.tanh(-100)

def get_teped(d_ped, ip, circumference, neped, tesep=100.0, zimp=4.0, 
                              zeff=1.2, z_main_ion=1.0, width_const=0.076, dengrad=1.0, 
                              denexpo=-1.0):
        """
        Calculate teped according to the KBM constraint.

        Args:
            d_ped (float): Pedestal width.
            ip (float): Plasma current.
            circumference (float): Plasma circumference.
            neped (float): Pedestal electron density.
            tesep (float): Separatrix electron temperature.
            zimp (float): Impurity charge.
            zeff (float): Effective charge.
            z_main_ion (float): Main ion charge.

        Returns:
            float: Pedestal electron temperature.
        """
        # Constants
        tiped_multip = 1.0
        tisep_multip = 1.0
        #width_const = 0.076
        beta_exponent = 0.5

        pedestal_dilution = (zimp + z_main_ion - zeff) / zimp / z_main_ion
        bp2 = (ip * 1e6 * mu_0 / circumference) ** 2
        betapolped = (d_ped / (width_const*dengrad**denexpo)) ** (1.0 / beta_exponent)
        pped = betapolped / (2 * mu_0) * bp2 
       
        # Caluclate the pedestal temperature based on the KBM constraint
        ti_contribution = (
            1 + tiped_multip * pedestal_dilution *
            (1 + (1 - TANH_NORMALIZER / TANH_NORMALIZER_TI) * (tisep_multip - 1.0) * tesep))
     
        
        teped = pped / neped / ti_contribution / e / 1e19
        return teped    
        
def run_eped_profile(x0, eq_model, karhu_model, dataset, ip=2.0, b_mag=2.0, neped=3.0, 
                                           nesep=1.5, r_mag=2.94, shift=0.015, wconst=0.076, tria=0.25, a=0.88, 
                                           tesep=100, delta=0.02, beta_N=1.3, zeff=1.0, stability_fraction=1.0, 
                                           tiratio=1.0, dengrad=1.0, slope_c = 0.0, slope_exp1=1.0, slope_exp2=1.0,
                                           neprofin=torch.tensor(0), 
                                           teprofin=torch.tensor(0),
                                           theta_space=torch.linspace(1e-3, 2*np.pi, 128)):
        """
        This routine builds mtanh profiles and computes their linear MHD stability with KARHU.
        
        
        Args:
            x0 (array): Psi-grdi
            eq_model (PyTorch model): Equilbrium regression model
            karhu_model (PyTorch model): KARHU MHD stability surrogate model
            dataset (PyTorch dataset): The dataset that was used to train the models - Needed for normalization information
            ip (float): Plasma current (MA)
            b_mag (float): Toroidal magnetic field (T)
            neped (float): Pedestal density (1e19 m-3)
            r_mag (float): Plasma major radius (m)
            shift (float): Density shift if any
            wconst (float): Pedestal width constant in the transport model
            tria (float): Plasma triangularity
            a (float): Plasma minor radius (m)
            tesep (float): Separatrix temperature
            delta (float): Pedestal width
            beta_N (float): normalized beta (%)
            zeff (float): Effective charge of the plasma
            stability_fraction: fraction with whic h to push the plasma pressure towards MHD instability
            tiratio (float): Ratio of Ti to Te
            dengrad (float): Normalized density gradient (grad_n/n) at edge.
            slope_c (float): Multiplier for the core slope
            slope_exp1 (float): Exponent 1 in the core slope
            slope_exp2 (float): Exponent 2 in the core slope
            neprofin (torch.tensor): input density profile, if used
            teprofin (torch.tensort): input temperature profile, if used
            thete_space (array): theta_grid that is used in the shape calculations
        """
        
        
        
        # Compute shape input from triangularity:
        cm = get_circumference(torch.linspace(0,2*np.pi, 100), 1.67, tria, 0, 2.94, a)
        (r, z) = shape_function(theta_space, 1.67, tria, 0.0, 2.94, 0.88)
        rho, theta = get_polar_from_rz(r, z)
        interpolation_function = interp1d(theta, rho, kind="linear", fill_value='extrapolate')
        shape_int = interpolation_function(theta_space)
        shape_int = torch.tensor(shape_int, dtype=torch.float32)
        
        # Compute pedestal temperature, based on the assumed transport model
        teped = get_teped(delta, ip, cm, neped, width_const=wconst, dengrad=dengrad, zeff=1.0)        

        # If pedestal temperature is below 100 eV. Return immediately as there is no pedestal.
        if teped < 100:
            return torch.tensor([0]), torch.zeros(len(neprofin)), torch.zeros(len(neprofin))

        # Build pedestal profiles
        if torch.max(neprofin) == 0:      
            apedval = (neped - nesep)/TANH_NORMALIZER
            nesepfix = nesep #+ neasymp2 - neasymp1
            apedvalfix =  apedval #+ neasymp2 - neasymp
            neprof= mtanh_hel(x0, aped=apedvalfix, asep=nesepfix, delta=delta, shift=shift)     
        else:
           # Use input density profile
           neprof = neprofin
        tepedval = (teped - tesep)/TANH_NORMALIZER
        teprof = mtanh_hel(x0, aped=tepedval, asep=tesep, delta=delta, a1=slope_c, exp1=slope_exp1, 
                                               exp2=slope_exp2)
        # This is used to compute the linear MHD stability. Notice the application of the stability fraction.
        teprof_ins = mtanh_hel(x0, aped=tepedval/stability_fraction, asep=tesep, delta=delta, 
                                                       a1=slope_c, exp1=slope_exp1, exp2=slope_exp2)
        tiprof_ins = teprof_ins*tiratio
        
        # Normalize the inputs to be used in the ML models
        input_ne = dataset.minmax(neprof, 
                                                   dataset.scaling_params["ne"][0],
                                                   dataset.scaling_params["ne"][1],
                                                   )
        input_Te = dataset.minmax(teprof_ins, 
                                                   dataset.scaling_params["Te"][0],
                                                   dataset.scaling_params["Te"][1],
                                                   )
        input_Ti = dataset.minmax(tiprof_ins, 
                                                   dataset.scaling_params["Ti"][0],
                                                   dataset.scaling_params["Ti"][1],
                                                   )
        input_shape = dataset.minmax(shape_int, 
                                                   dataset.scaling_params["shape"][0],
                                                   dataset.scaling_params["shape"][1],
                                                   )
        if dataset.scaling_params["b_mag"][0] == dataset.scaling_params["b_mag"][1]:
            input_b_mag = torch.tensor(dataset.scaling_params["b_mag"][0], dtype=torch.float32)
        else:
            input_b_mag = dataset.minmax(b_mag, 
                                                   dataset.scaling_params["b_mag"][0],
                                                   dataset.scaling_params["b_mag"][1],
                                                   )

        if dataset.scaling_params["r_mag"][0] == dataset.scaling_params["r_mag"][1]:
            input_r_mag = torch.tensor(dataset.scaling_params["r_mag"][0], dtype=torch.float32)
        else:
            input_r_mag = dataset.minmax(r_mag, 
                                                   dataset.scaling_params["r_mag"][0],
                                                   dataset.scaling_params["r_mag"][1],
                                                   )
    
        if dataset.scaling_params["ip"][0] == dataset.scaling_params["ip"][1]:
            input_ip = torch.tensor(dataset.scaling_params["ip"][0], dtype=torch.float32)
        else:
            input_ip = dataset.minmax(ip, 
                                                   dataset.scaling_params["ip"][0],
                                                   dataset.scaling_params["ip"][1],
                                                   )
        if dataset.scaling_params["beta_n"][0] == dataset.scaling_params["beta_n"][1]:
            input_beta_n = torch.tensor(dataset.scaling_params["beta_n"][0], dtype=torch.float32)
        else:
            input_beta_n = dataset.minmax(beta_N, 
                                                   dataset.scaling_params["beta_n"][0],
                                                   dataset.scaling_params["beta_n"][1],
                                                   )
        if dataset.scaling_params["zeff"][0] == dataset.scaling_params["zeff"][1]:
            input_zeff = torch.tensor(dataset.scaling_params["zeff"][0], dtype=torch.float32)
        else:
            input_zeff = dataset.minmax(zeff, 
                                                   dataset.scaling_params["zeff"][0],
                                                   dataset.scaling_params["zeff"][1],
                                                   )  

        # Correct the tensor dimensions and make sure that dtype is float32.
        ine = torch.tensor(input_ne, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        ite = torch.tensor(input_Te, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        iti = torch.tensor(input_Ti, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        ish = torch.tensor(input_shape, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        ibm = torch.tensor(input_b_mag, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        irm = torch.tensor(input_r_mag, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        iip = torch.tensor(input_ip, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        ibn = torch.tensor(input_beta_n, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        izn = torch.tensor(input_zeff, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # ML models    
        y_pred = eq_model(ine, ite, iti, ish, 
                                           ibm, irm, iip, ibn, izn)
        gammap = karhu_model(y_pred[:,0].reshape((-1, 1, 64)), y_pred[:,1].reshape((-1, 1, 64)), y_pred[:,2].reshape((-1, 1, 64)),
                                                      ish, ibm, irm, ibn)
         
        # Denormalize the predictions
        gammap = dataset.descale_minmax(gammap, 
                                                   dataset.scaling_params["growthrate"][0],
                                                   dataset.scaling_params["growthrate"][1],
                                                   )
        # Peak gamma ready to be returned
        gams = gammap.detach().numpy()


        return gams, neprof, teprof
        
def run_eped_analysis(casedict, eq_model, karhu_model, dataset, x0=torch.linspace(0.9, 1.0, 64), 
                                              ip=2.0, b_mag=2.0, neped=3.0, 
                                              nesep=1.5, r_mag=2.94, shift=0.015, wconst=0.076, tria=0.25, a=0.88, 
                                              tesep=100, delta=0.02, beta_N=1.3, zeff=1.2, stability_fraction=1.0, 
                                              tiratio=1.0, dengrad=1.0, slope_c = 0.0, slope_exp1=1.0, slope_exp2=1.0,
                                              neprofin=torch.tensor(0), 
                                              teprofin=torch.tensor(0),
                                              theta_space=torch.linspace(1e-3, 2*np.pi, 128)):
    """
    This is a helper function to run eped-like analysis, assuming an experimental case represented in a dictionary.
    The dictionary is expected to contain following keys:
        - 'psi': psi-coordinate
        - 'nes': electron density (1e19 m-3)
        - 'tes': electron temperature (keV)
        - 'beta_N': normalized pressure (%)
        - 'tria': triangularity
        - 'bt': toroidal field (T)
        - 'ip': plasma current (A)
    """
    psis = torch.tensor(casedict['psi'], dtype=torch.float32)
    nes = torch.tensor(casedict['nes'], dtype=torch.float32)
    tes = torch.tensor(casedict['tes'], dtype=torch.float32)
    beta_N = torch.tensor(casedict['beta_N'], dtype=torch.float32)
    tria = torch.tensor(casedict['tria'], dtype=torch.float32)
    bt = torch.tensor(casedict['bt'], dtype=torch.float32)
    ip = torch.tensor(casedict['ip'], dtype=torch.float32)/1e6
     
    # Find T_e_sep = 100 eV to locate separatrix
    idx = torch.where(torch.abs(tes - 0.1) < 0.05)
    psif = CubicSpline(torch.flip(tes[idx], dims=[0]), torch.flip(psis[idx], dims=[0]))
    psisep = psif(0.1)
    shift = 1.0 - psisep
    # To be used later
    tesep = 100
     
    # Interpolate profiles
    nef = CubicSpline(psis[600:] + shift, nes[600:])              
    tef = CubicSpline(psis[600:] + shift, tes[600:])
    # Map to x0-grid and turn Te to units of eV
    netarget = torch.tensor(nef(x0), dtype=torch.float32)
    tetarget = torch.tensor(tef(x0), dtype=torch.float32)*1e3 
     
    if dengrad > 1:
        dnnfull = torch.diff(netarget)/torch.diff(x0)
        dnnedge = dnnfull[-5:]/netarget[-5:]
        dnn = torch.mean(dnnedge)
    else:
        dnn = 1.0
     

    gam = 0
    delta = 0.005
    while gam < 0.03:
        delta += 0.001
        # Pedestal density - location at 1.0 - pedestal width
        neped = netarget[0]
        gam, neprof1, teprof1 = run_eped_profile(x0, eq_model, karhu_model, dataset, neped=neped, b_mag=bt,
                                                                                         ip=ip, tria=tria, beta_N=beta_N, 
                                                                                         shift=0.00, wconst=wconst, delta=delta, zeff=2.0,
                                                                                         stability_fraction=stability_fraction, 
                                                                                         dengrad=dnn,
                                                                                         neprofin=torch.tensor(netarget, dtype=torch.float32),
                                                                                         teprofin = torch.tensor(tetarget, dtype=torch.float32),
                                                                                         )
    outputdict = {'ne_target':netarget, 'te_target':tetarget, 'ne_pred':neprof1, 'te_pred': teprof1, 'gamma':gam}
    return outputdict
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
         
