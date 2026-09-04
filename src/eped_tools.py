import torch
from scipy.constants import e, mu_0

def get_teped(d_ped, ip, circumference, neped, tesep=100.0, zimp=4.0, zeff=1.2, z_main_ion=1.0, width_const=0.076, dengrad=1.0, denexpo=-1.0):
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
        #TODO: Add the possibility for Ti not equal Te
        #ti_contribution = (
        #    1 + tiped_multip * pedestal_dilution *
        #    (1 + (1 - TANH_NORMALIZER / TANH_NORMALIZER_TI) * (tisep_multip - 1.0) * tesep))
        ti_contribution = 1.0
        
        teped = pped / neped / ti_contribution / e / 1e19
        return teped    
        
def run_eped_profile(x0, eq_model, karhu_model, dataset, ip=2.0, b_mag=2.0, neped=3.0, nesep=1.5, r_mag=2.94,
                            shift=0.00, wconst=0.076, tria=0.25, a=0.88, tesep=100, delta=0.02,
                            beta_N=1.3, zeff=1.0, stability_fraction=1.0, tiratio=1.0, dengrad=-1, 
                            neprofin=torch.tensor(0), teprofin=torch.tensor(0)):
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
            neprofin (torch.tensor): input density profile, if used
            teprofin (torch.tensort): input temperature profile, if used
        """
        
        
        
        # Compute shape features:
        cm = get_circumference(torch.linspace(0,2*np.pi, 100), 1.67, tria, 0, 2.94, a)
        (r, z) = shape_function(theta_space, 1.67, tria, 0.0, 2.94, 0.88)
        rho, theta = get_polar_from_rz(r, z)
        interpolation_function = interp1d(theta, rho, kind="linear", fill_value='extrapolate')
        shape_int = interpolation_function(theta_space)
        shape_int = torch.tensor(shape_int, dtype=torch.float32)
        pastne = torch.tensor([0])
        pastte = torch.tensor([0])
        tria = torch.tensor([tria])
        beta_N = torch.tensor([beta_N])
        zeff = torch.tensor([zeff])
        insta = False
        #while delta < 0.08:
        teped = get_teped(delta, ip, cm, neped, width_const=wconst, dengrad=dengrad)
        if teped < 100:
            return torch.tensor([0]), torch.zeros(len(neprofin)), torch.zeros(len(neprofin))
        if torch.max(neprofin) == 0:      
            apedval = (neped - nesep)/TANH_NORMALIZER
            nesepfix = nesep #+ neasymp2 - neasymp1
            apedvalfix =  apedval #+ neasymp2 - neasymp
            neprof= mtanh_hel(x0, aped=apedvalfix, asep=nesepfix, delta=delta, shift=shift, slope=0.0)     
        else:
           neprof = neprofin
        tepedval = (teped - tesep)/TANH_NORMALIZER
        
        teprof = mtanh_hel(x0, aped=tepedval, asep=tesep, delta=delta, slope=0.0)
        #if len(teprofin) > 2:
        #    slope=0.0
        #    slope_step=0.2
        #    under = False
        #    over = False
        #    for i in range(20):
        #        if teprofin[-1] - teprof[-1] > 0:
        #            if under == False and over == True:
        #                slope_step = slope_step/2
        #            under = True
        #            over = False
        #            slope += slope_step
        #        else:
        #            if under == True and over == False:
        #                slope_step = slope_step/2
        #            under = False
        #            over = True
        #            slope  -= slope_step
        #        teprof = mtanh_hel(x0, aped=tepedval, asep=tesep, delta=delta, slope=slope)
        poke  = stability_fraction
        teprof_ins = mtanh_hel(x0, aped=tepedval/poke, asep=tesep, delta=delta, slope=0.0)
        tiprof_ins = teprof_ins*tiratio
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

        ine = torch.tensor(input_ne.unsqueeze(0).unsqueeze(0), dtype=torch.float32)
        ite = torch.tensor(input_Te.unsqueeze(0).unsqueeze(0), dtype=torch.float32)
        iti = torch.tensor(input_Ti.unsqueeze(0).unsqueeze(0), dtype=torch.float32)
        ish = torch.tensor(input_shape.unsqueeze(0).unsqueeze(0), dtype=torch.float32)
        ibm = torch.tensor(input_b_mag.unsqueeze(0).unsqueeze(0), dtype=torch.float32)
        irm = torch.tensor(input_r_mag.unsqueeze(0).unsqueeze(0), dtype=torch.float32)
        iip = torch.tensor(input_ip.unsqueeze(0).unsqueeze(0), dtype=torch.float32)
        ibn = torch.tensor(input_beta_n.unsqueeze(0), dtype=torch.float32)
        izn = torch.tensor(input_zeff.unsqueeze(0), dtype=torch.float32)
            
        y_pred = eq_model(ine, ite, iti, ish, 
                                           ibm, irm, iip, ibn, izn)
        gammap = karhu_model(y_pred[:,0].reshape((-1, 1, 64)), y_pred[:,1].reshape((-1, 1, 64)), y_pred[:,2].reshape((-1, 1, 64)),
                                                      ish, ibm, irm, ibn)
        gammap = dataset.descale_minmax(gammap, 
                                                   dataset.scaling_params["growthrate"][0],
                                                   dataset.scaling_params["growthrate"][1],
                                                   )
        gams = gammap.detach().numpy()


        return gams, neprof, teprof
