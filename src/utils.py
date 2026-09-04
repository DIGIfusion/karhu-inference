import torch
import numpy as np
import matplotlib.pyplot as plt



def mtanh_hel(x,  aped=10.0, asep=2.0, delta=0.04,shift=0.0, a1=0.0, exp1=0.0, exp2=0.0):
    """
    EPED mtanh profile definition Snyder PoP 16 056118 (2009)
    """
    pos = 1.0 - 0.5*delta + shift
    pedestal = 1.0 - delta + shift
    ase0 = asep
    
    # Tanh normalization to get the right pedestal top density
    # When x = pedestal, y=aped
    # First tanh
    tanh1 = torch.tanh(torch.tensor(2*(1-pos)/delta, dtype=torch.float32))
    # Second tanh
    tanh2 = torch.tanh(torch.tensor(2*(pedestal - pos)/delta, dtype=torch.float32))
    # a0 calculation
    a0 = (aped - ase0)/(tanh1 - tanh2)
    
    # Base mtanh-profile
    output = torch.zeros(len(x))
    output = ase0 + a0*(torch.tanh(torch.tensor(2*(1-pos)/delta, dtype=torch.float32)) - torch.tanh(torch.tensor(2*(x-pos)/delta, dtype=torch.float32))) 

    # Core slope implemented via multiplier (a1), and two exponents.
    idx = np.where(x > pedestal)
    for i in range(int(idx[0][0])):
        output[i] += a1*(1 - (x[i]/pedestal)**exp1)**exp2
    return output
    

