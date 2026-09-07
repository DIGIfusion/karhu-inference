import torch
import numpy as np

TANH_NORMALIZER = (np.tanh(1) - np.tanh(-1))*1.155
TANH_NORMALIZER_TI = np.tanh(1) - np.tanh(-100)

def mtanh_hel(x,  aped=10.0, asep=2.0, delta=0.04,shift=torch.tensor(0.0), a1=0.0, exp1=0.0, exp2=0.0):
    """
    EPED mtanh profile definition Snyder PoP 16 056118 (2009)
    
    Args:
        x (array): Psi-grid
        aped (float): Pedestal top value
        asep (float): Separatrix value
        delta (float): Pedestal width
        shift (float): Pedestal shift
        a1 (float): Core slope multiplier
        exp1 (float): Core slope exponent 1
        exp2 (float: Core slope exponent 2 
    """
    pos = torch.tensor(1.0) - 0.5*delta + shift
    pos = pos.detach().clone().to(torch.float32)
    pedestal = torch.tensor(1.0) - delta + shift
    pedestal = pedestal.detach().clone().to(torch.float32)
    xin = x.detach().clone().to(torch.float32)
    
    ase0 = asep
    a0 = aped
    
    # Base mtanh-profile
    output = torch.zeros(len(x))
    output = ase0 + a0*(torch.tanh(2*(1-pos)/delta) - torch.tanh(2*(xin-pos)/delta) )

    # Core slope implemented via multiplier (a1), and two exponents.
    idx = np.where(x > pedestal)
    for i in range(int(idx[0][0])):
        output[i] += a1*(1 - (x[i]/pedestal)**exp1)**exp2
    return output
    
def get_circumference(theta, ellip, tria, quad, rvac, a):
        """
        Calculate the circumference of the parametrised shape.

        Args:
            theta (array): Array of poloidal angles.
            ellip (float): Ellipticity.
            tria (float): Triangularity.
            quad (float): Squareness.
            rvac (float): Vacuum radius.
            a (float): Minor radius.

        Returns:
            float: The circumference of the shape.
        """
        (r, z) = shape_function(theta, ellip=ellip, tria=tria, quad=quad, rvac=rvac, a=a)
        dr = r[0:-1] - r[1:]
        dz = z[0:-1] - z[1:]
        ds = torch.pow(torch.real(torch.pow(dr, 2)) + torch.real(torch.pow(dz, 2)), .5)
        circumference = torch.sum(ds)
        return circumference    
        
def shape_function(theta, ellip, tria, quad, rvac, a):
        """
        EUROPED
        returns the R and Z values given the boundary parameters (HELENA equation 11)
        ellip = ellipticity (input)
        tria = triangularity (input)
        quad = squareness (input)
        theta = the poloidal angle, can be a vector or a scalar (input)

        Args:
            theta (float or array): The poloidal angle.
            ellip (float): Ellipticity.
            tria (float): Triangularity.
            quad (float): Squareness.
            rvac (float): Vacuum radius.
            a (float): Minor radius.

        Returns:
            tuple: (r, z) where r and z are arrays of R and Z coordinates.
        """
        sint = np.sin(theta)
        r = rvac + a * np.cos(theta + tria * sint + quad * np.sin(theta * 2.0))
        z = ellip * a * sint
        return (r, z)
        
def get_polar_from_rz(r_vals, z_vals, symmetric=False):
        """
        Convert (R, Z) boundary coordinates to polar coordinates (rho, theta)
        relative to the boundary center (r0, z0).

        Handles both symmetric and asymmetric boundaries:

        - If symmetric (self.symmetric=True): input contains only the top half,
            and the function mirrors it to produce a full 0-2π contour.

        - If asymmetric: uses the full input directly.

        Args:
            r_vals (array_like): R (major radius) coordinates of the boundary.
            z_vals (array_like): Z (vertical) coordinates of the boundary.
            symmetric (bool): Whether the boundary is symmetric.

        Returns:
            tuple: (rho, theta) where rho is normalized radius and theta is
            poloidal angle.
        """
        #r_vals = np.asarray(r_vals)
        #z_vals = np.asarray(z_vals)
        r0 = (max(r_vals) + min(r_vals)) / 2
        ind = r_vals.argmax()
        z0 = 0 if symmetric else z_vals[ind]

        # Minor radius:
        amin = (max(r_vals) - min(r_vals)) / 2

        # Compute normalized radius and poloidal angle
        rho = torch.sqrt((r_vals - r0)**2 + (z_vals - z0)**2) / amin
        theta = torch.arctan2(z_vals - z0, r_vals - r0)

        # Convert θ range from (-π, π] → [0, 2π)
        theta = np.mod(theta, 2 * np.pi)

        # Sort points by increasing θ to ensure continuous boundary
        # order = np.argsort(theta)
        # return rho[order], theta[order]
        return rho, theta
        

