import numpy as np
import matplotlib.pyplot as plt
import threading as th

from math import ceil, sqrt

from Schulz_Zimm import SZ_avg, SZ_PPF
# Schulz-Zimm module handles mathematical operations related to particle size distributions

class Spheroid:
    """
        Class representing a 3D spheroid scatterer.
        This includes methods for generating scatterers and calculating their Debye scattering patterns.
        """

    # Initialie the Spheroid object with parameters
    def __init__(
        self, 
        R: float, 
        epsilon: float, 
        p: float, 
        rho: float=0.001
    ) -> None:

        self.R = R
        self.epsilon = epsilon
        self.p = p
        self.rho = rho

        # Calculate effective radius for the spheroid
        self.r_eff = (4/3)*((p + 1)/(p + 2))*R

        # Calculate volume of the spheroid and determine the number of scatterers
        V = np.pi*(4/3)*epsilon*R**3
        self.V = V
        self.num = int(V*rho)   # Total number of scatterers
    
    # Generate scatterer positions for the spheroid
    def generate_scatterers(self, n: int) -> np.ndarray:
        
        R = self.R
        epsilon = self.epsilon
        p = self.p

        # Generate random positions within the spheroid using spherical coordinates
        y = np.random.uniform(-1.0 + 1e-6, 1.0 - 1e-6, n).astype('f')
        rho_arr = R*np.power(np.random.rand(n).astype('f'), 1/(p + 1))
        theta_arr = np.arccos(epsilon*y/np.sqrt(1 - np.square(y)*(1 - epsilon**2)))
        phi_arr = 2*np.pi*np.random.rand(n).astype('f')

        # Compute scatterer positions in Cartesian coordinates
        r_arr = epsilon*rho_arr*np.sqrt((epsilon**2 - 1)*np.square(np.sin(theta_arr)) + 1)
        x_arr = r_arr*np.sin(theta_arr)*np.cos(phi_arr)
        y_arr = r_arr*np.sin(theta_arr)*np.sin(phi_arr)
        z_arr = r_arr*np.cos(theta_arr)

        # Center the scatterers around the origin
        x_arr = x_arr - np.mean(x_arr)
        y_arr = y_arr - np.mean(y_arr)
        z_arr = z_arr - np.mean(z_arr)
        
        return np.hstack((x_arr.reshape(-1, 1), y_arr.reshape(-1, 1), z_arr.reshape(-1, 1)))
    
    # Compute the Debye scattering pattern for the spheroid
    def Debye_Scattering(
        self, 
        q_arr: np.ndarray, 
        pop: int=2048, 
        div: int=128
    ) -> np.ndarray:

        # Determine the number of attempts needed based on the population size
        tries = ceil(ceil(self.num/pop)**2/2)
        shape = q_arr.shape
        f_total = np.zeros(shape=shape, dtype='f')  # Initialize total scattering intensities

        for i in range(tries):
            # Generate scatterer positions
            scatterers = self.generate_scatterers(pop)

            # Compute pairwise distances between scatterers
            r_ij = np.zeros(shape=(pop, pop), dtype='f')
            for j in range(3):
                r = scatterers[:, j].reshape(-1, 1)
                r_ij += np.square(r - r.T)

            # Determine distance bins and bin edges
            max_ = np.max(r_ij)
            bins = np.square(np.linspace(start=0, stop=1, num=div + 1, dtype='f'))*max_
            inds = np.digitize(r_ij, bins)
            vals, count = np.unique(inds, return_counts=True)

            # Map indices to corresponding distances
            vals = np.sqrt(max_)*(vals - 0.5)/div

            # Compute scattering intensity for each q value
            f_q = np.zeros(shape=shape, dtype='f')
            for l, q in enumerate(q_arr):
                qr = np.multiply(q, vals)
                f_q[l] += np.sum(count*np.sin(qr)/qr)

            # Normalize the scattering intensity
            f_q /= np.max(f_q)
            f_q[f_q <= 0] = np.min(f_q[f_q > 0])    # Handle non-positive values
            f_total += f_q  # Accumulate intensities
                
        return f_total/tries    # Average the intensities over all attempts
    
# Class representing a 3D cyclindrical scatterer.
class Cylinder:
    # Initialize the Cylinder object with parameters
    def __init__(
        self, 
        R: float, 
        L: float, 
        p: float, 
        rho: float=0.001
    ) -> None:
        
        self.R = R
        self.L = L
        self.p = p
        self.rho = rho

        # Calculate effective radius for the cylinder
        self.r_eff = (3/2)*((p + 1)/(p + 2))*R

        # Calculate volume of the cylinder and determine the number of scatterers
        V = np.pi*(R**2)*L
        self.V = V
        self.num = int(V*rho)   # Total number of scatterers
    
    # Generate scatterer positions for the cylinder
    def generate_scatterers(self, n: int) -> np.ndarray:
        R = self.R
        L = self.L
        p = self.p

        # Generate random positions within the cylinder using cylindrical coordiantes
        rho_arr = R*np.power(np.random.rand(n).astype('f'), 1/(1 + p))
        theta_arr = 2*np.pi*np.random.rand(n).astype('f')
        x_arr = rho_arr*np.cos(theta_arr)
        y_arr = rho_arr*np.sin(theta_arr)
        z_arr = L*np.random.rand(n).astype('f') # Uniformly distributed along cylinder length
        
        return np.hstack((x_arr.reshape(-1, 1), y_arr.reshape(-1, 1), z_arr.reshape(-1, 1)))
    
    # Calculates the Debye scattering intensity for the given array of scattering wave vectors
    def Debye_Scattering(
        self, 
        q_arr: np.ndarray, 
        pop: int=2048, 
        div: int=128
    ) -> np.ndarray:

        # Calculate the number of iterations for averaging scattering contributions
        tries = ceil(ceil(self.num/pop)**2/2)
        shape = q_arr.shape

        # Initialize total scattering intensity to zero
        f_total = np.zeros(shape=shape, dtype='f')

        # Loop over the number of tries to average the scattering results
        for i in range(tries):
            # Generate scatterers for the current batch
            scatterers = self.generate_scatterers(pop)

            # Initialize the pairwise distance matrix
            r_ij = np.zeros(shape=(pop, pop), dtype='f')

            # Compute squared pairwise distance for x, y, and z coordinates
            for j in range(3):
                r = scatterers[:, j].reshape(-1, 1)
                r_ij += np.square(r - r.T)

            # Find the maximum pairwise distance
            max_ = np.max(r_ij)

            # Divide distances into bins using square scaling
            bins = np.square(np.linspace(start=0, stop=1, num=div + 1, dtype='f'))*max_
            inds = np.digitize(r_ij, bins)
            vals, count = np.unique(inds, return_counts=True)

            # Convert bin indices back to physical distances
            vals = np.sqrt(max_)*(vals - 0.5)/div

            # Initialize the scattering intensity for this batch
            f_q = np.zeros(shape=shape, dtype='f')

            # Calculate intensity for each scattering vector q
            for k, q in enumerate(q_arr):
                qr = q*vals # q * r values
                f_q[k] += np.sum(count*np.sin(qr)/qr)   # Debye formula

            # Normalize and handle any zero values in the intensity
            f_q /= np.max(f_q)
            f_q[f_q <= 0] = np.min(f_q[f_q > 0])

            # Accumulate the batch scattering intensity
            f_total += f_q

        # Return the averaged scatterting intensity over all tries
        return f_total/tries


class Spheroid_Shell:
    # Initializes a Spheroid_Shell object with given geometric and material properties
    def __init__(
        self,
        R: float, 
        epsilon: float, 
        L: float, 
        p: float, 
        q: float, 
        f_core: float, 
        rho_delta: float=0.025, 
        rho: float=0.001
    ) -> None:
        
        self.R = R
        self.epsilon = epsilon
        self.L = L
        self.p = p
        self.q = q
        self.f_core = f_core
        self.rho_delta = rho_delta
        
        self.r_eff = (4/3)*((p + 1)/(p + 2))*R

        self.rho = rho
        self.coeffs = {0: 1, 1: rho_delta, 2: rho_delta, 3: rho_delta**2}

        # Calculate volume and estimate the number of scatterers
        V = np.pi*(4/3)*epsilon*R**3
        self.V = V
        self.num = int(V*rho)
        

    # Generates scatterers within the core of the spheroid
    def generate_core(self, n: int) -> np.ndarray:
        
        if n:
            R = self.R
            epsilon = self.epsilon
            p = self.p

            # Generate random scatterers uniformly distributed in the core
            y = np.random.uniform(-1.0 + 1e-6, 1.0 - 1e-6, n).astype('f')
            rho_arr = R*np.power(np.random.rand(n).astype('f'), 1/(p + 1))
            theta_arr = np.arccos(epsilon*y/np.sqrt(1 - np.square(y)*(1 - epsilon**2)))
            phi_arr = 2*np.pi*np.random.rand(n).astype('f')

            # Calculate the position of scatterers in 3D
            r_arr = epsilon*rho_arr/np.sqrt((epsilon**2 - 1)*np.square(np.sin(theta_arr)) + 1)
            x_arr = r_arr*np.sin(theta_arr)*np.cos(phi_arr)
            y_arr = r_arr*np.sin(theta_arr)*np.sin(phi_arr)
            z_arr = r_arr*np.cos(theta_arr)
            
            return np.hstack((x_arr.reshape(-1, 1), y_arr.reshape(-1, 1), z_arr.reshape(-1, 1)))
        
        else:
            return None

    # Generates scatterers within the shell of the spheroid
    def generate_shell(self, n: int) -> np.ndarray:
        if n:
            R = self.R
            epsilon = self.epsilon
            L = self.L
            q = self.q

            # Generate random distances following the shell density distribution
            F_rho = ((R + L)**(q + 1) - R**(q + 1))*np.random.rand(n).astype('f') + R**(q + 1)
            y = np.random.uniform(-1.0 + 1e-6, 1.0 - 1e-6, n).astype('f')
            rho_arr = np.power(F_rho, 1/(q + 1))
            theta_arr = np.arccos(epsilon*y/np.sqrt(1 - np.square(y)*(1 - epsilon**2)))
            phi_arr = 2*np.pi*np.random.rand(n).astype('f')

            # Calculate the position of scatterers in 3D
            r_arr = epsilon*rho_arr/np.sqrt((epsilon**2 - 1)*np.square(np.sin(theta_arr)) + 1)
            x_arr = r_arr*np.sin(theta_arr)*np.cos(phi_arr)
            y_arr = r_arr*np.sin(theta_arr)*np.sin(phi_arr)
            z_arr = r_arr*np.cos(theta_arr)
            
            return np.hstack((x_arr.reshape(-1, 1), y_arr.reshape(-1, 1), z_arr.reshape(-1, 1)))
        
        else:
            return None
    
    # Generate a set of scatterrers consisting of points in the core and shell of a structure
    def generate_scatterers(self, n: int) -> np.ndarray:
        
        f_core = self.f_core    # Fraction of scatterrers belonging to the core
        n1 = min(int(f_core*n), n - 1)  # Number of scatterers in the core
        n2 = n - n1 # Number of scatterers in the shell
        
        self.n1 = n1    # Save the number of core scatterers
        self.n2 = n2    # Save the number of shell scatterers

        # Generate core and shell scatterers
        core = self.generate_core(n=n1)
        shell = self.generate_shell(n=n2)

        # Combine core and shell scatterers into a single array
        return np.vstack((core, shell))
    
    # Perform Debye scattering simulation for a given set of wavevectors and scatterers
    def Debye_Scattering(
        self, 
        q_arr: np.ndarray, 
        pop: int=2048, 
        div: int=128
    ) -> np.ndarray:
        
        tries = ceil(ceil(self.num/pop)**2/2) # Number of batches for computation
        shape = q_arr.shape # Shape of the wavevector array
        
        f_final = np.zeros(shape=shape, dtype='f')  # Initialize total scattering intensity
        
        for i in range(tries):
            # Generate scatterers for the current batch
            scatterers = self.generate_scatterers(pop)
            n1 = self.n1    # Number of core scatterrers
            
            r_ij = np.zeros(shape=(pop, pop), dtype='f')    # Initialize pairwise distance matrix

            # Compute squared pairwise distances in 3D space
            for j in range(3):
                r = scatterers[:, j].reshape(-1, 1)
                r_ij += np.square(r - r.T)

            # Split distance matrix into different regions
            d_cc = r_ij[:n1, :n1]
            d_cs = r_ij[n1:, :n1]
            d_sc = r_ij[:n1, n1:]
            d_ss = r_ij[:n1, :n1]
            
            f_total = np.zeros(shape=shape, dtype='f')

            # Iterate over each region of the distance matrix
            for k, d_arr in enumerate([d_cc, d_cs, d_sc, d_ss]):
            
                d_arr = d_arr[d_arr > 0]    # Remove zero distances
                max_ = np.max(d_arr)    # Maximum distance in the current region

                coeff = self.coeffs[k]  # Coefficient for this region based on scattering density

                # Create bins for distance ranges
                bins = np.square(np.linspace(start=0, stop=1, num=div + 1, dtype='f'))*max_
                inds = np.digitize(d_arr, bins) # Bin indices for distances
                vals, count = np.unique(inds, return_counts=True)   # Unique bin values and counts

                # Compute midpoints of bins
                vals = np.sqrt(max_)*(vals - 0.5)/div

                f_q = np.zeros(shape=shape, dtype='f')  # Initialize intensity for this region

                # Compute scattering intensity for each wavevector
                for l, q in enumerate(q_arr):
                    qr = q*vals # Product of wavevector and distance
                    f_q[l] += np.sum(count*np.sin(qr)/qr)
                            
                f_q /= np.max(f_q)  # Normalize intensity
                f_q[f_q <= 0] = np.min(f_q[f_q > 0])
                
                f_total += coeff*f_q    # Add weighted intensity for the region
            
            f_final += f_total/np.max(f_total)  # Normalize and add to final intensity
                
        return f_final/tries    # Return average scattering intensity over all batches

# Class representing a cylindrical shell structure, including methods for generating scatterers
class Cylinder_Shell:
    
    def __init__(
        self, 
        R: float, 
        L: float, 
        t: float, 
        p: float, 
        q: float, 
        f_core: float,
        rho_delta: float=0.025, 
        rho: float=0.001
    ) -> None:
        # Initialize a Cylinder_Shell object with specified parameters
        self.R = R
        self.L = L
        self.t = t
        self.p = p
        self.q = q
        self.f_core = f_core
        self.rho_delta = rho_delta

        # Compute effective radius for the cylinder
        self.r_eff = (3/2)*((p + 1)/(p + 2))*R

        self.rho = rho  # Density of scatterers
        self.coeffs = {0: 1, 1: rho_delta, 2: rho_delta, 3: rho_delta**2}

        # Calculate volume and number of scatterers
        V = np.pi*(R**2)*L
        self.V = V
        self.num = int(V*rho)
    
    # Generate scatterers in the core of the cylinder
    def generate_core(self, n: int) -> np.ndarray:
        
        if n:
            R = self.R  # Radius of the cylinder
            L = self.L  # Length of the cylinder
            p = self.p  # Shape parameter for the core

            # Generate random radi based on shape parameter
            rho_arr = R*np.power(np.random.rand(n).astype('f'), 1/(1 + p))
            theta_arr = 2*np.pi*np.random.rand(n).astype('f')   # Random angular coordinates

            # Compute Cartesian coordinates for scatterers
            x_arr = rho_arr*np.cos(theta_arr)
            y_arr = rho_arr*np.sin(theta_arr)
            z_arr = L*np.random.rand(n).astype('f')
            
            return np.hstack((x_arr.reshape(-1, 1), y_arr.reshape(-1, 1), z_arr.reshape(-1, 1)))
        
        else:
            return None
    
    # Generate scatterer positions wihtin the shell region
    def generate_shell(self, n: int) -> np.ndarray:
        
        if n:
            # Initialize parameters for the shell
            R = self.R      # Inner radius of the shell
            L = self.L      # Length of the cylindrical shell
            t = self.t      # Thickness of the shell
            q = self.q      # Shape parameter

            # Generate radial distances using a power-law distribution
            F_rho = ((R + t)**(q + 1) - R**(q + 1))*np.random.rand(n).astype('f') + R**(q + 1)
            rho_arr = np.power(F_rho, 1/(1 + q))

            # Generate angular coordinates and Cartesian coordinates
            theta_arr = 2*np.pi*np.random.rand(n).astype('f')
            x_arr = rho_arr*np.cos(theta_arr)
            y_arr = rho_arr*np.sin(theta_arr)
            z_arr = L*np.random.rand(n).astype('f')

            # Return array of (x,y,z) coordinates
            return np.hstack((x_arr.reshape(-1, 1), y_arr.reshape(-1, 1), z_arr.reshape(-1, 1)))
        
        else:
            return None
    
    # Generate a mix of core and shell scatterers
    def generate_scatterers(self, n: int) -> np.ndarray:
        
        f_core = self.f_core    # Fraction of scatterers in the core
        n1 = min(int(f_core*n), n - 1)  # Number of core scatterers
        n2 = n - n1 # Number of shell scatterers
        
        self.n1 = n1
        self.n2 = n2

        # Generate core and shell scatterers
        core = self.generate_core(n=n1)
        shell = self.generate_shell(n=n2)

        # Combine core and shell scatterers into one array
        return np.vstack((core, shell))
    
    # Compute the Debye scattering intensity
    def Debye_Scattering(
        self, 
        q_arr: np.ndarray,  # Array of scattering vectors
        pop: int=2048,  # Number of scatterers per iteration
        div: int=128    # Number of bins for radial distances
    ) -> np.ndarray:
        # Calculate the number of iterations required
        tries = ceil(ceil(self.num/pop)**2/2)
        shape = q_arr.shape # Shape of the scattering vector array

        f_final = np.zeros(shape=shape, dtype='f')  # Final scattering intensity
        
        for i in range(tries):
            # Generate scatterers for this iteration
            scatterers = self.generate_scatterers(pop)
            n1 = self.n1

            # Compute pairwise distance between scatterers
            r_ij = np.zeros(shape=(pop, pop), dtype='f')
            for j in range(3):  # Iterate over x, y, z dimension
                r = scatterers[:, j].reshape(-1, 1)
                r_ij += np.square(r - r.T)

            # Divide distances into core-core, core-shell, etc.
            d_cc = r_ij[:n1, :n1]
            d_cs = r_ij[n1:, :n1]
            d_sc = r_ij[:n1, n1:]
            d_ss = r_ij[:n1, :n1]
            
            f_total = np.zeros(shape=shape, dtype='f')
            
            for k, d_arr in enumerate([d_cc, d_cs, d_sc, d_ss]):
                # Remove zero distances and find the maximum distance
                d_arr = d_arr[d_arr > 0]
                max_ = np.max(d_arr)
                
                coeff = self.coeffs[k]  # Weighting coefficient for this distance type

                # Bin disntances into equally spaced intervals
                bins = np.square(np.linspace(start=0, stop=1, num=div + 1, dtype='f'))*max_
                inds = np.digitize(d_arr, bins)
                vals, count = np.unique(inds, return_counts=True)

                # Compute bin center distances
                vals = np.sqrt(max_)*(vals - 0.5)/div
                
                f_q = np.zeros(shape=shape, dtype='f')
                        
                for l, q in enumerate(q_arr):
                    # Calculate scattering contributions for each q value
                    qr = q*vals
                    f_q[l] += np.sum(count*np.sin(qr)/qr)

                # Normalize and handle zero or negative values
                f_q /= np.max(f_q)
                f_q[f_q <= 0] = np.min(f_q[f_q > 0])

                # Add contribution to total intensity
                f_total += coeff*f_q

            # Add normalized total intensity to final result
            f_final += f_total/np.max(f_total)
        
        return f_final/tries
    
# Class representing a dispersed spheroid model
class Disperse_Spheroid:
    
    def __init__(
        self,
        R: float,   # Radius of the spheroid
        epsilon: float,     # Aspect ratio
        p: float,   # Shape parameter
        PDI: float,     # Polydispersity index
        rho: float=0.001,   # Density
        accuracy: int=10    # Accuracy of calculations
    ) -> None:
        
        self.R = R
        self.epsilon = epsilon
        self.p = p
        self.PDI = PDI
        self.accuracy = accuracy

        # Effective radius of the spheroid
        self.r_eff = (4/3)*((p + 1)/(p + 2))*R

        self.rho = rho

        # Volume and number of scatterers
        V = np.pi*(4/3)*epsilon*R**3
        self.V = V
        self.num = int(V*rho)
        
    # Generate scatterers based on spheroid properties
    def generate_scatterers(self, n: int) -> np.ndarray:
        s = Spheroid(
            R=self.R, 
            epsilon=self.epsilon, 
            p=self.p, 
            rho=self.rho
        )
        
        return s.generate_scatterers(n=n)
    
    # Compute the Debye scattering intensity for a spheroid model with polydispersity
    def Debye_Scattering(
        self, 
        q_arr: np.ndarray, 
        pop: int=2048, 
        div: int=128
    ):
        # Set accuracy for the polydispersity distribution
        division = self.accuracy

        # Generate cumulative probability for the polydispersity distribution
        probability = np.linspace(start=0, stop=1, num=division + 1, dtype='f')
        probability[0] = 0.001  # Avoid exact zero
        probability[-1] = 0.999 # Avoid exact one

        # Extract class parameters for the model
        R = self.R  # Base raidus
        epsilon = self.epsilon  # Eccentricity
        PDI = self.PDI  # Polydispersity index

        # Calculate the shape parameter for the polydispersity
        k = 1/PDI

        # Calculate particle size distribution values
        Zs = SZ_PPF(y=probability, k=k) # Get values of the polydispersity function
        Xs = SZ_avg(x_0=Zs[:-1], x_1=Zs[1:], k=k)   # Average size in each bin
        Xs = Xs.astype('f') # Convert to single-precision floats
        cXs = np.cbrt(Xs)   # Cube root to adjust scaling for radii

        # Initialize storage for intermediate scattering results
        storage = np.zeros((division, q_arr.size), dtype='f')

        # Initialize a list to store threads
        threads = []

        # Create threads of scattering computation per division
        for _ in range(division):
            threads.append(None)
        
        for i in range(division):
            # Slightly vary the exponent parameter to add variability
            p = self.p*np.random.uniform(0.9, 1.1)

            # Prepare parameters for the scattering calculation
            params = (R*cXs[i], epsilon, p)
            args = (q_arr, i, storage, params, pop, div)

            # Start a new thread for scattering calculation
            t_ = th.Thread(target=self.scattering, args=args)
            t_.start()
            threads[i] = t_

        # Wait for all threads to finish
        for t in threads:
            t.join()

        # Initialize array to store the final intensity
        I_q = np.zeros(shape=q_arr.shape, dtype='f')

        # Accumulate intensity for each division
        for i, I in enumerate(storage):
            I_q += Xs[i]*I*(probability[i + 1] - probability[i])

        # Normalize the final intensity
        return I_q/np.max(I_q)
    
    # Compute the scattering intensity for a given set of parameters and store the result
    def scattering(
        self, 
        q_arr: np.ndarray, 
        id_: int, 
        storage: np.ndarray, 
        params: tuple, 
        pop: int=2048, 
        div: int=128
    ) -> None:
        # Unpack the spheroid parameters
        R, epsilon, p = params
        rho = 0.001/sqrt(self.accuracy) # Adjust density for accuracy

        # Create a Spheroid object with the given parameters
        S_ = Spheroid(
            R=R, 
            epsilon=epsilon, 
            p=p, 
            rho=rho
        )

        # Perform Debye scattering computation and store the result
        storage[id_, :] += S_.Debye_Scattering(q_arr=q_arr, pop=pop, div=div)

        
# A class representing a model for dispersed cylinders with polydispersity
class Disperse_Cylinder:
    
    def __init__(
        self,
        R: float, 
        L: float, 
        p: float, 
        PDI: float, 
        rho: float=0.001, 
        accuracy: int=10
    ) -> None:
        # Initialize the cylinder model with given parameters
        self.R = R
        self.L = L
        self.p = p
        self.PDI = PDI
        self.accuracy = accuracy

        # Compute effective radius
        self.r_eff = (3/2)*((p + 1)/(p + 2))*R
        
        self.rho = rho

        # Compute the volume of the cylinder
        V = np.pi*(R**2)*L
        self.V = V
        self.num = int(V*rho)   # Estimate the number of scatterers
        
    # Generate scatterers for the cylinder model
    def generate_scatterers(self, n: int) -> np.ndarray:
        
        s = Cylinder(
            R=self.R, 
            L=self.L, 
            p=self.p, 
            rho=self.rho
        )
        
        return s.generate_scatterers(n=n)
    
    # Compute the Debye scattering intensity for a cylindrical model with polydispersity
    def Debye_Scattering(
        self, 
        q_arr: np.ndarray, 
        pop: int=2048, 
        div: int=128
    ):
        # Set accuracy for the polydispersity distribution
        division = self.accuracy

        # Generate cumulative probability values for polydispersity distribution
        probability = np.linspace(start=0, stop=1, num=division + 1, dtype='f')
        probability[0] = 0.001
        probability[-1] = 0.999

        # Extract model parameters
        R = self.R  # Base radius
        L = self.L  # Cylinder length
        PDI = self.PDI  # Polydispersity index

        # Calculate shape parameter for polydispersity
        k = 1/PDI

        # Compute polydispersity size distribution
        Zs = SZ_PPF(y=probability, k=k) # Get values of the polydispersity function
        Xs = SZ_avg(x_0=Zs[:-1], x_1=Zs[1:], k=k)   # Average size per bin
        Xs = Xs.astype('f') # Convert to single-precision float
        cXs = np.cbrt(Xs)   # Adjust scaling for radii

        # Initialize storage for intermediate scattering results
        storage = np.zeros((division, q_arr.size), dtype='f')

        # Create thread list for parallel processing
        threads = []
        for _ in range(division):
            threads.append(None)

        # Compute scattering for each division in parallel
        for i in range(division):
            # Slightly vary exponent parameter to introduce variability
            p = self.p*np.random.uniform(0.9, 1.1)

            # Define scattering parameters
            params = (R*cXs[i], L*cXs[i], p)
            args = (q_arr, i, storage, params, pop, div)

            # Start a new thread for computation
            t_ = th.Thread(target=self.scattering, args=args)
            t_.start()
            
            threads[i] = t_

        # Wait for all threads to complete execution
        for t in threads:
            t.join()

        # Initialize final intensity array
        I_q = np.zeros(shape=q_arr.shape, dtype='f')

        # Sum up contributions from each division
        for i, I in enumerate(storage):
            I_q += Xs[i]*I*(probability[i + 1] - probability[i])

        # Normalize intensity and return
        return I_q/np.max(I_q)
    
    
    def scattering(
        self, 
        q_arr: np.ndarray, 
        id_: int, 
        storage: np.ndarray, 
        params: tuple, 
        pop: int=2048, 
        div: int=128
    ) -> None:
        """
            Compute the scattering intensity for a given set of parameters and store the result.

            Parameters:
                q_arr (np.ndarray): Array of scattering wave vectors.
                id_ (int): ID of the current division for storage indexing.
                storage (np.ndarray): Array to store intermediate scattering results.
                params (tuple): Parameters for the cylinder (R, L, p).
                pop (int): Number of scatterers to generate per division (default: 2048).
                div (int): Number of bins to divide distances into (default: 128).
            """
        # Unpack cylinder parameters
        R, L, p = params
        rho = 0.001/sqrt(self.accuracy) # Adjust density for accuracy

        # Create a Cylinder object with given parameters
        S_ = Cylinder(
            R=R, 
            L=L, 
            p=p, 
            rho=rho
        )
        # Compute and store scattering results
        storage[id_, :] += S_.Debye_Scattering(q_arr=q_arr, pop=pop, div=div)

    

class Disperse_Spheroid_Shell:
    """
        A class representing a model for dispersed spheroidal shells with polydispersity.
        """
    def __init__(
        self,
        R: float, 
        epsilon: float, 
        PDI: float, 
        rho_delta: float, 
        f_core: float, 
        L: float, 
        p: float, 
        q: float, 
        rho: float=0.001, 
        accuracy: int=16
    ) -> None:
        """
               Initialize the spheroid shell model with given parameters.

               Parameters:
                   R (float): Base radius of the spheroid.
                   epsilon (float): Eccentricity of the spheroid.
                   PDI (float): Polydispersity index.
                   rho_delta (float): Density contrast.
                   f_core (float): Core fraction.
                   L (float): Length parameter.
                   p (float): Shape exponent.
                   q (float): Additional shape parameter.
                   rho (float): Density of scatterers (default: 0.001).
                   accuracy (int): Accuracy level for computations (default: 16).
               """
        self.R = R
        self.epsilon = epsilon
        self.L = L
        self.p = p
        self.q = q
        self.f_core = f_core
        self.PDI = PDI
        self.rho_delta = rho_delta
        self.accuracy = accuracy

        # Compute effective radius
        self.r_eff = (4/3)*((p + 1)/(p + 2))*R

        self.rho = rho

        # Compute volume of the spheroid
        V = np.pi*(4/3)*epsilon*R**3
        self.V = V
        self.num = int(V*rho)   # Estimate number of scatterers
        
    
    def generate_scatterers(self, n: int) -> np.ndarray:
        """
                Generate scatterers for the spheroid shell model.

                Parameters:
                    n (int): Number of scatterers to generate.

                Returns:
                    np.ndarray: Array of scatterer positions.
                """
        s = Spheroid_Shell(
            R=self.R, 
            epsilon=self.epsilon, 
            L=self.L, 
            p=self.p, 
            q=self.q, 
            f_core=self.f_core, 
            rho_delta=self.rho_delta, 
            rho=self.rho
        )
        
        return s.generate_scatterers(n=n)
    
    
    def Debye_Scattering(
        self, 
        q_arr: np.ndarray,  # Array of scattering wavevectors
        pop: int=2048,  # Population size for simulation
        div: int=128    # Number of divisions for numerical integration
    ):
        # Define the number of divisions based on accuracy parameter
        division = self.accuracy

        # Create a probability distribution for size polydispersity
        probability = np.linspace(start=0, stop=1, num=division + 1, dtype='f')
        probability[0] = 0.001  # Avoid zero probability
        probability[-1] = 0.999 # Avoid full probability

        # Retrieve key parameters from the object
        R = self.R  # Base radius
        epsilon = self.epsilon  # Aspect ratio factor
        PDI = self.PDI  # Polydispersity index

        # Compute shape factor for polydispersity
        k = 1/PDI

        # Generate size distribution based on polydispersity
        Zs = SZ_PPF(y=probability, k=k)
        Xs = SZ_avg(x_0=Zs[:-1], x_1=Zs[1:], k=k)
        Xs = Xs.astype('f') # Ensure floating point format
        cXs = np.cbrt(Xs)   # Cube root transformation for scaling

        # Initialize storage for computed scattering intensities
        storage = np.zeros((division, q_arr.size), dtype='f')

        # List to store threads for parallel execution
        threads = []
        for _ in range(division):
            threads.append(None)

        # Start scattering calculations for each size distribution
        for i in range(division):
            # Introduce small random variations to model variations
            L = self.L*np.random.uniform(0.9, 1.1)  # Length variation
            p = self.p*np.random.uniform(0.9, 1.1)  # Shape factor p variation
            q = self.q*np.random.uniform(0.9, 1.1)  # Shape factor q variation
            f_core = min(self.f_core*np.random.uniform(0.9, 1.1), 1.0 - 1e-6)

            # Prepare parameters for scattering computation
            params = (R*cXs[i], epsilon, L*cXs[i], p, q, f_core)
            args = (q_arr, i, storage, params, pop, div)

            # Create a thread to perform scattering calculation
            t_ = th.Thread(target=self.scattering, args=args)
            t_.start()

            # Store the thread reference
            threads[i] = t_

        # Wait for all threads to complete execution
        for t in threads:
            t.join()

        # Initialize the final scattering intensity array
        I_q = np.zeros(shape=q_arr.shape, dtype='f')

        # Aggregate results from each division
        for i, I in enumerate(storage):
            I_q += Xs[i]*I*(probability[i + 1] - probability[i])

        # Normalize intensity before returning
        return I_q/np.max(I_q)
    
    
    def scattering(
        self, 
        q_arr: np.ndarray,  # Scattering wavevector array
        id_: int,   # Index for storing results
        storage: np.ndarray,    # Storage array for scattering results
        params: tuple,  # Tuple of scattering parameters
        pop: int=2048,  # Population size for Monte Carlo calculations
        div: int=128    # Number of divisions for numerical integration
    ) -> None:

        # Unpack parameters for scattering calculation
        R, epsilon, L, p, q, f_core = params

        # Compute density variation based on accuracy
        rho = 0.001/sqrt(self.accuracy)

        # Create a spheroidal shell object for scattering calculations
        S_ = Spheroid_Shell(
            R=R, 
            epsilon=epsilon, 
            L=L, 
            p=p, 
            q=q, 
            f_core=f_core, 
            rho=rho
        )

        # Compute the Debye scattering intensity and store results
        storage[id_, :] += S_.Debye_Scattering(q_arr=q_arr, pop=pop, div=div)


# Class representing a polydisperse cylindrical shell model
class Disperse_Cylinder_Shell:
    
    def __init__(
        self,
        R: float, # Base radius of the cylinder
        L: float,   # Cylinder length
        PDI: float,     # Polydispersity index
        f_core: float,  # Core fraction
        rho_delta: float,   # Density contrast
        t: float,   # Shell thickness
        p: float,   # Shape parameter
        q: float,   # Shape parameter
        rho: float=0.001,   # Density parameter
        accuracy: int=16    # Number of divisions for numerical integration
    ) -> None:

        # Initialize object properties
        self.R = R
        self.L = L
        self.t = t
        self.p = p
        self.q = q
        self.f_core = f_core
        self.PDI = PDI
        self.rho_delta = rho_delta
        self.accuracy = accuracy

        # Compute effective radius based on shape factors
        self.r_eff = (3/2)*((p + 1)/(p + 2))*R

        # Store density parameter
        self.rho = rho

        # Compute total volume of the cylindrical shell
        V = np.pi*(R**2)*L
        self.V = V
        self.num = int(V*rho)   # Number of scatterers based on density
        
    # Generate scatterers for Monte Carlo simulation
    def generate_scatterers(self, n: int) -> np.ndarray:
        # Create an instance of the Cylinder_Shell class
        s = Cylinder_Shell(
            R=self.R, 
            L=self.L, 
            t=self.t, 
            p=self.p, 
            q=self.q, 
            f_core=self.f_core, 
            rho_delta=self.rho_delta, 
            rho=self.rho
        )

        # Generate and return a set of scatterers
        return s.generate_scatterers(n=n)
    
    
    def Debye_Scattering(
        self, 
        q_arr: np.ndarray,  # Input array of scattering vectors
        pop: int=2048,  # Number of scatterers
        div: int=128    # Number of divisions for polydispersity calculation
    ):
        """
            Computes the Debye scattering intensity for a polydisperse system using
            Monte Carlo sampling and multithreading.

            Parameters:
            - q_arr (np.ndarray): Array of scattering wave vectors.
            - pop (int): Population size for Monte Carlo simulations.
            - div (int): Number of subdivisions used in the polydispersity integration.

            Returns:
            - I_q (np.ndarray): Normalized scattering intensity.
            """

        # Number of subdivisions for polydispersity calculations
        division = self.accuracy

        # Generate a probability distribution from 0 to 1 for polydispersity
        probability = np.linspace(start=0, stop=1, num=division + 1, dtype='f')
        probability[0] = 0.001
        probability[-1] = 0.999

        # Extract key parameters from the object
        R = self.R
        L = self.L
        PDI = self.PDI  # Polydispersity index

        # Calculate shape factor for the Schulz-Zimm distribution
        k = 1/PDI

        # Compute size distribution using the Schulz-Zimm probability density function
        Zs = SZ_PPF(y=probability, k=k)
        Xs = SZ_avg(x_0=Zs[:-1], x_1=Zs[1:], k=k)
        Xs = Xs.astype('f')
        cXs = np.cbrt(Xs)   # Cube root of sizes for volume scaling

        # Storage array to hold the scattering results for each polydispersity bin
        storage = np.zeros(shape=(division, q_arr.size), dtype='f')

        # Initialize a list to hold threads
        threads = []
        for _ in range(division):
            threads.append(None)

        # Iterate over each polydispersity bin and start a thread for scattering computation
        for i in range(division):

            # Introduce small variations in parameters to account for experimental uncertainties
            t = self.t*np.random.uniform(0.9, 1.1)  # Shell thickness
            p = self.p*np.random.uniform(0.9, 1.1)  # Shape parameter p
            q = self.q*np.random.uniform(0.9, 1.1)  # Shape parameter q
            f_core = min(self.f_core*np.random.uniform(0.9, 1.1), 1.0 - 1e-6)   # Core fraction

            # Generate parameters for scattering calculation
            params = (R*cXs[i], L*cXs[i], t*cXs[i], p, q, f_core)
            args = (q_arr, i, storage, params, pop, div)

            # Start a new thread to compute scattering for this parameter set
            t_ = th.Thread(target=self.scattering, args=args)
            t_.start()
            threads[i] = t_ # Store the thread reference

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Initialize the final scattering intensity array
        I_q = np.zeros(shape=q_arr.shape, dtype='f')

        # Compute the weighted sum of scattering intensities across all distributions
        for i, I in enumerate(storage):
            I_q += Xs[i]*I*(probability[i + 1] - probability[i])

        # Normalize the final intensity to a maximum value of 1
        return I_q/np.max(I_q)
    
    
    def scattering(
        self, 
        q_arr: np.ndarray,  # Array of scattering wave vectors
        id_: int,   # Index of the division
        storage: np.ndarray,    # Storage array for intensity values
        params: tuple,  # Scattering parameters
        pop: int=2048,  # Popoulation size for Monte Carlo sampling
        div: int=128    # Number of divisions for integration
    ) -> None:
        """
            Computes the scattering intensity for a single polydispersity bin and stores it.

            Parameters:
            - q_arr (np.ndarray): Array of scattering wave vectors.
            - id_ (int): Index corresponding to the polydispersity bin.
            - storage (np.ndarray): Pre-allocated array to store the results.
            - params (tuple): Tuple of shape parameters (R, L, t, p, q, f_core).
            - pop (int): Population size for Monte Carlo calculations.
            - div (int): Number of subdivisions for numerical integration.
            """

        # Unpack the scattering parameters
        R, L, t, p, q, f_core = params

        # Define the particle density parameter
        rho = 0.001/sqrt(self.accuracy)

        # Create a Cylinder_Shell object with the given parameters
        S_ = Cylinder_Shell(
            R=R, 
            L=L, 
            t=t, 
            p=p, 
            q=q, 
            f_core=f_core, 
            rho=rho
        )

        # Compute and store the scattering intensity
        storage[id_, :] += S_.Debye_Scattering(q_arr=q_arr, pop=pop, div=div)


def test_shapes() -> None:
    """"
    Generates and visualizes different 3D shapes using randomly generated parameters. It also
    plots their respective scattering functions.
    """
    # Define shape parameters with random variations
    R = np.power(2, np.random.triangular(6, 7, 8))  # Generate base radius
    R = 100 # Override radius with a fixed value
    epsilon = np.random.uniform(low=0.8, high=1.2)  # Shape elongation factor
    epsilon = 1.2   # Override epsilon with a fixed value
    p = np.random.uniform(low=1.5, high=2.0)    # Power factor for shape
    p = 2.0 # Override p with a fixed value
    q = 1.0 # Secondary shape parameter
    L = R*np.random.uniform(low=5.0, high=7.0)  # Compute length
    
    print((R, epsilon, p, q, L))    # Print selected parameters

    # Additional shape parameters
    L_ = 2*R    # Adjusted length
    t = R   # Thickness
    f_core = 0.75   # Core fraction
    q = 0   # Reset q
    num = 1000  # Number of scatterers to generate

    # Create shape objects
    s = Spheroid(R=R, epsilon=epsilon, p=p)
    c = Cylinder(R=R, L=L, p=p - 1)
    cs = Spheroid_Shell(R=R, epsilon=epsilon, L=L_, f_core=f_core, p=p, q=q)
    cc = Cylinder_Shell(R=R, L=L, t=t, f_core=f_core, p=p - 1, q=q)

    # Generate scatterer distributions for each shape
    scatterers_s = s.generate_scatterers(num)
    scatterers_c = c.generate_scatterers(num)
    scatterers_cs = cs.generate_scatterers(num)
    scatterers_cc = cc.generate_scatterers(num)

    # Extract x,y,z coordinates for plotting
    xs_s, ys_s, zs_s = scatterers_s[:, 0], scatterers_s[:, 1], scatterers_s[:, 2]
    xs_c, ys_c, zs_c = scatterers_c[:, 0], scatterers_c[:, 1], scatterers_c[:, 2]
    xs_cs, ys_cs, zs_cs = scatterers_cs[:, 0], scatterers_cs[:, 1], scatterers_cs[:, 2]
    xs_cc, ys_cc, zs_cc = scatterers_cc[:, 0], scatterers_cc[:, 1], scatterers_cc[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    ax.scatter(xs_s, ys_s, zs_s)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    ax.scatter(xs_c, ys_c, zs_c)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    ax.scatter(xs_cs, ys_cs, zs_cs)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()
    
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    ax.scatter(xs_cc, ys_cc, zs_cc)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()

    # Compute scattering intensity over a logarithmic range of q values
    q_log_arr = np.arange(-2, 0, np.true_divide(1, 128))
    q_arr = np.power(10, q_log_arr - 2*np.log10(2))
    
    f_q_s = s.Debye_Scattering(q_arr=q_arr)
    f_q_c = c.Debye_Scattering(q_arr=q_arr)
    f_q_cs = cs.Debye_Scattering(q_arr=q_arr)
    f_q_cc = cc.Debye_Scattering(q_arr=q_arr)

    # Plot scattering intensity for all shapes
    plt.figure()
    plt.plot(q_arr, f_q_s, label='Spheroid')
    plt.plot(q_arr, f_q_c, label='Cylinder')
    plt.plot(q_arr, f_q_cs, label='Core Shell (Spheroid)')
    plt.plot(q_arr, f_q_cc, label='Core Shell (Cylinder)')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()


def test_single() -> None:
    """
        Generates a single instance of a Disperse Spheroid Shell object with random
        parameters and plots its scattering intensity.
        """
    # Define scattering vector range
    q_log_arr = np.arange(-2, 0, np.true_divide(1, 128))
    q_arr = np.power(10, q_log_arr - 2*np.log10(2))

    # Generate random shape parameters
    R = np.power(2, np.random.triangular(left=6.0, mode=7.0, right=8.0))
    epsilon = np.power(2, np.random.triangular(left=-1.0, mode=0.0, right=1.0))
    PDI = np.power(10, np.random.uniform(np.log10(0.001), np.log10(0.5)))
    L = R*np.random.uniform(1.0, 2.0)
    p = np.random.uniform(1.75, 2.0)
    q = np.random.uniform(0.0, 0.25)
    f_core = np.random.triangular(0.5, 0.75, 1.0)

    # Create a Disperse Spheroid Shel object
    s = Disperse_Spheroid_Shell(
        R=R, 
        epsilon=epsilon, 
        L=L, 
        p=p, 
        q=q, 
        f_core=f_core, 
        PDI=PDI
    )

    # Print parameters
    print(f'Radius: {R:.3f}')
    print(f'Aspect ratio: {epsilon:.3f}')
    print(f'PDI: {PDI:.3f}')
    print(f'Shell strength: {f_core:.3f}')

    # Compute and plot scattering intensity
    I_arr = s.Debye_Scattering(q_arr=q_arr)
    plt.figure()
    plt.plot(q_arr, I_arr)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid()
    plt.show()
    

def main(*args, **kwargs) -> int:
    """
       Main function to execute the test_single() function.
    """
    test_single()
    
    return 0


if __name__ == '__main__':
    main()
