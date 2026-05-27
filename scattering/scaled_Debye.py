import numpy as np
import matplotlib.pyplot as plt
import threading as th

# import Debye as db

from time import perf_counter

from .Schulz_Zimm import SZ_avg, SZ_PPF


PI = np.pi

plt.rc('axes', titlesize=20)
plt.rc('axes', labelsize=16)
plt.rc('legend', fontsize=12)


def radius_of_gyration(shape: str, R: np.ndarray, epsilon: np.ndarray) -> np.ndarray:
    
    if shape == 'spheroid':
        return np.sqrt((2 + np.square(epsilon))*np.square(R)/5)
    elif shape == 'cylinder':
        return np.sqrt(np.square(R)/2 + np.square(R*epsilon)/12)
    else:
        raise ValueError('Invalid shape. Expected "spheroid" or "cylinder".')


class Spheroid:
    
    def __init__(
        self, 
        R:       float, 
        epsilon: float, 
        p:       float
    ) -> None:
        
        self.R       = R
        self.epsilon = epsilon
        self.p       = p
        
        self.class_ = 'spheroid'
        
        self.R_eff = (4/3)*((p + 1)/(p + 2))*R
        self.R_es  = np.cbrt(epsilon)*R
        self.R_g   = radius_of_gyration(shape=self.class_, R=R, epsilon=epsilon)
        self.V     = PI*(4/3)*epsilon*R**3
        self.d_max = (2*epsilon)**2
        
        return None
    
    
    def generate_scatterers(self, n: int) -> np.ndarray:
        
        epsilon = self.epsilon
        p       = self.p
                
        y = np.random.uniform(-1.0 + 1e-6, 1.0 - 1e-6, n).astype('f')
                
        rho_arr   = np.power(np.random.rand(n).astype('f'), 1/(p + 1))
        theta_arr = np.arccos(epsilon*y/np.sqrt(1 - np.square(y)*(1 - epsilon**2)))
        phi_arr   = 2*PI*np.random.rand(n).astype('f')
        
        r_arr = epsilon*rho_arr/np.sqrt((epsilon**2 - 1)*np.square(np.sin(theta_arr)) + 1)
                
        x_arr = r_arr*np.sin(theta_arr)*np.cos(phi_arr)
        y_arr = r_arr*np.sin(theta_arr)*np.sin(phi_arr)
        z_arr = r_arr*np.cos(theta_arr)
        
        x_arr = x_arr - np.mean(x_arr)
        y_arr = y_arr - np.mean(y_arr)
        z_arr = z_arr - np.mean(z_arr)
        
        return np.hstack((x_arr.reshape(-1, 1), y_arr.reshape(-1, 1), z_arr.reshape(-1, 1)))
    
    
    def Debye_scattering(
        self, 
        q_arr: np.ndarray, 
        pop:   int=2048, 
        div:   int=256, 
        iter_: int=16
    ) -> np.ndarray:
                
        max_   = self.d_max
        bins   = np.square(np.linspace(start=0, stop=1, num=div + 1, dtype='f'))*max_
        vals   = np.sqrt(max_)*(np.arange(start=1, stop=div + 1) - 0.5)/div
        counts = np.zeros(shape=(div, ))
                
        for i in range(iter_):
            
            scatterers = self.generate_scatterers(pop)
            
            r_ij = np.zeros(shape=(pop, pop), dtype='f')
            
            for j in range(3):
                r = scatterers[:, j].reshape(-1, 1)
                r_ij += np.square(r - r.T)

            inds = np.digitize(r_ij, bins)
            temp_vals, temp_count = np.unique(inds, return_counts=True)
            
            counts[temp_vals - 1] += temp_count
                        
        qr    = q_arr[:, np.newaxis]*self.R*vals[np.newaxis, :]
        I_arr = np.sum(counts*np.sinc(qr/PI), axis=1)
    
        I_arr /= np.max(I_arr)
        I_arr[I_arr <= 0] = np.min(I_arr[I_arr > 0])
                        
        return I_arr
    
    
    def distance_distribution(
        self, 
        probes:    int=4096, 
        div:       int=256, 
        normalize: bool=False
    ) -> tuple[np.ndarray, np.ndarray]:
        
        max_   = self.d_max
        bins   = np.square(np.linspace(start=0, stop=1, num=div + 1, dtype='f'))*max_
        vals   = np.sqrt(max_)*(np.arange(start=1, stop=div + 1) - 0.5)/div
        counts = np.zeros(shape=(div, ))
                            
        scatterers = self.generate_scatterers(probes)
        
        r_ij = np.zeros(shape=(probes, probes), dtype='f')
        
        for j in range(3):
            r = scatterers[:, j].reshape(-1, 1)
            r_ij += np.square(r - r.T)

        inds = np.digitize(r_ij, bins)
        temp_vals, temp_count = np.unique(inds, return_counts=True)
        
        counts[temp_vals - 1] += temp_count
        
        if normalize:
            return vals, counts/np.sum(counts)
        else:
            return self.R*vals, counts/np.sum(counts)
    

class Cylinder:
    
    def __init__(
        self, 
        R:       float, 
        epsilon: float, 
        p:       float
    ) -> None:
        
        self.R       = R
        self.epsilon = epsilon
        self.p       = p
        
        self.class_ = 'cylinder'
        
        self.R_eff = (3/2)*((p + 1)/(p + 2))*R
        self.R_es  = np.cbrt(3*epsilon/2)*R
        self.R_g   = radius_of_gyration(shape=self.class_, R=R, epsilon=epsilon)
        self.V     = PI*epsilon*R**3
        self.d_max = (2*epsilon)**2 + 1
        
        return None
    
    
    def generate_scatterers(self, n: int) -> np.ndarray:
        
        R       = self.R
        epsilon = self.epsilon
        p       = self.p
                
        rho_arr   = np.power(np.random.rand(n).astype('f'), 1/(1 + p))
        theta_arr = 2*PI*np.random.rand(n).astype('f')
        
        x_arr = rho_arr*np.cos(theta_arr)
        y_arr = rho_arr*np.sin(theta_arr)
        z_arr = epsilon*np.random.rand(n).astype('f')
        
        return np.hstack((x_arr.reshape(-1, 1), y_arr.reshape(-1, 1), z_arr.reshape(-1, 1)))
    
    
    def Debye_scattering(
        self, 
        q_arr: np.ndarray, 
        pop:   int=2048, 
        div:   int=256, 
        iter_: int=16
    ) -> np.ndarray:
        
        max_   = self.d_max
        bins   = np.square(np.linspace(start=0, stop=1, num=div + 1, dtype='f'))*max_
        vals   = np.sqrt(max_)*(np.arange(start=1, stop=div + 1) - 0.5)/div
        counts = np.zeros(shape=(div, ))

        for i in range(iter_):
            
            scatterers = self.generate_scatterers(pop)
            
            r_ij = np.zeros(shape=(pop, pop), dtype='f')
            
            for j in range(3):
                r = scatterers[:, j].reshape(-1, 1)
                r_ij += np.square(r - r.T)

            inds = np.digitize(r_ij, bins)
            temp_vals, temp_count = np.unique(inds, return_counts=True)
            
            counts[temp_vals - 1] += temp_count
                        
        qr    = q_arr[:, np.newaxis]*self.R*vals[np.newaxis, :]
        I_arr = np.sum(counts*np.sinc(qr/PI), axis=1)
    
        I_arr /= np.max(I_arr)
        I_arr[I_arr <= 0] = np.min(I_arr[I_arr > 0])
                        
        return I_arr
    
    
    def distance_distribution(
        self, 
        probes:    int=4096, 
        div:       int=256, 
        normalize: bool=False
    ) -> tuple[np.ndarray, np.ndarray]:
        
        max_   = self.d_max
        bins   = np.square(np.linspace(start=0, stop=1, num=div + 1, dtype='f'))*max_
        vals   = np.sqrt(max_)*(np.arange(start=1, stop=div + 1) - 0.5)/div
        counts = np.zeros(shape=(div, ))
                            
        scatterers = self.generate_scatterers(probes)
        
        r_ij = np.zeros(shape=(probes, probes), dtype='f')
        
        for j in range(3):
            r = scatterers[:, j].reshape(-1, 1)
            r_ij += np.square(r - r.T)

        inds = np.digitize(r_ij, bins)
        temp_vals, temp_count = np.unique(inds, return_counts=True)
        
        counts[temp_vals - 1] += temp_count
        
        if normalize:
            return vals, counts/np.sum(counts)
        else:
            return self.R*vals, counts/np.sum(counts)


class Spheroid_Shell:
    
    def __init__(
        self,
        R:         float, 
        epsilon:   float, 
        f_core:    float, 
        rho_delta: float,
        t:         float, 
        p:         float, 
        q:         float, 
        lt:        float=0, 
        ls:        float=0,
        # esld_core: float=1,
    ) -> None:
        
        """
        Parameters:
            - R:         Core radius of the spheroid.
            - epsilon:   Aspect ratio of the spheroid (length/width).
            - f_core:    Fraction of scatterers in the core.
            - rho_delta: Scattering length density contrast between the shell and the core.
            - t:         Thickness of the shell.
            - p:         Exponent for the radial distribution of scatterers in the core.
            - q:         Exponent for the radial distribution of scatterers in the shell.
            - lt:        Location of the bump in the distance distribution (in units of R).
            - ls:        Strength of the bump in the distance distribution (relative to the total number of scatterers).
            # - esld_core: Excess scattering length density of the core (default is 1).
        """


        self.R         = R
        self.epsilon   = epsilon
        self.f_core    = f_core
        self.rho_delta = rho_delta
        self.t         = t
        self.p         = p
        self.q         = q
        self.lt        = lt
        self.ls        = ls
        # self.esld_core = esld_core

        self.class_ = 'spheroid'
        
        self.R_eff  = (4/3)*((p + 1)/(p + 2))*R
        self.R_es   = np.cbrt(epsilon)*R
        self.R_g    = radius_of_gyration(shape=self.class_, R=R, epsilon=epsilon)
        self.coeffs = {0: 1, 1: rho_delta, 2: rho_delta, 3: rho_delta**2}

        # self.coeffs[0] *= esld_core**2
        # self.coeffs[1] *= esld_core**2
        # self.coeffs[2] *= esld_core**2
        # self.coeffs[3] *= esld_core**2

        self.V      = PI*(4/3)*epsilon*R**3
        
        f_corona      = t/R
        self.f_corona = f_corona
        
        if epsilon >= 1.0:
            self.maxes = np.array((
                (2*epsilon)**2, 
                (epsilon*(2 + f_corona))**2, 
                (epsilon*(2 + f_corona))**2,
                (epsilon*(2 + 2*f_corona))**2,
            ))
        else:
            self.maxes = np.array((
                2**2, 
                (2 + f_corona)**2, 
                (2 + f_corona)**2,
                (2 + 2*f_corona)**2,
            ))
        
        return None
    
    
    def generate_points(self, n: int, in_core: bool) -> np.ndarray:
        
        if n:
            
            if in_core:
                r_i = 0.0
                r_o = 1.0
                epsilon = self.epsilon
                p = self.p
            else:
                r_i = 1.0
                r_o = 1 + self.f_corona
                epsilon = self.epsilon
                p = self.q
                
            F_rho = (r_o**(p + 1) - r_i**(p + 1))*np.random.rand(n).astype('f') + r_i**(p + 1)
            y     = np.random.uniform(-1.0 + 1e-6, 1.0 - 1e-6, n).astype('f')
                    
            rho_arr   = np.power(F_rho, 1/(p + 1))
            theta_arr = np.arccos(epsilon*y/np.sqrt(1 - np.square(y)*(1 - epsilon**2)))
            phi_arr   = 2*PI*np.random.rand(n).astype('f')
            
            r_arr = epsilon*rho_arr/np.sqrt((epsilon**2 - 1)*np.square(np.sin(theta_arr)) + 1)
            
            x_arr = r_arr*np.sin(theta_arr)*np.cos(phi_arr)
            y_arr = r_arr*np.sin(theta_arr)*np.sin(phi_arr)
            z_arr = r_arr*np.cos(theta_arr)
            
            return np.hstack((x_arr.reshape(-1, 1), y_arr.reshape(-1, 1), z_arr.reshape(-1, 1)))
        
        else:
            raise ValueError('Number of points must be greater than 0.')
    
    
    def generate_scatterers(self, n: int) -> tuple[np.ndarray, tuple[int, int]]:
        
        f_core = self.f_core
        n1     = min(int(f_core*n), n - 1)
        n2     = n - n1
        
        core  = self.generate_points(n=n1, in_core=True)
        shell = self.generate_points(n=n2, in_core=False)
        
        return np.vstack((core, shell)), (n1, n2)
    
    
    def Debye_scattering(
        self, 
        q_arr: np.ndarray, 
        pop:   int=2048, 
        div:   int=256, 
        iter_: int=16
    ) -> np.ndarray:
                
        bins_arr = [
            np.square(np.linspace(start=0, stop=1, num=div + 1, dtype='f'))*self.maxes[0], 
            np.square(np.linspace(start=0, stop=1, num=div + 1, dtype='f'))*self.maxes[1], 
            np.square(np.linspace(start=0, stop=1, num=div + 1, dtype='f'))*self.maxes[2], 
            np.square(np.linspace(start=0, stop=1, num=div + 1, dtype='f'))*self.maxes[3]
        ]
        vals_arr = [
            np.sqrt(self.maxes[0])*(np.arange(start=1, stop=div + 1) - 0.5)/div, 
            np.sqrt(self.maxes[1])*(np.arange(start=1, stop=div + 1) - 0.5)/div, 
            np.sqrt(self.maxes[2])*(np.arange(start=1, stop=div + 1) - 0.5)/div, 
            np.sqrt(self.maxes[3])*(np.arange(start=1, stop=div + 1) - 0.5)/div, 
        ]
        counts_arr = [
            np.zeros(shape=(div, )), 
            np.zeros(shape=(div, )), 
            np.zeros(shape=(div, )), 
            np.zeros(shape=(div, ))
        ]
        
        for i in range(iter_):
            
            scatterers, (n1, n2) = self.generate_scatterers(pop)
            
            r_ij = np.zeros(shape=(pop, pop), dtype='f')
            
            for j in range(3):
                r = scatterers[:, j].reshape(-1, 1)
                r_ij += np.square(r - r.T)
            
            d_cc = r_ij[:n1, :n1]
            d_cs = r_ij[n1:, :n1]
            d_sc = r_ij[:n1, n1:]
            d_ss = r_ij[n1:, n1:]

            for k, d_arr in enumerate([d_cc, d_cs, d_sc, d_ss]):
                            
                inds = np.digitize(d_arr[d_arr > 0], bins_arr[k])
                temp_vals, temp_count = np.unique(inds, return_counts=True)
                
                counts_arr[k][temp_vals - 1] += temp_count
                
        if self.lt and self.ls:
            
            loc   = self.lt/self.R
            scale = loc/4
            size  = int(0.0005*self.ls*np.sum(counts_arr[0]))
            
            bump = np.random.normal(loc=loc, scale=scale, size=size)
            bump[bump <= 0] = np.min(bump[bump > 0])
            
            inds = np.digitize(np.square(bump), bins_arr[0])
            temp_vals, temp_count = np.unique(inds, return_counts=True)
            counts_arr[0][temp_vals - 1] += temp_count
        
        qr_0    = q_arr[:, np.newaxis]*self.R*vals_arr[0][np.newaxis, :]
        I_arr_0 = self.coeffs[0]*np.sum(counts_arr[0]*np.sinc(qr_0/PI), axis=1)
        qr_1    = q_arr[:, np.newaxis]*self.R*vals_arr[1][np.newaxis, :]
        I_arr_1 = self.coeffs[1]*np.sum(counts_arr[1]*np.sinc(qr_1/PI), axis=1)
        qr_2    = q_arr[:, np.newaxis]*self.R*vals_arr[2][np.newaxis, :]
        I_arr_2 = self.coeffs[2]*np.sum(counts_arr[2]*np.sinc(qr_2/PI), axis=1)
        qr_3    = q_arr[:, np.newaxis]*self.R*vals_arr[3][np.newaxis, :]
        I_arr_3 = self.coeffs[3]*np.sum(counts_arr[3]*np.sinc(qr_3/PI), axis=1)
        
        I_arr = I_arr_0 + I_arr_1 + I_arr_2 + I_arr_3
        
        I_arr /= np.max(I_arr)
        I_arr[I_arr <= 0] = np.min(I_arr[I_arr > 0])
                
        return I_arr
    
    
    def distance_distribution(
        self, 
        probes:    int=4096, 
        div:       int=256, 
        normalize: bool=False
    ) -> tuple[np.ndarray, np.ndarray]:
        
        max_   = self.maxes[-1]
        bins   = np.square(np.linspace(start=0, stop=1, num=div + 1, dtype='f'))*max_
        vals   = np.sqrt(max_)*(np.arange(start=1, stop=div + 1) - 0.5)/div
        counts = np.zeros(shape=(div, ))
                            
        scatterers, _ = self.generate_scatterers(probes)
        
        r_ij = np.zeros(shape=(probes, probes), dtype='f')
        
        for j in range(3):
            r = scatterers[:, j].reshape(-1, 1)
            r_ij += np.square(r - r.T)

        inds = np.digitize(r_ij, bins)
        temp_vals, temp_count = np.unique(inds, return_counts=True)
        
        counts[temp_vals - 1] += temp_count
        
        if self.lt and self.ls:
                        
            loc   = self.lt/self.R
            scale = loc/4
            size  = int(0.001*self.ls*np.sum(counts))
            
            bump = np.random.normal(loc=loc, scale=scale, size=size)
            bump[bump <= 0] = np.min(bump[bump > 0])
            
            inds = np.digitize(np.square(bump), bins)
            temp_vals, temp_count = np.unique(inds, return_counts=True)
            counts[temp_vals - 1] += temp_count
        
        if normalize:
            return vals, counts/np.sum(counts)
        else:
            return self.R*vals, counts/np.sum(counts)


class Cylinder_Shell:
    
    def __init__(
        self, 
        R:         float, 
        epsilon:   float, 
        f_core:    float,
        rho_delta: float, 
        t:         float, 
        p:         float, 
        q:         float, 
        lt:        float=0, 
        ls:        float=0
    ) -> None:
        
        self.R         = R
        self.epsilon   = epsilon
        self.f_core    = f_core
        self.rho_delta = rho_delta
        self.t         = t
        self.p         = p
        self.q         = q
        self.lt        = lt
        self.ls        = ls
        
        self.class_ = 'cylinder'

        self.R_eff  = (3/2)*((p + 1)/(p + 2))*R
        self.R_es   = np.cbrt(3*epsilon/2)*R
        self.R_g    = radius_of_gyration(shape=self.class_, R=R, epsilon=epsilon)
        self.coeffs = {0: 1, 1: rho_delta, 2: rho_delta, 3: rho_delta**2}        
        self.V      = PI*epsilon*R**3
        
        f_corona      = t/R
        self.f_corona = f_corona
        
        self.maxes = np.array((
            (2*epsilon)**2 + 2**2, 
            (2*epsilon)**2 + (2 + f_corona)**2, 
            (2*epsilon)**2 + (2 + f_corona)**2,
            (2*epsilon)**2 + (2 + 2*f_corona)**2,
        ))
        
        return None
        
    
    def generate_points(self, n: int, in_core: bool) -> np.ndarray:
        
        if n:

            epsilon = self.epsilon
            
            if in_core:
                r_i     = 0.0
                r_o     = 1.0
                p       = self.p
            else:
                r_i     = 1.0
                r_o     = 1 + self.f_corona
                p       = self.q
                
            F_rho   = (r_o**(p + 1) - r_i**(p + 1))*np.random.rand(n).astype('f') + r_i**(p + 1)                    
            rho_arr = np.power(F_rho, 1/(p + 1))
            
            theta_arr = 2*PI*np.random.rand(n).astype('f')            
            
            x_arr = rho_arr*np.cos(theta_arr)
            y_arr = rho_arr*np.sin(theta_arr)
            z_arr = epsilon*np.random.rand(n).astype('f')
            
            return np.hstack((x_arr.reshape(-1, 1), y_arr.reshape(-1, 1), z_arr.reshape(-1, 1)))
        
        else:
            raise ValueError('Number of points must be greater than 0.')
    
    
    def generate_scatterers(self, n: int) -> tuple[np.ndarray, tuple[int, int]]:
        
        f_core = self.f_core
        n1     = min(int(f_core*n), n - 1)
        n2     = n - n1
        
        core  = self.generate_points(n=n1, in_core=True)
        shell = self.generate_points(n=n2, in_core=False)
        
        return np.vstack((core, shell)), (n1, n2)
    
    
    def Debye_scattering(
        self, 
        q_arr: np.ndarray, 
        pop:   int=2048, 
        div:   int=256, 
        iter_: int=16
    ) -> np.ndarray:
                
        bins_arr = [
            np.square(np.linspace(start=0, stop=1, num=div + 1, dtype='f'))*self.maxes[0], 
            np.square(np.linspace(start=0, stop=1, num=div + 1, dtype='f'))*self.maxes[1], 
            np.square(np.linspace(start=0, stop=1, num=div + 1, dtype='f'))*self.maxes[2], 
            np.square(np.linspace(start=0, stop=1, num=div + 1, dtype='f'))*self.maxes[3]
        ]
        vals_arr = [
            np.sqrt(self.maxes[0])*(np.arange(start=1, stop=div + 1) - 0.5)/div, 
            np.sqrt(self.maxes[1])*(np.arange(start=1, stop=div + 1) - 0.5)/div, 
            np.sqrt(self.maxes[2])*(np.arange(start=1, stop=div + 1) - 0.5)/div, 
            np.sqrt(self.maxes[3])*(np.arange(start=1, stop=div + 1) - 0.5)/div, 
        ]
        counts_arr = [
            np.zeros(shape=(div, )), 
            np.zeros(shape=(div, )), 
            np.zeros(shape=(div, )), 
            np.zeros(shape=(div, ))
        ]
        
        for i in range(iter_):
            
            scatterers, (n1, _) = self.generate_scatterers(pop)
            
            r_ij = np.zeros(shape=(pop, pop), dtype='f')
            
            for j in range(3):
                r = scatterers[:, j].reshape(-1, 1)
                r_ij += np.square(r - r.T)
            
            d_cc = r_ij[:n1, :n1]
            d_cs = r_ij[n1:, :n1]
            d_sc = r_ij[:n1, n1:]
            d_ss = r_ij[n1:, n1:]

            for k, d_arr in enumerate([d_cc, d_cs, d_sc, d_ss]):
                            
                inds = np.digitize(d_arr[d_arr > 0], bins_arr[k])
                temp_vals, temp_count = np.unique(inds, return_counts=True)
                
                idx = temp_vals - 1
                idx = np.clip(idx, 0, counts_arr[k].shape[0] - 1)
                counts_arr[k][idx] += temp_count
        
        if self.lt and self.ls:
            
            loc   = self.lt/self.R
            scale = loc/4
            size  = int(0.0005*self.ls*np.sum(counts_arr[0]))
            
            bump = np.random.normal(loc=loc, scale=scale, size=size)
            bump[bump <= 0] = np.min(bump[bump > 0])
            
            inds = np.digitize(np.square(bump), bins_arr[0])
            temp_vals, temp_count = np.unique(inds, return_counts=True)
            idx = temp_vals - 1
            idx = np.clip(idx, 0, counts_arr[0].shape[0] - 1)
            counts_arr[0][idx] += temp_count
        
        qr_0    = q_arr[:, np.newaxis]*self.R*vals_arr[0][np.newaxis, :]
        I_arr_0 = self.coeffs[0]*np.sum(counts_arr[0]*np.sinc(qr_0/PI), axis=1)
        qr_1    = q_arr[:, np.newaxis]*self.R*vals_arr[1][np.newaxis, :]
        I_arr_1 = self.coeffs[1]*np.sum(counts_arr[1]*np.sinc(qr_1/PI), axis=1)
        qr_2    = q_arr[:, np.newaxis]*self.R*vals_arr[2][np.newaxis, :]
        I_arr_2 = self.coeffs[2]*np.sum(counts_arr[2]*np.sinc(qr_2/PI), axis=1)
        qr_3    = q_arr[:, np.newaxis]*self.R*vals_arr[3][np.newaxis, :]
        I_arr_3 = self.coeffs[3]*np.sum(counts_arr[3]*np.sinc(qr_3/PI), axis=1)
        
        I_arr = I_arr_0 + I_arr_1 + I_arr_2 + I_arr_3
        
        I_arr /= np.max(I_arr)
        I_arr[I_arr <= 0] = np.min(I_arr[I_arr > 0])
                
        return I_arr
    
    
    def distance_distribution(
        self, 
        probes:    int=4096, 
        div:       int=256, 
        normalize: bool=False
    ) -> tuple[np.ndarray, np.ndarray]:
        
        max_   = self.maxes[-1]
        bins   = np.square(np.linspace(start=0, stop=1, num=div + 1, dtype='f'))*max_
        vals   = np.sqrt(max_)*(np.arange(start=1, stop=div + 1) - 0.5)/div
        counts = np.zeros(shape=(div, ))
                            
        scatterers, _ = self.generate_scatterers(probes)
        
        r_ij = np.zeros(shape=(probes, probes), dtype='f')
        
        for j in range(3):
            r = scatterers[:, j].reshape(-1, 1)
            r_ij += np.square(r - r.T)

        inds = np.digitize(r_ij, bins)
        temp_vals, temp_count = np.unique(inds, return_counts=True)
        idx = temp_vals - 1
        idx = np.clip(idx, 0, counts.shape[0] - 1)
        counts[idx] += temp_count
        
        if self.lt and self.ls:
                        
            loc   = self.lt/self.R
            scale = loc/4
            size  = int(0.001*self.ls*np.sum(counts))
            
            bump = np.random.normal(loc=loc, scale=scale, size=size)
            bump[bump <= 0] = np.min(bump[bump > 0])
            
            inds = np.digitize(np.square(bump), bins)
            temp_vals, temp_count = np.unique(inds, return_counts=True)
            idx = temp_vals - 1
            idx = np.clip(idx, 0, counts.shape[0] - 1)
            counts[idx] += temp_count
        
        if normalize:
            return vals, counts/np.sum(counts)
        else:
            return self.R*vals, counts/np.sum(counts)
    
    
class Disperse_Spheroid:
    
    def __init__(
        self,
        R:        float, 
        epsilon:  float, 
        PDI:      float, 
        p:        float, 
        accuracy: int=16
    ) -> None:
        
        self.R        = R
        self.epsilon  = epsilon
        self.PDI      = PDI
        self.p        = p
        self.accuracy = accuracy
        
        self.class_ = 'spheroid'

        self.R_eff = (4/3)*((p + 1)/(p + 2))*R
        self.R_es  = np.cbrt(epsilon)*R
        self.R_g   = radius_of_gyration(shape=self.class_, R=R, epsilon=epsilon)
        self.V     = PI*(4/3)*epsilon*R**3
        
        return None
        
    
    def generate_scatterers(self, n: int) -> np.ndarray:
        return Spheroid(
            R      =self.R, 
            epsilon=self.epsilon, 
            p      =self.p
        ).generate_scatterers(n=n)
    
    
    def Debye_scattering(
        self, 
        q_arr: np.ndarray, 
        pop:   int=2048, 
        div:   int=256, 
        iter_: int=16
    ) -> np.ndarray:
        
        division = self.accuracy
        
        probability     = np.linspace(start=0, stop=1, num=division + 1, dtype='f')
        probability[0]  = 1e-6
        probability[-1] = 1.0 - 1e-6

        PDI = self.PDI
        k   = 1/PDI
        
        Zs  = SZ_PPF(y=probability, k=k)
        Xs  = SZ_avg(x_0=Zs[:-1], x_1=Zs[1:], k=k)
        Xs  = Xs.astype('f')
        cXs = np.cbrt(Xs)

        storage = np.zeros((division, q_arr.size), dtype='f')
        threads = []

        for _ in range(division):
            threads.append(None)
        
        for i in range(division):                               
            t_ = th.Thread(target=self.scattering, args=(
                q_arr, i, storage, cXs[i], pop, div, iter_
            ))
            t_.start()
            threads[i] = t_
        
        for t in threads:
            t.join()
        
        I_q = np.zeros(shape=q_arr.shape, dtype='f')
        
        for i, I in enumerate(storage):
            I_q += Xs[i]*I*(probability[i + 1] - probability[i])
        
        return I_q/np.max(I_q)
    
    
    def scattering(
        self, 
        q_arr:   np.ndarray, 
        id_:     int, 
        storage: np.ndarray, 
        scale:   float, 
        pop:     int=2048, 
        div:     int=256, 
        iter_:   int=16
    ) -> None:         
        storage[id_, :] += Spheroid(
            R      =scale*self.R, 
            epsilon=self.epsilon, 
            p      =self.p
        ).Debye_scattering(
            q_arr=q_arr, 
            pop  =pop, 
            div  =div, 
            iter_=iter_
        )
        return None
    
    
    def distance_distribution(
        self, 
        probes:    int=4096, 
        div:       int=256, 
        normalize: bool=False
    ) -> tuple[np.ndarray, np.ndarray]:        
        return Spheroid(
            R=self.R, 
            epsilon=self.epsilon, 
            p=self.p
        ).distance_distribution(
            probes   =probes, 
            div      =div, 
            normalize=normalize
        )
        

class Disperse_Cylinder:
    
    def __init__(
        self,
        R:        float, 
        epsilon:  float, 
        PDI:      float, 
        p:        float, 
        accuracy: int=16
    ) -> None:
        
        self.R        = R
        self.epsilon  = epsilon
        self.PDI      = PDI
        self.p        = p
        self.accuracy = accuracy
        
        self.class_ = 'cylinder'

        self.R_eff = (3/2)*((p + 1)/(p + 2))*R
        self.R_es  = np.cbrt(3*epsilon/2)*R
        self.R_g   = radius_of_gyration(shape=self.class_, R=R, epsilon=epsilon)
        self.V     = PI*epsilon*R**3
        
        return None
        
    
    def generate_scatterers(self, n: int) -> np.ndarray:
        return Cylinder(
            R=self.R, 
            epsilon=self.epsilon, 
            p=self.p
        ).generate_scatterers(n=n)
    
    
    def Debye_scattering(
        self, 
        q_arr: np.ndarray, 
        pop:   int=2048, 
        div:   int=256, 
        iter_: int=16
    ) -> np.ndarray:
        
        division = self.accuracy
        
        probability     = np.linspace(start=0, stop=1, num=division + 1, dtype='f')
        probability[0]  = 1e-6
        probability[-1] = 1.0 - 1e-6
        
        PDI = self.PDI
        k   = 1/PDI
        
        Zs  = SZ_PPF(y=probability, k=k)
        Xs  = SZ_avg(x_0=Zs[:-1], x_1=Zs[1:], k=k)
        Xs  = Xs.astype('f')
        cXs = np.cbrt(Xs)

        storage = np.zeros((division, q_arr.size), dtype='f')
        threads = []

        for _ in range(division):
            threads.append(None)
        
        for i in range(division):            
            t_ = th.Thread(target=self.scattering, args=(
                q_arr, i, storage, cXs[i], pop, div, iter_
            ))
            t_.start()
            threads[i] = t_
        
        for t in threads:
            t.join()
        
        I_q = np.zeros(shape=q_arr.shape, dtype='f')
        
        for i, I in enumerate(storage):
            I_q += Xs[i]*I*(probability[i + 1] - probability[i])
        
        return I_q/np.max(I_q)
    
    
    def scattering(
        self, 
        q_arr:   np.ndarray, 
        id_:     int, 
        storage: np.ndarray, 
        scale:   float, 
        pop:     int=2048, 
        div:     int=256, 
        iter_:   int=16
    ) -> None:
        storage[id_, :] += Cylinder(
            R      =scale*self.R, 
            epsilon=self.epsilon, 
            p      =self.p
        ).Debye_scattering(
            q_arr=q_arr, 
            pop  =pop, 
            div  =div, 
            iter_=iter_
        )
        return None
    
    
    def distance_distribution(
        self, 
        probes:    int=4096, 
        div:       int=256, 
        normalize: bool=False
    ) -> tuple[np.ndarray, np.ndarray]:
        return Cylinder(
            R      =self.R, 
            epsilon=self.epsilon, 
            p      =self.p
        ).distance_distribution(
            probes   =probes,
            div      =div, 
            normalize=normalize
        )
    

class Disperse_Spheroid_Shell:
    
    def __init__(
        self,
        R:         float, 
        epsilon:   float, 
        PDI:       float, 
        f_core:    float, 
        rho_delta: float, 
        t:         float, 
        p:         float, 
        q:         float, 
        lt:        float=0, 
        ls:        float=0, 
        accuracy:  int=16
    ) -> None:
        
        self.R         = R
        self.epsilon   = epsilon
        self.PDI       = PDI
        self.f_core    = f_core
        self.rho_delta = rho_delta
        self.t         = t
        self.p         = p
        self.q         = q
        self.lt        = lt
        self.ls        = ls
        self.accuracy  = accuracy
        
        self.class_ = 'spheroid'

        self.R_eff = (4/3)*((p + 1)/(p + 2))*R
        self.R_es  = np.cbrt(epsilon)*R
        self.R_g   = radius_of_gyration(shape=self.class_, R=R, epsilon=epsilon)
        self.V     = PI*(4/3)*epsilon*R**3
        
        return None
        
    
    def generate_scatterers(self, n: int) -> np.ndarray:
        return Spheroid_Shell(
            R        =self.R, 
            epsilon  =self.epsilon, 
            f_core   =self.f_core, 
            rho_delta=self.rho_delta, 
            t        =self.t, 
            p        =self.p, 
            q        =self.q
        ).generate_scatterers(n=n)[0]
    
    
    def Debye_scattering(
        self, 
        q_arr: np.ndarray, 
        pop:   int=2048, 
        div:   int=256, 
        iter_: int=16
    ):
        
        division = self.accuracy
        
        probability     = np.linspace(start=0, stop=1, num=division + 1, dtype='f')
        probability[0]  = 1e-6
        probability[-1] = 1.0 - 1e-6
        
        PDI = self.PDI
        k   = 1/PDI
        
        Zs  = SZ_PPF(y=probability, k=k)
        Xs  = SZ_avg(x_0=Zs[:-1], x_1=Zs[1:], k=k)
        Xs  = Xs.astype('f')
        cXs = np.cbrt(Xs)

        storage = np.zeros((division, q_arr.size), dtype='f')
        threads = []

        for _ in range(division):
            threads.append(None)
        
        for i in range(division):            
            t_ = th.Thread(target=self.scattering, args=(
                q_arr, i, storage, cXs[i], pop, div, iter_
            ))
            t_.start()
            threads[i] = t_
        
        for t in threads:
            t.join()
        
        I_q = np.zeros(shape=q_arr.shape, dtype='f')
        
        for i, I in enumerate(storage):
            I_q += Xs[i]*I*(probability[i + 1] - probability[i])
        
        return I_q/np.max(I_q)
    
    
    def scattering(
        self, 
        q_arr:   np.ndarray, 
        id_:     int, 
        storage: np.ndarray, 
        scale:   float, 
        pop:     int=2048, 
        div:     int=256, 
        iter_:   int=16
    ) -> None:
        storage[id_, :] += Spheroid_Shell(
            R        =scale*self.R, 
            epsilon  =self.epsilon, 
            f_core   =self.f_core, 
            rho_delta=self.rho_delta, 
            t        =scale*self.t, 
            p        =self.p, 
            q        =self.q, 
            lt       =scale*self.lt, 
            ls       =self.ls
        ).Debye_scattering(
            q_arr=q_arr, 
            pop  =pop, 
            div  =div, 
            iter_=iter_
        )
        return None
    
    
    def distance_distribution(
        self, 
        probes:    int=4096, 
        div:       int=256, 
        normalize: bool=False
    ) -> tuple[np.ndarray, np.ndarray]:
        return Spheroid_Shell(
            R        =self.R, 
            epsilon  =self.epsilon, 
            f_core   =self.f_core, 
            rho_delta=self.rho_delta, 
            t        =self.t, 
            p        =self.p, 
            q        =self.q, 
            lt       =self.lt, 
            ls       =self.ls
        ).distance_distribution(
            probes   =probes, 
            div      =div, 
            normalize=normalize
        )


class Disperse_Cylinder_Shell:
    
    def __init__(
        self,
        R:         float, 
        epsilon:   float, 
        PDI:       float, 
        f_core:    float, 
        rho_delta: float, 
        t:         float, 
        p:         float, 
        q:         float, 
        lt:        float=0, 
        ls:        float=0, 
        accuracy:  int=16
    ) -> None:
        
        self.R         = R
        self.epsilon   = epsilon
        self.PDI       = PDI
        self.f_core    = f_core
        self.rho_delta = rho_delta
        self.t         = t
        self.p         = p
        self.q         = q
        self.lt        = lt
        self.ls        = ls
        self.accuracy  = accuracy
        
        self.class_ = 'cylinder'

        self.R_eff = (3/2)*((p + 1)/(p + 2))*R
        self.R_es  = np.cbrt(3*epsilon/2)*R
        self.R_g   = radius_of_gyration(shape=self.class_, R=R, epsilon=epsilon)
        self.V     = PI*epsilon*R**3
        
        return None
        
    
    def generate_scatterers(self, n: int) -> np.ndarray:
        return Cylinder_Shell(
            R        =self.R, 
            epsilon  =self.epsilon, 
            f_core   =self.f_core, 
            rho_delta=self.rho_delta, 
            t        =self.t, 
            p        =self.p, 
            q        =self.q, 
            lt       =self.lt, 
            ls       =self.ls
        ).generate_scatterers(n=n)[0]
    
    
    def Debye_scattering(
        self, 
        q_arr: np.ndarray, 
        pop:   int=2048, 
        div:   int=256, 
        iter_: int=16
    ) -> np.ndarray:
        
        division = self.accuracy
        
        probability     = np.linspace(start=0, stop=1, num=division + 1, dtype='f')
        probability[0]  = 1e-6
        probability[-1] = 1.0 - 1e-6
        
        PDI = self.PDI
        k   = 1/PDI
        
        Zs  = SZ_PPF(y=probability, k=k)
        Xs  = SZ_avg(x_0=Zs[:-1], x_1=Zs[1:], k=k)
        Xs  = Xs.astype('f')
        cXs = np.cbrt(Xs)

        storage = np.zeros(shape=(division, q_arr.size), dtype='f')
        threads = []

        for _ in range(division):
            threads.append(None)
        
        for i in range(division):            
            t_ = th.Thread(target=self.scattering, args=(
                q_arr, i, storage, cXs[i], pop, div, iter_
            ))
            t_.start()
            threads[i] = t_
        
        for t in threads:
            t.join()
        
        I_q = np.zeros(shape=q_arr.shape, dtype='f')
        
        for i, I in enumerate(storage):
            I_q += Xs[i]*I*(probability[i + 1] - probability[i])
        
        return I_q/np.max(I_q)
    
    
    def scattering(
        self, 
        q_arr: np.ndarray, 
        id_: int, 
        storage: np.ndarray, 
        scale: float, 
        pop: int=2048, 
        div: int=256, 
        iter_: int=16
    ) -> None:
        storage[id_, :] += Cylinder_Shell(
            R        =scale*self.R, 
            epsilon  =self.epsilon, 
            f_core   =self.f_core, 
            rho_delta=self.rho_delta, 
            t        =scale*self.t, 
            p        =self.p, 
            q        =self.q, 
            lt       =self.lt, 
            ls       =self.ls
        ).Debye_scattering(
            q_arr=q_arr, 
            pop  =pop, 
            div  =div, 
            iter_=iter_
        )
        return None
    
    
    def distance_distribution(
        self, 
        probes:    int=4096, 
        div:       int=256, 
        normalize: bool=False
    ) -> tuple[np.ndarray, np.ndarray]:
        return Cylinder_Shell(
            R        =self.R, 
            epsilon  =self.epsilon, 
            f_core   =self.f_core, 
            rho_delta=self.rho_delta, 
            t        =self.t, 
            p        =self.p, 
            q        =self.q
        ).distance_distribution(
            probes   =probes, 
            div      =div, 
            normalize=normalize
        )


# def comparison(shape: str, *args, **kwargs) -> None:
    
#     q_log_arr = np.arange(-2, 0, np.true_divide(1, 128))
#     q_arr = np.power(10, q_log_arr - 2*np.log10(2))
    
#     if shape in ['spheroid', 'spheroid shell']:
#         R = 256
#         epsilon = 1.25
#         f_core = 0.75
#         rho_delta = 0.025
#         t = R
#         p = 2.0
#         q = 0.0
        
#     elif shape in ['cylinder', 'cylinder shell']:
#         R = 128
#         epsilon = 8
#         L = R*epsilon
#         f_core = 0.75
#         rho_delta = 0.025
#         t = R
#         p = 1.0
#         q = 0.0
    
#     if shape == 'spheroid':
#         s_0 = Spheroid(R=R, epsilon=epsilon, p=p)
#         s_1 = db.Spheroid(R=R, epsilon=epsilon, p=p)
#     elif shape == 'cylinder':
#         s_0 = Cylinder(R=R, epsilon=epsilon, p=p)
#         s_1 = db.Cylinder(R=R, L=L, p=p)
#     elif shape == 'spheroid shell':
#         s_0 = Spheroid_Shell(
#             R=R, 
#             epsilon=epsilon, 
#             f_core=f_core, 
#             rho_delta=rho_delta, 
#             t=t, 
#             p=p, 
#             q=q
#         )
#         s_1 = db.Spheroid_Shell(
#             R=R, 
#             epsilon=epsilon, 
#             f_core=f_core, 
#             rho_delta=rho_delta, 
#             L=t, 
#             p=p, 
#             q=q
#         )
#     elif shape == 'cylinder shell':
#         s_0 = Cylinder_Shell(
#             R=R, 
#             epsilon=epsilon, 
#             f_core=f_core, 
#             rho_delta=rho_delta, 
#             t=t, 
#             p=p, 
#             q=q
#         )
#         s_1 = db.Cylinder_Shell(
#             R=R, 
#             L=L, 
#             f_core=f_core, 
#             rho_delta=rho_delta, 
#             t=t, 
#             p=p, 
#             q=q
#         )
#     else:
#         return None
    
#     t_i_0 = perf_counter()
#     I_0 = s_0.Debye_scattering(q_arr=q_arr)
#     t_f_0 = perf_counter()
    
#     delta_t_0 = t_f_0 - t_i_0
    
#     t_i_1 = perf_counter()
#     I_1 = s_1.Debye_Scattering(q_arr=q_arr)
#     t_f_1 = perf_counter()
    
#     delta_t_1 = t_f_1 - t_i_1
    
#     plt.figure()
    
#     plt.plot(q_arr, I_0, label='New')
#     plt.plot(q_arr, I_1, label='Original')
    
#     plt.ylim([1e-5, 2])
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.title(f'{shape.capitalize()} Comparison')
#     plt.xlabel('q ($Ã…^{-1}$)')
#     plt.ylabel('Normalized scattering intensity')
#     plt.legend()
    
#     plt.show()
    
#     print(f'New Runtime: {delta_t_0:.3f} s')
#     print(f'Old Runtime: {delta_t_1:.3f} s')
    
#     return None


def test_single(shape: str='spheroid', *args, **kwargs) -> None:
    
    q_log_arr = np.arange(np.log10(0.001), np.log10(0.5), np.true_divide(1, 256))
    q_arr = np.power(10, q_log_arr)
    
    if shape in ['spheroid', 'spheroid shell', 
                 'disperse spheroid', 'disperse spheroid shell']:
        R = 1000
        epsilon = 1.25
        PDI = 0.05
        f_core = 0.75
        rho_delta = 0.025
        t = R
        p = 2.0
        q = 0.0
        lt = 0
        ls = 0
        
    elif shape in ['cylinder', 'cylinder shell', 
                   'disperse cylinder', 'disperse cylinder shell']:
        R = 128
        epsilon = 8
        PDI = 0.05
        f_core = 0.75
        rho_delta = 0.025
        t = R
        p = 1.0
        q = 0.0
        lt = 0
        ls = 0
    else:
        raise ValueError('Invalid shape specified. Please choose from "spheroid", "cylinder", "spheroid shell", "cylinder shell", "disperse spheroid", "disperse cylinder", "disperse spheroid shell", or "disperse cylinder shell".')
    
    if shape == 'spheroid':
        s = Spheroid(R=R, epsilon=epsilon, p=p)
    elif shape == 'cylinder':
        s = Cylinder(R=R, epsilon=epsilon, p=p)
    elif shape == 'spheroid shell':
        s = Spheroid_Shell(
            R=R, 
            epsilon=epsilon, 
            f_core=f_core, 
            rho_delta=rho_delta, 
            t=t, 
            p=p, 
            q=q, 
            lt=lt, 
            ls=ls
        )
    elif shape == 'cylinder shell':
        s = Cylinder_Shell(
            R=R, 
            epsilon=epsilon, 
            f_core=f_core, 
            rho_delta=rho_delta, 
            t=t, 
            p=p, 
            q=q, 
            lt=lt, 
            ls=ls
        )
    elif shape == 'disperse spheroid':
        s = Disperse_Spheroid(R=R, epsilon=epsilon, PDI=PDI, p=p)
    elif shape == 'disperse cylinder':
        s = Disperse_Cylinder(R=R, epsilon=epsilon, PDI=PDI, p=p)
    elif shape == 'disperse spheroid shell':
        s = Disperse_Spheroid_Shell(
            R=R, 
            epsilon=epsilon, 
            PDI=PDI, 
            f_core=f_core, 
            rho_delta=rho_delta, 
            t=t, 
            p=p, 
            q=q, 
            lt=lt, 
            ls=ls
        )
    elif shape == 'disperse cylinder shell':
        s = Disperse_Cylinder_Shell(
            R=R, 
            epsilon=epsilon, 
            PDI=PDI, 
            f_core=f_core, 
            rho_delta=rho_delta, 
            t=t, 
            p=p, 
            q=q, 
            lt=lt, 
            ls=ls
        )
    else:
        return None
    
    t_i = perf_counter()
    I_arr = s.Debye_scattering(q_arr=q_arr)
    t_f = perf_counter()
    
    delta_t = t_f - t_i
    
    plt.figure()
    
    plt.plot(q_arr, I_arr, label=f'Radius: {R:.3f} Ã…')
    
    print(I_arr)
    
    plt.ylim([1e-5, 2])
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f'{shape.title()} Test')
    plt.xlabel('q ($Ã…^{-1}$)')
    plt.ylabel('Normalized scattering intensity')
    plt.legend()
    
    plt.show()
    
    print(f'Runtime: {delta_t:.3f} s')
    
    test_images(s)
    
    return None


def test_images(s, *args, **kwargs) -> None:
    
    num = 4096
    
    scatterers = s.generate_scatterers(num)
    xs, ys, zs = scatterers[:, 0], scatterers[:, 1], scatterers[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    ax.scatter(xs, ys, zs, s=1.0)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(0, -90, 0)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    plt.savefig('assets/cylinder_dots.svg', bbox_inches='tight', pad_inches=0)

    plt.show()
    
    return None


def test_distance(shape: str='spheroid', normalize: bool=False, *args, **kwargs) -> None:
    
    if shape in ['spheroid', 'spheroid shell', 
                 'disperse spheroid', 'disperse spheroid shell']:
        R = 512
        epsilon = 1.25
        PDI = 0.05
        f_core = 0.75
        rho_delta = 0.025
        t = 2*R
        p = 2.0
        q = 0.0
        lt = 0
        ls = 0
        
    elif shape in ['cylinder', 'cylinder shell', 
                   'disperse cylinder', 'disperse cylinder shell']:
        R = 128
        epsilon = 8
        PDI = 0.05
        f_core = 0.75
        rho_delta = 0.025
        t = 2*R
        p = 1.0
        q = 0.0
        lt = 0
        ls = 0
    
    if shape == 'spheroid':
        s = Spheroid(R=R, epsilon=epsilon, p=p)
    elif shape == 'cylinder':
        s = Cylinder(R=R, epsilon=epsilon, p=p)
    elif shape == 'spheroid shell':
        s = Spheroid_Shell(
            R=R, 
            epsilon=epsilon, 
            f_core=f_core, 
            rho_delta=rho_delta, 
            t=t, 
            p=p, 
            q=q, 
            lt=lt, 
            ls=ls
        )
    elif shape == 'cylinder shell':
        s = Cylinder_Shell(
            R=R, 
            epsilon=epsilon, 
            f_core=f_core, 
            rho_delta=rho_delta, 
            t=t, 
            p=p, 
            q=q, 
            lt=lt, 
            ls=ls
        )
    elif shape == 'disperse spheroid':
        s = Disperse_Spheroid(R=R, epsilon=epsilon, PDI=PDI, p=p)
    elif shape == 'disperse cylinder':
        s = Disperse_Cylinder(R=R, epsilon=epsilon, PDI=PDI, p=p)
    elif shape == 'disperse spheroid shell':
        s = Disperse_Spheroid_Shell(
            R=R, 
            epsilon=epsilon, 
            PDI=PDI, 
            f_core=f_core, 
            rho_delta=rho_delta, 
            t=t, 
            p=p, 
            q=q, 
            lt=lt, 
            ls=ls
        )
    elif shape == 'disperse cylinder shell':
        s = Disperse_Cylinder_Shell(
            R=R, 
            epsilon=epsilon, 
            PDI=PDI, 
            f_core=f_core, 
            rho_delta=rho_delta, 
            t=t, 
            p=p, 
            q=q, 
            lt=lt, 
            ls=ls
        )
    else:
        return None
    
    vals, counts = s.distance_distribution(normalize=normalize)
    
    bin_widths = np.zeros(shape=vals.shape)
    bin_widths[:-1] = np.diff(vals)
    bin_widths[-1] = bin_widths[-2]
    
    if normalize:
    
        plt.figure()
        
        plt.bar(vals, counts, width=bin_widths, align='center', edgecolor='blue')
        plt.title(f'{shape.title()} Test')
        plt.xlabel('Normalized Distance')
        plt.ylabel('Normalized scattering intensity')
        
        plt.show()
        
    else:
        
        plt.figure()
        
        plt.bar(vals, counts, width=bin_widths, align='center', edgecolor='blue')
        plt.title(f'{shape.title()} Test')
        plt.xlabel('Distance (Ã…)')
        plt.ylabel('Normalized scattering intensity')
        
        plt.show()
        
    return None


def test_distribution(*args, **kwargs) -> None:
    
    R = 100
    epsilon = 1.25
    PDI = 0.05
    f_core = 0.75
    rho_delta = 0.025
    t = 2*R
    p = 2.0
    q = 0.0
    lt = 0
    ls = 0
        
    s = Disperse_Spheroid_Shell(
        R=R, 
        epsilon=epsilon, 
        PDI=PDI, 
        f_core=f_core, 
        rho_delta=rho_delta, 
        t=t, 
        p=p, 
        q=q, 
        lt=lt, 
        ls=ls
    )
    
    vals_0, counts_0 = s.distance_distribution()
    
    diffs = np.zeros(shape=vals_0.shape)
    diffs[:-1] = np.diff(vals_0)
    diffs[-1] = diffs[-2]
    
    A = np.sum(counts_0*diffs)
    
    R = 200
    epsilon = 1.25
    PDI = 0.05
    f_core = 0.75
    rho_delta = 0.025
    t = 2*R
    p = 2.0
    q = 0.0
    lt = 0
    ls = 0
        
    s = Disperse_Spheroid_Shell(
        R=R, 
        epsilon=epsilon, 
        PDI=PDI, 
        f_core=f_core, 
        rho_delta=rho_delta, 
        t=t, 
        p=p, 
        q=q, 
        lt=lt, 
        ls=ls
    )
    
    vals_1, counts_1 = s.distance_distribution()
    
    diffs = np.zeros(shape=vals_1.shape)
    diffs[:-1] = np.diff(vals_1)
    diffs[-1] = diffs[-2]
    
    B = np.sum(counts_1*diffs)
        
    plt.figure()
    
    plt.bar(vals_0, counts_0/A, align='center', edgecolor='red', label='Radius: 100 Ã…')
    plt.bar(vals_1, counts_1/B, align='center', edgecolor='blue', label='Radius: 200 Ã…')

    plt.xlabel('Intramolecular Distance (Ã…)')
    plt.ylabel('Point-Pair Distance Probability')
    plt.legend()
    
    plt.savefig('assets/disdis.svg', bbox_inches='tight')
    
    plt.show()
    
    return None


def test_vis(*args, **kwargs) -> None:
    
    R = 100
    epsilon = 1.25
    PDI = 0.05
    f_core = 0.75
    rho_delta = 0.025
    t = 2*R
    p = 2.0
    q = 0.0
    lt = 0
    ls = 0
        
    s = Disperse_Spheroid_Shell(
        R=R, 
        epsilon=epsilon, 
        PDI=PDI, 
        f_core=f_core, 
        rho_delta=rho_delta, 
        t=t, 
        p=p, 
        q=q, 
        lt=lt, 
        ls=ls
    )
    
    scatterers, _ = s.generate_scatterers(4096)
    scatterers *= R
    
    xs, ys, zs = scatterers[:, 0], scatterers[:, 1], scatterers[:, 2]
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(xs, ys, zs, marker='o', s=3)

    # ax.set_xlabel('(Ã…)')
    # ax.set_ylabel('(Ã…)')
    # ax.set_zlabel('(Ã…)')
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    
    plt.tight_layout()
    plt.savefig('assets/points.svg', bbox_inches='tight')
    
    plt.show()
    
    return None


def test_scattering(*args, **kwargs) -> None:
    
    q_log_arr = np.arange(-3, 0, np.true_divide(1, 256))
    q_arr = np.power(10, q_log_arr)
    
    R = 100
    epsilon = 1.25
    PDI = 0.05
    f_core = 0.75
    rho_delta = 0.025
    t = 2*R
    p = 2.0
    q = 0.0
    lt = 0
    ls = 0
        
    s = Disperse_Spheroid_Shell(
        R=R, 
        epsilon=epsilon, 
        PDI=PDI, 
        f_core=f_core, 
        rho_delta=rho_delta, 
        t=t, 
        p=p, 
        q=q, 
        lt=lt, 
        ls=ls
    )
    
    I_arr = s.Debye_scattering(q_arr=q_arr)    
        
    plt.figure()
    
    plt.plot(q_arr, I_arr)
    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel('q (Ã…)')
    plt.ylabel('Normalized Scattering Intensity')
    
    plt.tight_layout()
    plt.savefig('assets/scattering.svg', bbox_inches='tight')
    
    plt.show()
    
    return None


def get_data(*args, **kwargs) -> None:
    
    import os
    from datetime import datetime
    
    sample_num = 20
    
    q_log_arr = np.arange(-3, 0, np.true_divide(1, 256))
    q_arr = np.power(10, q_log_arr)
    
    cwd = os.getcwd()
    data_dir = os.path.join(cwd, 'Sample_Folder')
    
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    
    user = os.getlogin()
    date = datetime.today().strftime('%Y%m%d')
    
    name_base = f'{user}_{date}'
    end = 'tsv'
    
    for i in range(sample_num):
        
        R = np.random.triangular(left=64, mode=128, right=256)
        epsilon = np.random.triangular(left=0.5, mode=1.0, right=2.0)
        PDI = 10**np.random.triangular(left=-3, mode=-2, right=-1)
        f_core = np.random.triangular(left=0.65, mode=0.75, right=0.85)
        rho_delta = 0.025
        t = R*np.random.triangular(left=1.0, mode=1.0, right=2.0)
        p = 2.0
        q = 0.0
        lt = 0
        ls = 0
        
        titles = 'Radius,Aspect Ratio,Polydispersity,Core Fraction,Excess Scattering Length Density Ratio,Corona Length,Density Parameter 1,Density Parameteter 2,Layer Parameter 1,Layer Parameter 2'
        params = f'{R},{epsilon},{PDI},{f_core},{rho_delta},{t},{p},{q},{lt},{ls}'
        
        s = Disperse_Spheroid_Shell(
            R=R, 
            epsilon=epsilon, 
            PDI=PDI, 
            f_core=f_core, 
            rho_delta=rho_delta, 
            t=t, 
            p=p, 
            q=q, 
            lt=lt, 
            ls=ls
        )
        
        I_arr = s.Debye_scattering(q_arr=q_arr)
        s_arr = np.sqrt(I_arr)
        
        data = np.hstack((
            q_arr.reshape(-1, 1), 
            I_arr.reshape(-1, 1), 
            s_arr.reshape(-1, 1)
        ))
        
        id_ = i
        
        data_file = os.path.join(data_dir, f'{name_base}_{id_}.{end}')
        text_file = os.path.join(data_dir, f'{name_base}_{id_}_params.csv')
        
        while os.path.exists(data_file):
            id_ += 1
            data_file = os.path.join(data_dir, f'{name_base}_{id_}.{end}')
            text_file = os.path.join(data_dir, f'{name_base}_{id_}_params.txt')
        
        np.savetxt(data_file, data, delimiter='\t')
        
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(titles)
            f.write(params)
    
    return None


def test_speed(*args, **kwargs) -> None:
    
    import time
    
    q_log_arr = np.arange(-3, 0, np.true_divide(1, 256))
    q_arr = np.power(10, q_log_arr)
    
    R = 50
    epsilon = 1.25
    PDI = 0.05
    f_core = 0.75
    rho_delta = 0.025
    t = 2*R
    p = 2.0
    q = 0.0
    lt = 0
    ls = 0
        
    s = Disperse_Spheroid_Shell(
        R=R, 
        epsilon=epsilon, 
        PDI=PDI, 
        f_core=f_core, 
        rho_delta=rho_delta, 
        t=t, 
        p=p, 
        q=q, 
        lt=lt, 
        ls=ls
    )
    
    start = time.time()
    _ = s.Debye_scattering(q_arr=q_arr)  
    elapsed = time.time() - start
        
    print(f"Elapsed time (50 Ã…): {elapsed:.6f} s")
    
    R = 100
    epsilon = 1.25
    PDI = 0.05
    f_core = 0.75
    rho_delta = 0.025
    t = 2*R
    p = 2.0
    q = 0.0
    lt = 0
    ls = 0
        
    s = Disperse_Spheroid_Shell(
        R=R, 
        epsilon=epsilon, 
        PDI=PDI, 
        f_core=f_core, 
        rho_delta=rho_delta, 
        t=t, 
        p=p, 
        q=q, 
        lt=lt, 
        ls=ls
    )
    
    start = time.time()
    _ = s.Debye_scattering(q_arr=q_arr)  
    elapsed = time.time() - start
        
    print(f"Elapsed time (100 Ã…): {elapsed:.6f} s")
    
    R = 200
    epsilon = 1.25
    PDI = 0.05
    f_core = 0.75
    rho_delta = 0.025
    t = 2*R
    p = 2.0
    q = 0.0
    lt = 0
    ls = 0
        
    s = Disperse_Spheroid_Shell(
        R=R, 
        epsilon=epsilon, 
        PDI=PDI, 
        f_core=f_core, 
        rho_delta=rho_delta, 
        t=t, 
        p=p, 
        q=q, 
        lt=lt, 
        ls=ls
    )
    
    start = time.time()
    _ = s.Debye_scattering(q_arr=q_arr)  
    elapsed = time.time() - start
        
    print(f"Elapsed time (200 Ã…): {elapsed:.6f} s")
    
    R = 300
    epsilon = 1.25
    PDI = 0.05
    f_core = 0.75
    rho_delta = 0.025
    t = 2*R
    p = 2.0
    q = 0.0
    lt = 0
    ls = 0
    
    s = Disperse_Spheroid_Shell(
        R=R, 
        epsilon=epsilon, 
        PDI=PDI, 
        f_core=f_core, 
        rho_delta=rho_delta, 
        t=t, 
        p=p, 
        q=q, 
        lt=lt, 
        ls=ls
    )
    
    start = time.time()
    _ = s.Debye_scattering(q_arr=q_arr)  
    elapsed = time.time() - start
        
    print(f"Elapsed time (300 Ã…): {elapsed:.6f} s")
    
    return None


def uniform_sphere_plots() -> None:

    model  = Spheroid(R=100, epsilon=1.0, p=2.0)
    points = model.generate_scatterers(n=4096)[:, :3]

    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], s=1.0)
    unit_circle = plt.Circle((0, 0), 1, fill=False, color='blue')
    plt.gca().add_patch(unit_circle)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Uniform Scatterers Mode for Sphere')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

    model  = Spheroid(R=100, epsilon=1.0, p=0.0)
    points = model.generate_scatterers(n=4096)[:, :3]

    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], s=1.0)
    unit_circle = plt.Circle((0, 0), 1, fill=False, color='blue')
    plt.gca().add_patch(unit_circle)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Uniform Scatterers Mode for Sphere')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

    model  = Cylinder(R=100, epsilon=4.0, p=1.0)
    points = model.generate_scatterers(n=4096)[:, :3]

    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], s=1.0)
    unit_circle = plt.Circle((0, 0), 1, fill=False, color='blue')
    plt.gca().add_patch(unit_circle)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Uniform Scatterers Mode for Cylinder')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

    return None


def test(*args, **kwargs) -> None:
    
    # comparison('cylinder shell')
    # test_single('disperse cylinder shell')
    # test_distance('disperse spheroid shell', True)
    # test_distribution()
    # test_vis()
    # test_scattering()
    # get_data()
    # test_speed()
    uniform_sphere_plots()

    return None


def main(*args, **kwargs) -> int:
    
    test()
        
    return 0


if __name__ == '__main__':
    main()

