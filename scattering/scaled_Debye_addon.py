import os
import tqdm

import numpy as np
import matplotlib.pyplot as plt
import threading as th

from .scaled_Debye import Spheroid_Shell, Disperse_Spheroid_Shell, Cylinder_Shell, Disperse_Cylinder_Shell
from .Schulz_Zimm import SZ_avg, SZ_PPF

from math import pi as PI
from time import time
    

class Disk:
    
    def __init__(
        self,
        R:         float,
        h:         float,
        t_shell:   float,
        f_core:    float=None,
        rho_delta: float=0.05,
    ) -> None:
        
        self.R         = R
        self.h         = h
        self.t_shell   = t_shell
        self.f_core    = f_core if f_core is not None else h/(h + 2*t_shell)
        self.rho_delta = rho_delta

        self._class = 'disk'

        self.V = np.pi*R**2*h

        self.maxes = np.array((
            (h/R)**2 + 2**2,
            ((h + t_shell)/R)**2 + 2**2,
            ((h + t_shell)/R)**2 + 2**2,
            ((h + 2*t_shell)/R)**2 + 2**2
        ))
        self.coeffs = np.array((
            1.0,
            rho_delta,
            rho_delta,
            rho_delta**2
        ))

        return None
    

    def generate_points(self, n: int=2048, in_core: bool=True) -> np.ndarray:
        
        if in_core:
            r_min = 0.0
            r_max = 1.0
            h_min = 0.0
            h_max = (self.h/2)/self.R
        else:
            r_min = 0.0
            r_max = 1.0
            h_min = (self.h/2)/self.R
            h_max = (self.h/2 + self.t_shell)/self.R
        
        r     = np.sqrt(np.random.rand(n))*(r_max - r_min) + r_min
        theta = 2*np.pi*np.random.rand(n)
        h     = np.random.rand(n)*(h_max - h_min) + h_min

        x = r*np.cos(theta)
        y = r*np.sin(theta)
        z = h

        points = np.stack((x, y, z), axis=1)

        return points


    def generate_scatterers(self, n: int=2048) -> tuple[np.ndarray, tuple[int, int]]:

        n_core    = int(n*self.f_core)
        n_core_a  = n_core//2
        n_core_b  = n_core - n_core_a
        n_shell   = n - n_core
        n_shell_a = n_shell//2
        n_shell_b = n_shell - n_shell_a

        rs_core_a  = self.generate_points(n_core_a, in_core=True)
        rs_core_b  = self.generate_points(n_core_b, in_core=True)
        rs_shell_a = self.generate_points(n_shell_a, in_core=False)
        rs_shell_b = self.generate_points(n_shell_b, in_core=False)

        rs_core_b  *= np.array([1, 1, -1])
        rs_shell_b *= np.array([1, 1, -1])
        points = np.vstack((rs_core_a, rs_core_b, rs_shell_a, rs_shell_b))

        return points, (n_core, n_shell)
    

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
        
        if normalize:
            return vals, counts/np.sum(counts)
        else:
            return self.R*vals, counts/np.sum(counts)
    

class Disperse_Disk:
    
    def __init__(
        self,
        R:         float, 
        h:         float, 
        t_shell:   float, 
        PDI:       float, 
        f_core:    float=None,
        rho_delta: float=0.05,
        accuracy:  int=16
    ) -> None:
        
        self.R         = R
        self.h         = h
        self.t_shell   = t_shell
        self.PDI       = PDI
        self.f_core    = f_core
        self.rho_delta = rho_delta
        self.accuracy  = accuracy
        
        self.class_ = 'disk'

        self.V = np.pi*R**2*h
        
        return None
        
    
    def generate_scatterers(self, n: int) -> np.ndarray:
        return Disk(
            R        =self.R, 
            h        =self.h, 
            t_shell  =self.t_shell, 
            f_core   =self.f_core,
            rho_delta=self.rho_delta
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
        storage[id_, :] += Disk(
            R      =scale*self.R, 
            h      =scale*self.h, 
            t_shell=scale*self.t_shell, 
            f_core =self.f_core,
            rho_delta=self.rho_delta
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
        return Disk(
            R        =self.R, 
            h        =self.h, 
            t_shell  =self.t_shell, 
            f_core   =self.f_core,
            rho_delta=self.rho_delta
        ).distance_distribution(
            probes   =probes, 
            div      =div, 
            normalize=normalize
        )
    

class Worm():

    def __init__(
        self, 
        R:         float,
        L_seg:     float,
        n_seg:     int,
        t_shell:   float,
        p_core:    float,
        p_shell:   float,
        f_core:    float=None,
        rho_delta: float=0.05,
        theta_max: float=np.pi/6
    ) -> None:
        
        self.R         = R
        self.L_seg     = L_seg
        self.n_seg     = n_seg
        self.t_shell   = t_shell
        self.p_core    = p_core
        self.p_shell   = p_shell
        self.f_core    = f_core if f_core is not None else (L_seg/2)/(L_seg/2 + t_shell)
        self.rho_delta = rho_delta
        self.theta_max = theta_max

        self._class = 'worm'

        self.V = np.pi*R**2*L_seg*n_seg

        self.coeffs = np.array((
            1.0,
            rho_delta,
            rho_delta,
            rho_delta**2
        ))

        self._sample_model()
        self._generate_backbone()

        return None

    def _sample_direction(self, v_prev: np.ndarray, theta_max: float) -> np.ndarray:

        v_prev /= np.linalg.norm(v_prev)

        cos_theta = 1 - np.random.rand()*(1 - np.cos(theta_max))
        sin_theta = np.sqrt(1 - cos_theta**2)
        phi = 2*np.pi*np.random.rand()

        x_local = sin_theta*np.cos(phi)
        y_local = sin_theta*np.sin(phi)
        z_local = cos_theta

        if np.allclose(v_prev, [0, 0, 1]):
            ortho = np.array([1, 0, 0])
        else:
            ortho = np.cross(v_prev, [0, 0, 1])
            ortho /= np.linalg.norm(ortho)
        ortho2 = np.cross(v_prev, ortho)
        new_dir = (
            x_local*ortho +
            y_local*ortho2 +
            z_local*v_prev
        )

        return new_dir/np.linalg.norm(new_dir)


    def _generate_backbone(self) -> None:

        origins       = np.zeros(shape=(self.n_seg, 3))
        vecs_tangent  = np.zeros(shape=(self.n_seg, 3))
        vecs_normal   = np.zeros(shape=(self.n_seg, 3))
        vecs_binormal = np.zeros(shape=(self.n_seg, 3))

        v_prev = np.random.randn(3)
        v_prev /= np.linalg.norm(v_prev)
        v_norm = np.random.randn(3)
        v_norm -= v_norm.dot(v_prev)*v_prev
        v_norm /= np.linalg.norm(v_norm)
        v_binorm = np.cross(v_prev, v_norm)

        origins[0]       += np.zeros(3)
        vecs_tangent[0]  += v_prev
        vecs_normal[0]   += v_norm
        vecs_binormal[0] += v_binorm

        for i in range(1, self.n_seg):

            origins[i, :] = origins[i - 1, :] + (self.L_seg/self.R)*vecs_tangent[i - 1, :]

            v_tan = self._sample_direction(v_prev, self.theta_max)
            v_norm = v_tan - v_prev
            v_norm -= v_norm.dot(v_tan)*v_tan
            v_norm /= np.linalg.norm(v_norm)
            v_binorm = np.cross(v_tan, v_norm)

            vecs_tangent[i, :]  = v_tan
            vecs_normal[i, :]   = v_norm
            vecs_binormal[i, :] = v_binorm

            v_prev = v_tan

        self.origins       = origins
        self.vecs_tangent  = vecs_tangent
        self.vecs_normal   = vecs_normal
        self.vecs_binormal = vecs_binormal

        return None
    

    def _sample_model(self) -> None:

        self.model = Cylinder_Shell(
            R=self.R,
            epsilon=self.L_seg/self.R,
            f_core=self.f_core,
            rho_delta=self.rho_delta,
            t=self.t_shell,
            p=self.p_core,
            q=self.p_shell
        )

        return None
    

    def generate_points(self, seg_num: int, n: int=2048) -> tuple[np.ndarray, tuple[int, int]]:

        points, (n_core, n_shell) = self.model.generate_scatterers(n)

        origin   = self.origins[seg_num, :]
        tangent  = self.vecs_tangent[seg_num, :]
        normal   = self.vecs_normal[seg_num, :]
        binormal = self.vecs_binormal[seg_num, :]

        R = np.column_stack((normal, binormal, tangent))

        points = points@R.T + origin

        return points, (n_core, n_shell)
    

    def generate_scatterers(self, n: int=2048) -> tuple[np.ndarray, tuple[int, int]]:

        n_sample = n//self.n_seg

        temp_scatterers = list()
        ns_arr = np.zeros(shape=(self.n_seg, 2), dtype=int)

        for i in range(self.n_seg):
            if i < self.n_seg - 1:
                points, (n_core, n_shell) = self.generate_points(seg_num=i, n=n_sample)
            else:
                points, (n_core, n_shell) = self.generate_points(seg_num=i, n=n_sample + n%self.n_seg)
            ns_arr[i, :] += np.array((n_core, n_shell))
            temp_scatterers.append(points)
        
        core_scatterers  = np.vstack([temp_scatterers[i][:ns_arr[i, 0]] for i in range(self.n_seg)])
        shell_scatterers = np.vstack([temp_scatterers[i][ns_arr[i, 0]:] for i in range(self.n_seg)])

        ns_arr = ns_arr.sum(axis=0)

        return np.vstack((core_scatterers, shell_scatterers)), (ns_arr[0], ns_arr[1])
    

    def _find_max(self, arr1: np.ndarray, arr2: np.ndarray) -> float:

        # arr1: (n1, 1, 3)
        # arr2: (1, n2, 3)
        # distance: (n1, n2, 3)
        distances = arr1[:, np.newaxis, :] - arr2[np.newaxis, :, :]
        maximum = np.max(np.sum(np.square(distances), axis=-1))

        return maximum
    

    def Debye_scattering(
        self, 
        q_arr: np.ndarray, 
        pop:   int=2048, 
        div:   int=256, 
        iter_: int=16
    ) -> np.ndarray:
        
        I_arr = np.zeros(shape=q_arr.shape, dtype='f')

        for _ in range(iter_):
            
            scatterers, (n1, _) = self.generate_scatterers(pop)

            max_0 = self._find_max(scatterers[:n1, :], scatterers[:n1, :])
            max_1 = self._find_max(scatterers[n1:, :], scatterers[:n1, :])
            max_2 = self._find_max(scatterers[:n1, :], scatterers[n1:, :])
            max_3 = self._find_max(scatterers[n1:, :], scatterers[n1:, :])

            bins_arr = [
                np.square(np.linspace(start=0, stop=1, num=div + 1, dtype='f'))*max_0, 
                np.square(np.linspace(start=0, stop=1, num=div + 1, dtype='f'))*max_1, 
                np.square(np.linspace(start=0, stop=1, num=div + 1, dtype='f'))*max_2, 
                np.square(np.linspace(start=0, stop=1, num=div + 1, dtype='f'))*max_3
            ]
            vals_arr = [
                np.sqrt(max_0)*(np.arange(start=1, stop=div + 1) - 0.5)/div, 
                np.sqrt(max_1)*(np.arange(start=1, stop=div + 1) - 0.5)/div, 
                np.sqrt(max_2)*(np.arange(start=1, stop=div + 1) - 0.5)/div, 
                np.sqrt(max_3)*(np.arange(start=1, stop=div + 1) - 0.5)/div, 
            ]
            counts_arr = [
                np.zeros(shape=(div, )), 
                np.zeros(shape=(div, )), 
                np.zeros(shape=(div, )), 
                np.zeros(shape=(div, ))
            ]
            
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
        
            qr_0    = q_arr[:, np.newaxis]*self.R*vals_arr[0][np.newaxis, :]
            I_arr_0 = self.coeffs[0]*np.sum(counts_arr[0]*np.sinc(qr_0/PI), axis=1)
            qr_1    = q_arr[:, np.newaxis]*self.R*vals_arr[1][np.newaxis, :]
            I_arr_1 = self.coeffs[1]*np.sum(counts_arr[1]*np.sinc(qr_1/PI), axis=1)
            qr_2    = q_arr[:, np.newaxis]*self.R*vals_arr[2][np.newaxis, :]
            I_arr_2 = self.coeffs[2]*np.sum(counts_arr[2]*np.sinc(qr_2/PI), axis=1)
            qr_3    = q_arr[:, np.newaxis]*self.R*vals_arr[3][np.newaxis, :]
            I_arr_3 = self.coeffs[3]*np.sum(counts_arr[3]*np.sinc(qr_3/PI), axis=1)
        
            I_arr += I_arr_0 + I_arr_1 + I_arr_2 + I_arr_3
        
        I_arr /= np.max(I_arr)
        I_arr[I_arr <= 0] = np.min(I_arr[I_arr > 0])
                
        return I_arr
    

    def distance_distribution(
        self, 
        probes:    int=4096, 
        div:       int=256, 
        normalize: bool=False
    ) -> tuple[np.ndarray, np.ndarray]:
                            
        scatterers, _ = self.generate_scatterers(probes)

        max_   = self._find_max(scatterers, scatterers)
        bins   = np.square(np.linspace(start=0, stop=1, num=div + 1, dtype='f'))*max_
        vals   = np.sqrt(max_)*(np.arange(start=1, stop=div + 1) - 0.5)/div
        counts = np.zeros(shape=(div, ))
        
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
        

class Disperse_Worm:
    
    def __init__(
        self, 
        R:         float,
        L_seg:     float,
        n_seg:     int,
        t_shell:   float,
        p_core:    float,
        p_shell:   float,
        PDI:       float,
        f_core:    float=None,
        rho_delta: float=0.05,
        theta_max: float=np.pi/6,
        accuracy:  int=16
    ) -> None:
        
        self.R         = R
        self.L_seg     = L_seg
        self.n_seg     = n_seg
        self.t_shell   = t_shell
        self.p_core    = p_core
        self.p_shell   = p_shell
        self.PDI       = PDI
        self.f_core    = f_core
        self.rho_delta = rho_delta
        self.theta_max = theta_max
        self.accuracy  = accuracy
        
        self._class = 'worm'

        self.V = np.pi*R**2*L_seg*n_seg

        self.coeffs = np.array((
            1.0,
            rho_delta,
            rho_delta,
            rho_delta**2
        ))

        self._sample_model()
        self.model._generate_backbone()

        self.origins       = self.model.origins
        self.vecs_tangent  = self.model.vecs_tangent
        self.vecs_normal   = self.model.vecs_normal
        self.vecs_binormal = self.model.vecs_binormal
        
        return None
    

    def _sample_model(self) -> None:
        self.model = Worm(
            R        =self.R, 
            L_seg    =self.L_seg,
            n_seg    =self.n_seg,
            t_shell  =self.t_shell, 
            p_core   =self.p_core,
            p_shell  =self.p_shell,
            f_core   =self.f_core,
            rho_delta=self.rho_delta,
            theta_max=self.theta_max
        )
        return None
        
    
    def generate_scatterers(self, n: int) -> np.ndarray:
        return Worm(
            R        =self.R, 
            L_seg    =self.L_seg,
            n_seg    =self.n_seg,
            t_shell  =self.t_shell, 
            p_core   =self.p_core,
            p_shell  =self.p_shell,
            f_core   =self.f_core,
            rho_delta=self.rho_delta,
            theta_max=self.theta_max
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
        storage[id_, :] += Worm(
            R        =scale*self.R, 
            L_seg    =scale*self.L_seg,
            n_seg    =self.n_seg,
            t_shell  =scale*self.t_shell, 
            p_core   =self.p_core,
            p_shell  =self.p_shell,
            f_core   =self.f_core,
            rho_delta=self.rho_delta,
            theta_max=self.theta_max
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
        return Worm(
            R        =self.R, 
            L_seg    =self.L_seg,
            n_seg    =self.n_seg,
            t_shell  =self.t_shell, 
            p_core   =self.p_core,
            p_shell  =self.p_shell,
            f_core   =self.f_core,
            rho_delta=self.rho_delta,
            theta_max=self.theta_max
        ).distance_distribution(
            probes   =probes, 
            div      =div, 
            normalize=normalize
        )


class Empty_Shell(Spheroid_Shell):
    def __init__(
        self, 
        R:         float, 
        epsilon:   float, 
        t:         float, 
        p:         float, 
        rho_delta: float=0.05
    ) -> None:
        
        super().__init__(
            R=R, 
            epsilon=epsilon, 
            t=t, 
            p=p, 
            q=p, 
            f_core=0.01,
            rho_delta=rho_delta
        )

        return None
    

class Disperse_Empty_Shell(Disperse_Spheroid_Shell):
    def __init__(
        self, 
        R:         float, 
        epsilon:   float, 
        t:         float, 
        p:         float, 
        PDI:       float, 
        rho_delta: float=0.05,
        accuracy:  int=16
    ) -> None:
        
        super().__init__(
            R=R, 
            epsilon=epsilon, 
            t=t, 
            p=p, 
            q=p, 
            f_core=0.01,
            PDI=PDI,
            rho_delta=rho_delta,
            accuracy=accuracy
        )

        return None


def test_shapes():

    def set_axes_equal(ax):
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])

        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)

        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    # worm = Disperse_Worm(
    #     R=10.0,
    #     L_seg=20.0,
    #     n_seg=10,
    #     t_shell=5.0,
    #     p_core=1.0,
    #     p_shell=0.0,
    #     PDI=0.01,
    #     f_core=0.85,
    #     rho_delta=0.05,
    #     theta_max=np.pi/3
    # )

    # worm_scatterers = worm.generate_scatterers(n=2048)

    # fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    # ax.scatter(worm_scatterers[:, 0], worm_scatterers[:, 1], worm_scatterers[:, 2], s=1)
    # set_axes_equal(ax)
    # plt.show()

    disk = Disperse_Disk(
        R=100.0,
        h=20.0,
        t_shell=10.0,
        PDI=0.01,
        f_core=0.80,
        rho_delta=0.05
    )
    disk_scatterers = disk.generate_scatterers(n=2048)

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.scatter(disk_scatterers[:, 0], disk_scatterers[:, 1], disk_scatterers[:, 2], s=1)
    set_axes_equal(ax)
    plt.show()

    return None


def generate_disk(
    q_arr:   np.ndarray,
    R:       float, 
    h:       float,
    t_shell: float,
    PDI:     float,
    f_core:  float=None,
    rho_delta: float=0.05,
) -> np.ndarray:
    return Disperse_Disk(
        R=R,
        h=h,
        t_shell=t_shell,
        PDI=PDI,
        f_core=f_core,
        rho_delta=rho_delta
    ).Debye_scattering(q_arr=q_arr, iter_=8, pop=1024)


def generate_worm(
    q_arr:   np.ndarray,
    R:       float,
    L_seg:   float,
    n_seg:   int,
    t_shell: float,
    p_core:  float,
    p_shell: float,
    PDI:     float,
    f_core:  float=None,
    rho_delta: float=0.05,
    theta_max: float=np.pi/6,
) -> np.ndarray:
    return Disperse_Worm(
        R=R,
        L_seg=L_seg,
        n_seg=n_seg,
        t_shell=t_shell,
        p_core=p_core,
        p_shell=p_shell,
        PDI=PDI,
        f_core=f_core,
        rho_delta=rho_delta,
        theta_max=theta_max
    ).Debye_scattering(q_arr=q_arr, iter_=8, pop=1024)


def generate_empty_shell(
    q_arr:   np.ndarray,
    R:       float,
    epsilon: float,
    t:       float,
    p:       float,
    PDI:     float,
    rho_delta: float=0.05,
    accuracy: int=16
) -> np.ndarray:
    return Disperse_Empty_Shell(
        R=R,
        epsilon=epsilon,
        t=t,
        p=p,
        PDI=PDI,
        rho_delta=rho_delta,
        accuracy=accuracy
    ).Debye_scattering(q_arr=q_arr, iter_=8, pop=1024)


def generate_inverse(
    q_arr:   np.ndarray,
    R:       float, 
    epsilon: float,
    PDI:     float,
    inverse: bool=False,
    rho_delta: float=0.05,
) -> np.ndarray:
    factor = 1.0 if not inverse else -1.0
    return Disperse_Spheroid_Shell(
        R=R,
        epsilon=epsilon,
        PDI=PDI,
        f_core=0.50,
        rho_delta=factor*rho_delta,
        t=R,
        p=2.0,
        q=0.0
    ).Debye_scattering(q_arr=q_arr, iter_=8, pop=1024)


def generate_random_disk(q_arr: np.ndarray, n: int=2048) -> None:

    cwd      = os.getcwd()
    base_dir = os.path.join(cwd, "cross_test")
    data_dir = os.path.join(base_dir, "inverse")
    os.makedirs(data_dir, exist_ok=True)

    X = np.zeros(shape=(3*n, q_arr.size))
    y = np.zeros(shape=(3*n, 6))

    for i in tqdm.tqdm(range(n)):

        R         = 10.0**np.random.uniform(low=np.log10(50.0), high=np.log10(500.0))
        h         = R*np.random.uniform(low=0.1, high=0.25)
        t_shell   = h*2**np.random.uniform(low=-2.0, high=1.0)
        PDI_0     = 10.0**np.random.uniform(low=np.log10(0.001), high=np.log10(0.01))
        PDI_1     = 10.0**np.random.uniform(low=np.log10(0.01), high=np.log10(0.1))
        PDI_2     = 10.0**np.random.uniform(low=np.log10(0.1), high=np.log10(0.5))
        f_core    = np.random.uniform(low=0.5, high=0.85)
        rho_delta = 0.1

        y[3*i, :]     = np.array((R, h, t_shell, PDI_0, f_core, rho_delta))
        y[3*i + 1, :] = np.array((R, h, t_shell, PDI_1, f_core, rho_delta))
        y[3*i + 2, :] = np.array((R, h, t_shell, PDI_2, f_core, rho_delta))
        X[3*i, :]     = generate_disk(q_arr=q_arr, R=R, h=h, t_shell=t_shell, PDI=PDI_0, f_core=f_core, rho_delta=rho_delta)
        X[3*i + 1, :] = generate_disk(q_arr=q_arr, R=R, h=h, t_shell=t_shell, PDI=PDI_1, f_core=f_core, rho_delta=rho_delta)
        X[3*i + 2, :] = generate_disk(q_arr=q_arr, R=R, h=h, t_shell=t_shell, PDI=PDI_2, f_core=f_core, rho_delta=rho_delta)

    np.save(os.path.join(data_dir, "disk_test.npy"), X)
    np.save(os.path.join(data_dir, "disk_test_labels.npy"), y)

    log_path = os.path.join(data_dir, "disk_log.txt")
    with open(log_path, 'w') as f:
        f.write("Parameter record setting:\n")
        f.write("Radius\tHeight\tShell Thickness\tPDI\tCore Fraction\tSLD Ratio\n")

    return None


def generate_random_worm(q_arr: np.ndarray, n: int=2048) -> None:

    cwd      = os.getcwd()
    base_dir = os.path.join(cwd, "cross_test")
    data_dir = os.path.join(base_dir, "inverse")
    os.makedirs(data_dir, exist_ok=True)

    X = np.zeros(shape=(3*n, q_arr.size))
    y = np.zeros(shape=(3*n, 9))

    for i in tqdm.tqdm(range(n)):

        R         = 10.0**np.random.uniform(low=np.log10(20.0), high=np.log10(128.0))
        L_seg     = R*np.random.uniform(low=2.0, high=8.0)
        n_seg     = np.random.randint(low=5, high=20)
        t_shell   = R*2**np.random.uniform(low=-1.0, high=1.0)
        p_core    = 1.0
        p_shell   = 0.0
        PDI_0     = 10.0**np.random.uniform(low=np.log10(0.001), high=np.log10(0.01))
        PDI_1     = 10.0**np.random.uniform(low=np.log10(0.01), high=np.log10(0.1))
        PDI_2     = 10.0**np.random.uniform(low=np.log10(0.1), high=np.log10(0.5))
        f_core    = np.random.uniform(low=0.5, high=0.85)
        rho_delta = 0.1

        y[3*i, :]     = np.array((R, L_seg, n_seg, t_shell, p_core, p_shell, PDI_0, f_core, rho_delta))
        y[3*i + 1, :] = np.array((R, L_seg, n_seg, t_shell, p_core, p_shell, PDI_1, f_core, rho_delta))
        y[3*i + 2, :] = np.array((R, L_seg, n_seg, t_shell, p_core, p_shell, PDI_2, f_core, rho_delta))
        X[3*i, :]     = generate_worm(q_arr=q_arr, R=R, L_seg=L_seg, n_seg=n_seg, t_shell=t_shell, p_core=p_core, p_shell=p_shell, PDI=PDI_0, f_core=f_core, rho_delta=rho_delta)
        X[3*i + 1, :] = generate_worm(q_arr=q_arr, R=R, L_seg=L_seg, n_seg=n_seg, t_shell=t_shell, p_core=p_core, p_shell=p_shell, PDI=PDI_1, f_core=f_core, rho_delta=rho_delta)
        X[3*i + 2, :] = generate_worm(q_arr=q_arr, R=R, L_seg=L_seg, n_seg=n_seg, t_shell=t_shell, p_core=p_core, p_shell=p_shell, PDI=PDI_2, f_core=f_core, rho_delta=rho_delta)

    np.save(os.path.join(data_dir, "worm_test.npy"), X)
    np.save(os.path.join(data_dir, "worm_test_labels.npy"), y)

    log_path = os.path.join(data_dir, "worm_log.txt")
    with open(log_path, 'w') as f:
        f.write("Parameter record setting:\n")
        f.write("Radius\tLength\tSegments\tShell Thickness\tCore Density Parameter\tShell Density Parameter\tPDI\tCore Fraction\tSLD Ratio\n")

    return None


def generate_random_empties(q_arr: np.ndarray, n: int=2048) -> None:

    cwd      = os.getcwd()
    base_dir = os.path.join(cwd, "cross_test")
    data_dir = os.path.join(base_dir, "inverse")
    os.makedirs(data_dir, exist_ok=True)

    X = np.zeros(shape=(3*n, q_arr.size))
    y = np.zeros(shape=(3*n, 6))

    for i in tqdm.tqdm(range(n)):

        R         = 10.0**np.random.uniform(low=np.log10(50.0), high=np.log10(500.0))
        epsilon   = 2**np.random.uniform(low=-1.0, high=2.0)
        t         = R*2**np.random.uniform(low=-1.0, high=1.0)
        p         = 2.0
        PDI_0     = 10.0**np.random.uniform(low=np.log10(0.001), high=np.log10(0.01))
        PDI_1     = 10.0**np.random.uniform(low=np.log10(0.01), high=np.log10(0.1))
        PDI_2     = 10.0**np.random.uniform(low=np.log10(0.1), high=np.log10(0.5))
        rho_delta = 0.1

        y[3*i, :]     = np.array((R, epsilon, t, p, PDI_0, rho_delta))
        y[3*i + 1, :] = np.array((R, epsilon, t, p, PDI_1, rho_delta))
        y[3*i + 2, :] = np.array((R, epsilon, t, p, PDI_2, rho_delta))
        X[3*i, :]     = generate_empty_shell(q_arr=q_arr, R=R, epsilon=epsilon, t=t, p=p, PDI=PDI_0, rho_delta=rho_delta)
        X[3*i + 1, :] = generate_empty_shell(q_arr=q_arr, R=R, epsilon=epsilon, t=t, p=p, PDI=PDI_1, rho_delta=rho_delta)
        X[3*i + 2, :] = generate_empty_shell(q_arr=q_arr, R=R, epsilon=epsilon, t=t, p=p, PDI=PDI_2, rho_delta=rho_delta)

    np.save(os.path.join(data_dir, "empty_test.npy"), X)
    np.save(os.path.join(data_dir, "empty_test_labels.npy"), y)

    log_path = os.path.join(data_dir, "empty_log.txt")
    with open(log_path, 'w') as f:
        f.write("Parameter record setting:\n")
        f.write("Radius\tAspect Ratio\tShell Thickness\tDensity Parameter\tPDI\tSLD Ratio\n")

    return None


def generate_random_inverse(q_arr: np.ndarray, n: int=1024) -> None:

    cwd      = os.getcwd()
    base_dir = os.path.join(cwd, "cross_test")
    data_dir = os.path.join(base_dir, "inverse")
    os.makedirs(data_dir, exist_ok=True)

    X = np.zeros(shape=(3*n, q_arr.size))
    y = np.zeros(shape=(3*n, 9))

    for i in tqdm.tqdm(range(n)):

        R       = 10.0**np.random.uniform(low=np.log10(50.0), high=np.log10(500.0))
        epsilon = 2.0**np.random.uniform(low=np.log10(0.4), high=np.log10(2.5))
        PDI_0   = 10.0**np.random.uniform(low=np.log10(0.001), high=np.log10(0.5))
        PDI_1   = 10.0**np.random.uniform(low=np.log10(0.001), high=np.log10(0.5))
        PDI_2   = 10.0**np.random.uniform(low=np.log10(0.001), high=np.log10(0.5))

        rho_delta = np.random.uniform(low=0.05, high=0.5)

        # y[6*i + 0, :] = np.array((R, epsilon, PDI_0, False, 0.50, 0.25, R, 2.0, 0.0))
        # y[6*i + 1, :] = np.array((R, epsilon, PDI_0, True,  0.50, 0.25, R, 2.0, 0.0))
        # y[6*i + 2, :] = np.array((R, epsilon, PDI_1, False, 0.50, 0.25, R, 2.0, 0.0))
        # y[6*i + 3, :] = np.array((R, epsilon, PDI_1, True,  0.50, 0.25, R, 2.0, 0.0))
        # y[6*i + 4, :] = np.array((R, epsilon, PDI_2, False, 0.50, 0.25, R, 2.0, 0.0))
        # y[6*i + 5, :] = np.array((R, epsilon, PDI_2, True,  0.50, 0.25, R, 2.0, 0.0))

        # X[6*i + 0, :] = generate_inverse(q_arr=q_arr, R=R, epsilon=epsilon, PDI=PDI_0, inverse=False)
        # X[6*i + 1, :] = generate_inverse(q_arr=q_arr, R=R, epsilon=epsilon, PDI=PDI_0, inverse=True)
        # X[6*i + 2, :] = generate_inverse(q_arr=q_arr, R=R, epsilon=epsilon, PDI=PDI_1, inverse=False)
        # X[6*i + 3, :] = generate_inverse(q_arr=q_arr, R=R, epsilon=epsilon, PDI=PDI_1, inverse=True)
        # X[6*i + 4, :] = generate_inverse(q_arr=q_arr, R=R, epsilon=epsilon, PDI=PDI_2, inverse=False)
        # X[6*i + 5, :] = generate_inverse(q_arr=q_arr, R=R, epsilon=epsilon, PDI=PDI_2, inverse=True)

        y[3*i + 0, :] = np.array((R, epsilon, PDI_0, True, 0.50, rho_delta, R, 2.0, 0.0))
        y[3*i + 1, :] = np.array((R, epsilon, PDI_1, True, 0.50, rho_delta, R, 2.0, 0.0))
        y[3*i + 2, :] = np.array((R, epsilon, PDI_2, True, 0.50, rho_delta, R, 2.0, 0.0))

        X[3*i + 0, :] = generate_inverse(q_arr=q_arr, R=R, epsilon=epsilon, PDI=PDI_0, inverse=True, rho_delta=rho_delta)
        X[3*i + 1, :] = generate_inverse(q_arr=q_arr, R=R, epsilon=epsilon, PDI=PDI_1, inverse=True, rho_delta=rho_delta)
        X[3*i + 2, :] = generate_inverse(q_arr=q_arr, R=R, epsilon=epsilon, PDI=PDI_2, inverse=True, rho_delta=rho_delta)

    np.save(os.path.join(data_dir, "inverse_test_many.npy"), X)
    np.save(os.path.join(data_dir, "inverse_test_labels_many.npy"), y)

    log_path = os.path.join(data_dir, "inverse_log_many.txt")
    with open(log_path, 'w') as f:
        f.write("Parameter record setting:\n")
        f.write("Radius\tAspect Ratio\tPDI\tInversed\tCore Fraction\tSLD Ratio\tShell Thickness\tCore Density Parameter\tShell Density Parameter\n")

    return None


def plot_sphere_scattering() -> None:

    q_arr = 10**np.linspace(start=np.log10(0.0025), stop=np.log10(0.5), num=257, dtype='f')

    R         = 100.0
    epsilon   = 1.0
    PDI       = 0.01
    f_core    = 0.50
    rho_delta = 0.25
    t         = 200.0
    p         = 2.0
    q         = 0.0

    I_normal = Disperse_Spheroid_Shell(
        R        =R,
        epsilon  =epsilon,
        t        =t,
        p        =p,
        q        =q,
        f_core   =f_core,
        PDI      =PDI,
        rho_delta=rho_delta
    ).Debye_scattering(q_arr=q_arr, iter_=16, pop=4096)

    I_empty = Disperse_Empty_Shell(
        R        =R,
        epsilon  =epsilon,
        t        =t,
        p        =p,
        PDI      =PDI,
        rho_delta=rho_delta
    ).Debye_scattering(q_arr=q_arr, iter_=16, pop=4096)

    I_inverse = Disperse_Spheroid_Shell(
        R        =R,
        epsilon  =epsilon,
        t        =t,
        p        =p,
        q        =q,
        f_core   =f_core,
        PDI      =PDI,
        rho_delta=-rho_delta
    ).Debye_scattering(q_arr=q_arr, iter_=16, pop=4096)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(q_arr, I_normal, label='Normal')
    axes[0].set_title('Normal Spheroid Core-Shell')
    axes[0].set_xlabel('q (1/Ã…)')
    axes[0].set_ylabel('Normalized Intensity')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')

    axes[1].plot(q_arr, I_empty, label='Empty', color='orange')
    axes[1].set_title('Empty Spheroid Shell')
    axes[1].set_xlabel('q (1/Ã…)')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')

    axes[2].plot(q_arr, I_inverse, label='Inverse', color='green')
    axes[2].set_title('Sign-Inversed Scattering Spheroid Core-Shell')
    axes[2].set_xlabel('q (1/Ã…)')
    axes[2].set_xscale('log')
    axes[2].set_yscale('log')

    plt.tight_layout()
    plt.show()

    return None


def plot_cylinder_scattering() -> None:
    
    q_arr = 10**np.linspace(start=np.log10(0.0025), stop=np.log10(0.5), num=257, dtype='f')

    R         = 100.0
    epsilon   = 1.0
    PDI       = 0.01
    f_core    = 0.50
    rho_delta = 0.25
    t         = 200.0
    p         = 1.0
    q         = 1.0

    I_normal = Disperse_Cylinder_Shell(
        R        =R,
        epsilon  =epsilon,
        t        =t,
        p        =p,
        q        =q,
        f_core   =f_core,
        PDI      =PDI,
        rho_delta=rho_delta
    ).Debye_scattering(q_arr=q_arr, iter_=16, pop=4096)

    I_disk = Disperse_Disk(
        R        =R,
        h        =0.1*R,
        t_shell  =t,
        PDI      =PDI,
        f_core   =f_core,
        rho_delta=rho_delta
    ).Debye_scattering(q_arr=q_arr, iter_=16, pop=4096)

    I_worm = Disperse_Worm(
        R        =R,
        L_seg    =4*R,
        n_seg    =16,
        t_shell  =t,
        p_core   =p,
        p_shell  =q,
        f_core   =f_core,
        PDI      =PDI,
        rho_delta=rho_delta
    ).Debye_scattering(q_arr=q_arr, iter_=16, pop=4096)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(q_arr, I_normal, label='Normal')
    axes[0].set_title('Normal Cylindrical Core-Shell')
    axes[0].set_xlabel('q (1/Ã…)')
    axes[0].set_ylabel('Normalized Intensity')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')

    axes[1].plot(q_arr, I_disk, label='Disk', color='orange')
    axes[1].set_title('Disk Scattering')
    axes[1].set_xlabel('q (1/Ã…)')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')

    axes[2].plot(q_arr, I_worm, label='Worm', color='green')
    axes[2].set_title('Worm Scattering')
    axes[2].set_xlabel('q (1/Ã…)')
    axes[2].set_xscale('log')
    axes[2].set_yscale('log')

    plt.tight_layout()
    plt.show()

    return None



def main() -> int:

    q_arr = 10**np.linspace(start=np.log10(0.0025), stop=np.log10(0.5), num=257, dtype='f')

    # print("Generating random disk scattering data...")
    # generate_random_disk(q_arr=q_arr, n=50)
    # print("Generating random worm scattering data...")
    # generate_random_worm(q_arr=q_arr, n=50)
    # print("Generating random empty scattering data...")
    # generate_random_empties(q_arr=q_arr, n=50)
    print("Generating random inverse scattering data...")
    generate_random_inverse(q_arr=q_arr, n=450)

    # plot_sphere_scattering()
    # plot_cylinder_scattering()

    # cwd       = os.getcwd()
    # base_dir  = os.path.join(cwd, "cross_test")
    # data_dir  = os.path.join(base_dir, "inverse")
    # data_path = os.path.join(data_dir, "inverse_test_new.npy")

    # data = np.load(data_path)

    # I_arr = data[np.random.randint(low=0, high=data.shape[0]), :]

    # plt.figure(figsize=(8, 6))
    # plt.plot(q_arr, I_arr)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()

    return 0


if __name__ == "__main__":
    main()

