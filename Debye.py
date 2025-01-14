import numpy as np
import matplotlib.pyplot as plt
import threading as th

from math import ceil, sqrt

from Schulz_Zimm import SZ_avg, SZ_PPF
    

class Spheroid:
    
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
        
        self.r_eff = (4/3)*((p + 1)/(p + 2))*R
        
        V = np.pi*(4/3)*epsilon*R**3
        self.V = V
        self.num = int(V*rho)
    
    
    def generate_scatterers(self, n: int) -> np.ndarray:
        
        R = self.R
        epsilon = self.epsilon
        p = self.p
                
        y = np.random.uniform(-1.0 + 1e-6, 1.0 - 1e-6, n).astype('f')
                
        rho_arr = R*np.power(np.random.rand(n).astype('f'), 1/(p + 1))
        theta_arr = np.arccos(epsilon*y/np.sqrt(1 - np.square(y)*(1 - epsilon**2)))
        phi_arr = 2*np.pi*np.random.rand(n).astype('f')
        
        r_arr = epsilon*rho_arr*np.sqrt((epsilon**2 - 1)*np.square(np.sin(theta_arr)) + 1)
        
        x_arr = r_arr*np.sin(theta_arr)*np.cos(phi_arr)
        y_arr = r_arr*np.sin(theta_arr)*np.sin(phi_arr)
        z_arr = r_arr*np.cos(theta_arr)
        
        x_arr = x_arr - np.mean(x_arr)
        y_arr = y_arr - np.mean(y_arr)
        z_arr = z_arr - np.mean(z_arr)
        
        return np.hstack((x_arr.reshape(-1, 1), y_arr.reshape(-1, 1), z_arr.reshape(-1, 1)))
    
    
    def Debye_Scattering(
        self, 
        q_arr: np.ndarray, 
        pop: int=2048, 
        div: int=128
    ) -> np.ndarray:
        
        tries = ceil(ceil(self.num/pop)**2/2)
        shape = q_arr.shape
        
        f_total = np.zeros(shape=shape, dtype='f')

        for i in range(tries):
            
            scatterers = self.generate_scatterers(pop)
            
            r_ij = np.zeros(shape=(pop, pop), dtype='f')
            
            for j in range(3):
                r = scatterers[:, j].reshape(-1, 1)
                r_ij += np.square(r - r.T)
            
            max_ = np.max(r_ij)

            bins = np.square(np.linspace(start=0, stop=1, num=div + 1, dtype='f'))*max_
            inds = np.digitize(r_ij, bins)
            vals, count = np.unique(inds, return_counts=True)
            
            vals = np.sqrt(max_)*(vals - 0.5)/div

            f_q = np.zeros(shape=shape, dtype='f')
                    
            for l, q in enumerate(q_arr):
                qr = np.multiply(q, vals)
                f_q[l] += np.sum(count*np.sin(qr)/qr)
                        
            f_q /= np.max(f_q)
            f_q[f_q <= 0] = np.min(f_q[f_q > 0])
            
            f_total += f_q
                
        return f_total/tries
    

class Cylinder:
    
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
        
        self.r_eff = (3/2)*((p + 1)/(p + 2))*R
        
        V = np.pi*(R**2)*L
        self.V = V
        self.num = int(V*rho)
    
    
    def generate_scatterers(self, n: int) -> np.ndarray:
        
        R = self.R
        L = self.L
        p = self.p
                
        rho_arr = R*np.power(np.random.rand(n).astype('f'), 1/(1 + p))
        theta_arr = 2*np.pi*np.random.rand(n).astype('f')
        
        x_arr = rho_arr*np.cos(theta_arr)
        y_arr = rho_arr*np.sin(theta_arr)
        z_arr = L*np.random.rand(n).astype('f')
        
        return np.hstack((x_arr.reshape(-1, 1), y_arr.reshape(-1, 1), z_arr.reshape(-1, 1)))
    
    
    def Debye_Scattering(
        self, 
        q_arr: np.ndarray, 
        pop: int=2048, 
        div: int=128
    ) -> np.ndarray:
        
        tries = ceil(ceil(self.num/pop)**2/2)
        shape = q_arr.shape
        
        f_total = np.zeros(shape=shape, dtype='f')

        for i in range(tries):
            
            scatterers = self.generate_scatterers(pop)

            r_ij = np.zeros(shape=(pop, pop), dtype='f')
            
            for j in range(3):
                r = scatterers[:, j].reshape(-1, 1)
                r_ij += np.square(r - r.T)
            
            max_ = np.max(r_ij)

            bins = np.square(np.linspace(start=0, stop=1, num=div + 1, dtype='f'))*max_
            inds = np.digitize(r_ij, bins)
            vals, count = np.unique(inds, return_counts=True)
            
            vals = np.sqrt(max_)*(vals - 0.5)/div

            f_q = np.zeros(shape=shape, dtype='f')
                    
            for k, q in enumerate(q_arr):
                qr = q*vals
                f_q[k] += np.sum(count*np.sin(qr)/qr)
                        
            f_q /= np.max(f_q)
            f_q[f_q <= 0] = np.min(f_q[f_q > 0])
            
            f_total += f_q
                
        return f_total/tries


class Spheroid_Shell:
    
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
        
        V = np.pi*(4/3)*epsilon*R**3
        self.V = V
        self.num = int(V*rho)
        
    
    def generate_core(self, n: int) -> np.ndarray:
        
        if n:
            R = self.R
            epsilon = self.epsilon
            p = self.p
            
            y = np.random.uniform(-1.0 + 1e-6, 1.0 - 1e-6, n).astype('f')
                    
            rho_arr = R*np.power(np.random.rand(n).astype('f'), 1/(p + 1))
            theta_arr = np.arccos(epsilon*y/np.sqrt(1 - np.square(y)*(1 - epsilon**2)))
            phi_arr = 2*np.pi*np.random.rand(n).astype('f')
            
            r_arr = epsilon*rho_arr/np.sqrt((epsilon**2 - 1)*np.square(np.sin(theta_arr)) + 1)
            
            x_arr = r_arr*np.sin(theta_arr)*np.cos(phi_arr)
            y_arr = r_arr*np.sin(theta_arr)*np.sin(phi_arr)
            z_arr = r_arr*np.cos(theta_arr)
            
            return np.hstack((x_arr.reshape(-1, 1), y_arr.reshape(-1, 1), z_arr.reshape(-1, 1)))
        
        else:
            return None

    
    def generate_shell(self, n: int) -> np.ndarray:
        
        if n:
            R = self.R
            epsilon = self.epsilon
            L = self.L
            q = self.q
            
            F_rho = ((R + L)**(q + 1) - R**(q + 1))*np.random.rand(n).astype('f') + R**(q + 1)
            
            y = np.random.uniform(-1.0 + 1e-6, 1.0 - 1e-6, n).astype('f')
                    
            rho_arr = np.power(F_rho, 1/(q + 1))
            theta_arr = np.arccos(epsilon*y/np.sqrt(1 - np.square(y)*(1 - epsilon**2)))
            phi_arr = 2*np.pi*np.random.rand(n).astype('f')
            
            r_arr = epsilon*rho_arr/np.sqrt((epsilon**2 - 1)*np.square(np.sin(theta_arr)) + 1)
            
            x_arr = r_arr*np.sin(theta_arr)*np.cos(phi_arr)
            y_arr = r_arr*np.sin(theta_arr)*np.sin(phi_arr)
            z_arr = r_arr*np.cos(theta_arr)
            
            return np.hstack((x_arr.reshape(-1, 1), y_arr.reshape(-1, 1), z_arr.reshape(-1, 1)))
        
        else:
            return None
    
    
    def generate_scatterers(self, n: int) -> np.ndarray:
        
        f_core = self.f_core
        n1 = min(int(f_core*n), n - 1)
        n2 = n - n1
        
        self.n1 = n1
        self.n2 = n2
        
        core = self.generate_core(n=n1)
        shell = self.generate_shell(n=n2)
        
        return np.vstack((core, shell))
    
    
    def Debye_Scattering(
        self, 
        q_arr: np.ndarray, 
        pop: int=2048, 
        div: int=128
    ) -> np.ndarray:
        
        tries = ceil(ceil(self.num/pop)**2/2)
        shape = q_arr.shape
        
        f_final = np.zeros(shape=shape, dtype='f')
        
        for i in range(tries):
            
            scatterers = self.generate_scatterers(pop)
            n1 = self.n1
            
            r_ij = np.zeros(shape=(pop, pop), dtype='f')
            
            for j in range(3):
                r = scatterers[:, j].reshape(-1, 1)
                r_ij += np.square(r - r.T)
            
            d_cc = r_ij[:n1, :n1]
            d_cs = r_ij[n1:, :n1]
            d_sc = r_ij[:n1, n1:]
            d_ss = r_ij[:n1, :n1]
            
            f_total = np.zeros(shape=shape, dtype='f')
            
            for k, d_arr in enumerate([d_cc, d_cs, d_sc, d_ss]):
            
                d_arr = d_arr[d_arr > 0]
                max_ = np.max(d_arr)

                coeff = self.coeffs[k]
                
                bins = np.square(np.linspace(start=0, stop=1, num=div + 1, dtype='f'))*max_
                inds = np.digitize(d_arr, bins)
                vals, count = np.unique(inds, return_counts=True)
                
                vals = np.sqrt(max_)*(vals - 0.5)/div

                f_q = np.zeros(shape=shape, dtype='f')
                        
                for l, q in enumerate(q_arr):
                    qr = q*vals
                    f_q[l] += np.sum(count*np.sin(qr)/qr)
                            
                f_q /= np.max(f_q)
                f_q[f_q <= 0] = np.min(f_q[f_q > 0])
                
                f_total += coeff*f_q
            
            f_final += f_total/np.max(f_total)
                
        return f_final/tries


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
        
        self.R = R
        self.L = L
        self.t = t
        self.p = p
        self.q = q
        self.f_core = f_core
        self.rho_delta = rho_delta
        
        self.r_eff = (3/2)*((p + 1)/(p + 2))*R

        self.rho = rho
        self.coeffs = {0: 1, 1: rho_delta, 2: rho_delta, 3: rho_delta**2}
        
        V = np.pi*(R**2)*L
        self.V = V
        self.num = int(V*rho)
    
    
    def generate_core(self, n: int) -> np.ndarray:
        
        if n:
            R = self.R
            L = self.L
            p = self.p
                    
            rho_arr = R*np.power(np.random.rand(n).astype('f'), 1/(1 + p))
            theta_arr = 2*np.pi*np.random.rand(n).astype('f')
            
            x_arr = rho_arr*np.cos(theta_arr)
            y_arr = rho_arr*np.sin(theta_arr)
            z_arr = L*np.random.rand(n).astype('f')
            
            return np.hstack((x_arr.reshape(-1, 1), y_arr.reshape(-1, 1), z_arr.reshape(-1, 1)))
        
        else:
            return None
    
    
    def generate_shell(self, n: int) -> np.ndarray:
        
        if n:
        
            R = self.R
            L = self.L
            t = self.t
            q = self.q
            
            F_rho = ((R + t)**(q + 1) - R**(q + 1))*np.random.rand(n).astype('f') + R**(q + 1)
            
            rho_arr = np.power(F_rho, 1/(1 + q))
            theta_arr = 2*np.pi*np.random.rand(n).astype('f')
            
            x_arr = rho_arr*np.cos(theta_arr)
            y_arr = rho_arr*np.sin(theta_arr)
            z_arr = L*np.random.rand(n).astype('f')
            
            return np.hstack((x_arr.reshape(-1, 1), y_arr.reshape(-1, 1), z_arr.reshape(-1, 1)))
        
        else:
            return None
    
    
    def generate_scatterers(self, n: int) -> np.ndarray:
        
        f_core = self.f_core
        n1 = min(int(f_core*n), n - 1)
        n2 = n - n1
        
        self.n1 = n1
        self.n2 = n2
        
        core = self.generate_core(n=n1)
        shell = self.generate_shell(n=n2)
        
        return np.vstack((core, shell))
    
    
    def Debye_Scattering(
        self, 
        q_arr: np.ndarray, 
        pop: int=2048, 
        div: int=128
    ) -> np.ndarray:
        
        tries = ceil(ceil(self.num/pop)**2/2)
        shape = q_arr.shape

        f_final = np.zeros(shape=shape, dtype='f')
        
        for i in range(tries):
            
            scatterers = self.generate_scatterers(pop)
            n1 = self.n1
            
            r_ij = np.zeros(shape=(pop, pop), dtype='f')
            
            for j in range(3):
                r = scatterers[:, j].reshape(-1, 1)
                r_ij += np.square(r - r.T)
            
            d_cc = r_ij[:n1, :n1]
            d_cs = r_ij[n1:, :n1]
            d_sc = r_ij[:n1, n1:]
            d_ss = r_ij[:n1, :n1]
            
            f_total = np.zeros(shape=shape, dtype='f')
            
            for k, d_arr in enumerate([d_cc, d_cs, d_sc, d_ss]):
            
                d_arr = d_arr[d_arr > 0]
                max_ = np.max(d_arr)
                
                coeff = self.coeffs[k]
                
                bins = np.square(np.linspace(start=0, stop=1, num=div + 1, dtype='f'))*max_
                inds = np.digitize(d_arr, bins)
                vals, count = np.unique(inds, return_counts=True)
                
                vals = np.sqrt(max_)*(vals - 0.5)/div
                
                f_q = np.zeros(shape=shape, dtype='f')
                        
                for l, q in enumerate(q_arr):
                    qr = q*vals
                    f_q[l] += np.sum(count*np.sin(qr)/qr)
                            
                f_q /= np.max(f_q)
                f_q[f_q <= 0] = np.min(f_q[f_q > 0])
                
                f_total += coeff*f_q
            
            f_final += f_total/np.max(f_total)
        
        return f_final/tries
    
    
class Disperse_Spheroid:
    
    def __init__(
        self,
        R: float, 
        epsilon: float, 
        p: float, 
        PDI: float, 
        rho: float=0.001, 
        accuracy: int=10
    ) -> None:
        
        self.R = R
        self.epsilon = epsilon
        self.p = p
        self.PDI = PDI
        self.accuracy = accuracy
        
        self.r_eff = (4/3)*((p + 1)/(p + 2))*R

        self.rho = rho
        
        V = np.pi*(4/3)*epsilon*R**3
        self.V = V
        self.num = int(V*rho)
        
    
    def generate_scatterers(self, n: int) -> np.ndarray:
        s = Spheroid(
            R=self.R, 
            epsilon=self.epsilon, 
            p=self.p, 
            rho=self.rho
        )
        
        return s.generate_scatterers(n=n)
    
    
    def Debye_Scattering(
        self, 
        q_arr: np.ndarray, 
        pop: int=2048, 
        div: int=128
    ):
        
        division = self.accuracy
        
        probability = np.linspace(start=0, stop=1, num=division + 1, dtype='f')
        probability[0] = 0.001
        probability[-1] = 0.999
        
        R = self.R
        epsilon = self.epsilon
        PDI = self.PDI
        
        k = 1/PDI
        
        Zs = SZ_PPF(y=probability, k=k)
        Xs = SZ_avg(x_0=Zs[:-1], x_1=Zs[1:], k=k)
        Xs = Xs.astype('f')
        cXs = np.cbrt(Xs)

        storage = np.zeros((division, q_arr.size), dtype='f')
        
        threads = []

        for _ in range(division):
            threads.append(None)
        
        for i in range(division):
            
            p = self.p*np.random.uniform(0.9, 1.1)
                                    
            params = (R*cXs[i], epsilon, p)
            args = (q_arr, i, storage, params, pop, div)
            
            t_ = th.Thread(target=self.scattering, args=args)
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
        params: tuple, 
        pop: int=2048, 
        div: int=128
    ) -> None:
        
        R, epsilon, p = params
        rho = 0.001/sqrt(self.accuracy)
            
        S_ = Spheroid(
            R=R, 
            epsilon=epsilon, 
            p=p, 
            rho=rho
        )
        
        storage[id_, :] += S_.Debye_Scattering(q_arr=q_arr, pop=pop, div=div)

        

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
        
        self.R = R
        self.L = L
        self.p = p
        self.PDI = PDI
        self.accuracy = accuracy
        
        self.r_eff = (3/2)*((p + 1)/(p + 2))*R
        
        self.rho = rho
        
        V = np.pi*(R**2)*L
        self.V = V
        self.num = int(V*rho)
        
    
    def generate_scatterers(self, n: int) -> np.ndarray:
        
        s = Cylinder(
            R=self.R, 
            L=self.L, 
            p=self.p, 
            rho=self.rho
        )
        
        return s.generate_scatterers(n=n)
    
    
    def Debye_Scattering(
        self, 
        q_arr: np.ndarray, 
        pop: int=2048, 
        div: int=128
    ):
        
        division = self.accuracy
        
        probability = np.linspace(start=0, stop=1, num=division + 1, dtype='f')
        probability[0] = 0.001
        probability[-1] = 0.999
        
        R = self.R
        L = self.L
        PDI = self.PDI
        
        k = 1/PDI
        
        Zs = SZ_PPF(y=probability, k=k)
        Xs = SZ_avg(x_0=Zs[:-1], x_1=Zs[1:], k=k)
        Xs = Xs.astype('f')
        cXs = np.cbrt(Xs)

        storage = np.zeros((division, q_arr.size), dtype='f')
        
        threads = []

        for _ in range(division):
            threads.append(None)
        
        for i in range(division):
            
            p = self.p*np.random.uniform(0.9, 1.1)
                        
            params = (R*cXs[i], L*cXs[i], p)
            args = (q_arr, i, storage, params, pop, div)
            
            t_ = th.Thread(target=self.scattering, args=args)
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
        params: tuple, 
        pop: int=2048, 
        div: int=128
    ) -> None:
        
        R, L, p = params
        rho = 0.001/sqrt(self.accuracy)
            
        S_ = Cylinder(
            R=R, 
            L=L, 
            p=p, 
            rho=rho
        )
        
        storage[id_, :] += S_.Debye_Scattering(q_arr=q_arr, pop=pop, div=div)

    

class Disperse_Spheroid_Shell:
    
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
        
        self.R = R
        self.epsilon = epsilon
        self.L = L
        self.p = p
        self.q = q
        self.f_core = f_core
        self.PDI = PDI
        self.rho_delta = rho_delta
        self.accuracy = accuracy
        
        self.r_eff = (4/3)*((p + 1)/(p + 2))*R

        self.rho = rho
        
        V = np.pi*(4/3)*epsilon*R**3
        self.V = V
        self.num = int(V*rho)
        
    
    def generate_scatterers(self, n: int) -> np.ndarray:
        
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
        q_arr: np.ndarray, 
        pop: int=2048, 
        div: int=128
    ):
        
        division = self.accuracy
        
        probability = np.linspace(start=0, stop=1, num=division + 1, dtype='f')
        probability[0] = 0.001
        probability[-1] = 0.999
        
        R = self.R
        epsilon = self.epsilon
        PDI = self.PDI
        
        k = 1/PDI
        
        Zs = SZ_PPF(y=probability, k=k)
        Xs = SZ_avg(x_0=Zs[:-1], x_1=Zs[1:], k=k)
        Xs = Xs.astype('f')
        cXs = np.cbrt(Xs)

        storage = np.zeros((division, q_arr.size), dtype='f')
        
        threads = []

        for _ in range(division):
            threads.append(None)
        
        for i in range(division):
            
            L = self.L*np.random.uniform(0.9, 1.1)
            p = self.p*np.random.uniform(0.9, 1.1)
            q = self.q*np.random.uniform(0.9, 1.1)
            f_core = min(self.f_core*np.random.uniform(0.9, 1.1), 1.0 - 1e-6)
                        
            params = (R*cXs[i], epsilon, L*cXs[i], p, q, f_core)
            args = (q_arr, i, storage, params, pop, div)
            
            t_ = th.Thread(target=self.scattering, args=args)
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
        params: tuple, 
        pop: int=2048, 
        div: int=128
    ) -> None:
        
        R, epsilon, L, p, q, f_core = params
        rho = 0.001/sqrt(self.accuracy)
            
        S_ = Spheroid_Shell(
            R=R, 
            epsilon=epsilon, 
            L=L, 
            p=p, 
            q=q, 
            f_core=f_core, 
            rho=rho
        )
        
        storage[id_, :] += S_.Debye_Scattering(q_arr=q_arr, pop=pop, div=div)



class Disperse_Cylinder_Shell:
    
    def __init__(
        self,
        R: float, 
        L: float, 
        PDI: float, 
        f_core: float, 
        rho_delta: float, 
        t: float, 
        p: float, 
        q: float, 
        rho: float=0.001, 
        accuracy: int=16
    ) -> None:
        
        self.R = R
        self.L = L
        self.t = t
        self.p = p
        self.q = q
        self.f_core = f_core
        self.PDI = PDI
        self.rho_delta = rho_delta
        self.accuracy = accuracy
        
        self.r_eff = (3/2)*((p + 1)/(p + 2))*R

        self.rho = rho
                
        V = np.pi*(R**2)*L
        self.V = V
        self.num = int(V*rho)
        
    
    def generate_scatterers(self, n: int) -> np.ndarray:
        
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
        
        return s.generate_scatterers(n=n)
    
    
    def Debye_Scattering(
        self, 
        q_arr: np.ndarray, 
        pop: int=2048, 
        div: int=128
    ):
        
        division = self.accuracy
        
        probability = np.linspace(start=0, stop=1, num=division + 1, dtype='f')
        probability[0] = 0.001
        probability[-1] = 0.999
        
        R = self.R
        L = self.L
        PDI = self.PDI
        
        k = 1/PDI
        
        Zs = SZ_PPF(y=probability, k=k)
        Xs = SZ_avg(x_0=Zs[:-1], x_1=Zs[1:], k=k)
        Xs = Xs.astype('f')
        cXs = np.cbrt(Xs)

        storage = np.zeros(shape=(division, q_arr.size), dtype='f')
        
        threads = []

        for _ in range(division):
            threads.append(None)
        
        for i in range(division):
            
            t = self.t*np.random.uniform(0.9, 1.1)
            p = self.p*np.random.uniform(0.9, 1.1)
            q = self.q*np.random.uniform(0.9, 1.1)
            f_core = min(self.f_core*np.random.uniform(0.9, 1.1), 1.0 - 1e-6)
                        
            params = (R*cXs[i], L*cXs[i], t*cXs[i], p, q, f_core)
            args = (q_arr, i, storage, params, pop, div)
            
            t_ = th.Thread(target=self.scattering, args=args)
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
        params: tuple, 
        pop: int=2048, 
        div: int=128
    ) -> None:
        
        R, L, t, p, q, f_core = params
        rho = 0.001/sqrt(self.accuracy)
            
        S_ = Cylinder_Shell(
            R=R, 
            L=L, 
            t=t, 
            p=p, 
            q=q, 
            f_core=f_core, 
            rho=rho
        )
        
        storage[id_, :] += S_.Debye_Scattering(q_arr=q_arr, pop=pop, div=div)


def test_shapes() -> None:
    
    R = np.power(2, np.random.triangular(6, 7, 8))
    R = 100
    epsilon = np.random.uniform(low=0.8, high=1.2)
    epsilon = 1.2
    p = np.random.uniform(low=1.5, high=2.0)
    p = 2.0
    q = 1.0
    L = R*np.random.uniform(low=5.0, high=7.0)
    
    print((R, epsilon, p, q, L))
    
    L_ = 2*R
    t = R
    f_core = 0.75
    q = 0
    
    num = 1000
    
    s = Spheroid(R=R, epsilon=epsilon, p=p)
    c = Cylinder(R=R, L=L, p=p - 1)
    
    cs = Spheroid_Shell(R=R, epsilon=epsilon, L=L_, f_core=f_core, p=p, q=q)
    cc = Cylinder_Shell(R=R, L=L, t=t, f_core=f_core, p=p - 1, q=q)
    
    scatterers_s = s.generate_scatterers(num)
    scatterers_c = c.generate_scatterers(num)
    scatterers_cs = cs.generate_scatterers(num)
    scatterers_cc = cc.generate_scatterers(num)
        
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
    
    q_log_arr = np.arange(-2, 0, np.true_divide(1, 128))
    q_arr = np.power(10, q_log_arr - 2*np.log10(2))
    
    f_q_s = s.Debye_Scattering(q_arr=q_arr)
    f_q_c = c.Debye_Scattering(q_arr=q_arr)
    f_q_cs = cs.Debye_Scattering(q_arr=q_arr)
    f_q_cc = cc.Debye_Scattering(q_arr=q_arr)

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
    
    q_log_arr = np.arange(-2, 0, np.true_divide(1, 128))
    q_arr = np.power(10, q_log_arr - 2*np.log10(2))
    
    R = np.power(2, np.random.triangular(left=6.0, mode=7.0, right=8.0))
    epsilon = np.power(2, np.random.triangular(left=-1.0, mode=0.0, right=1.0))
    PDI = np.power(10, np.random.uniform(np.log10(0.001), np.log10(0.5)))
    L = R*np.random.uniform(1.0, 2.0)
    p = np.random.uniform(1.75, 2.0)
    q = np.random.uniform(0.0, 0.25)
    f_core = np.random.triangular(0.5, 0.75, 1.0)
    
    s = Disperse_Spheroid_Shell(
        R=R, 
        epsilon=epsilon, 
        L=L, 
        p=p, 
        q=q, 
        f_core=f_core, 
        PDI=PDI
    )
    
    print(f'Radius: {R:.3f}')
    print(f'Aspect ratio: {epsilon:.3f}')
    print(f'PDI: {PDI:.3f}')
    print(f'Shell strength: {f_core:.3f}')
    
    I_arr = s.Debye_Scattering(q_arr=q_arr)
    
    plt.figure()
    plt.plot(q_arr, I_arr)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid()
    plt.show()
    

def main(*args, **kwargs) -> int:
    
    test_single()
    
    return 0


if __name__ == '__main__':
    main()
