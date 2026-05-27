import numpy as np
import matplotlib.pyplot as plt
import threading as th

from scipy.integrate import quad_vec
from scipy.special import jv
from math import sin, cos, sqrt

from Schulz_Zimm import SZ_avg, SZ_PPF


class Sphere:
    
    
    def __init__(self, R: float, rho: float=1.0) -> None:
        
        self.R = R
        self.rho = rho
        self.V = (4/3)*np.pi*R**3
        
        return None
    
    
    def scattering(self, q_arr: np.ndarray, normalize: bool=True):
        
        qr = np.multiply(q_arr, self.R)
        F = np.true_divide(np.sin(qr) - qr*np.cos(qr), np.power(qr, 3))
        
        if normalize:
            F /= np.max(F)
        else:
            F *= 3*self.V*self.rho

        return 4*np.square(F)
    

class Spheroid:
    
    
    def __init__(self, R: float, epsilon: float=1.0, rho: float=1.0) -> None:
        
        self.R = R
        self.epsilon = epsilon
        self.rho = rho
        self.V = (4/3)*np.pi*epsilon*R**3
        
        return None
        
    
    def Phi(self, qR: np.ndarray) -> np.ndarray:
        
        qR_0 = qR[qR <= 1e-6]
        qR_1 = qR[qR > 1e-6]
        
        _Phi_0 = np.ones(shape=qR_0.shape)
        _Phi_1 = 3*(np.sin(qR_1) - qR_1*np.cos(qR_1))/np.power(qR_1, 3)
        
        return np.concatenate((_Phi_0, _Phi_1))
        
        
    def r_theta(self, theta: float) -> float:
        return self.R*sqrt((sin(theta))**2 + self.epsilon*(cos(theta)**2))

    
    def scattering(self, q_arr: np.ndarray, normalize: bool=True) -> np.ndarray:
                
        def temp_vec(q_arr: np.ndarray) -> np.ndarray:
            
            def func(theta: float, q_arr: np.ndarray) -> np.ndarray:
                r_ = self.r_theta(theta=theta)
                Phi_ = self.Phi(q_arr*r_)
                
                return np.square(Phi_)*np.sin(theta)
            
            return quad_vec(f=func, a=0, b=np.pi/2, args=(q_arr, ))[0]
        
        P = 18*np.pi*(self.rho**2)*(self.V**2)*temp_vec(q_arr)
        
        if normalize:
            return P/np.max(P)
        else:
            return P

    
class Cylinder:
    
    
    def __init__(self, R: float, L: float, rho: float=1.0) -> None:
        
        self.R = R
        self.L = L
        self.rho = rho
        self.V = np.pi*L*R**2
        
        return None
    
    
    def Psi(self, theta: float, q_arr: np.ndarray) -> float:
        
        X = q_arr*self.R*np.sin(theta)
        Y = q_arr*self.L*np.cos(theta/2)
        
        return 2*np.divide(jv(1, X), X)*np.divide(np.sin(Y), Y)
    
    
    def scattering(self, q_arr: np.ndarray, normalize: bool=True):
        
        def temp_vec(q_arr: np.ndarray) -> np.ndarray:
            
            def func(theta: float, q_arr: np.ndarray) -> np.ndarray:
                return np.square(self.Psi(theta=theta, q_arr=q_arr))*sin(theta)
            
            return quad_vec(f=func, a=0, b=np.pi/2, args=(q_arr, ))[0]
        
        P = 18*np.pi*(self.rho**2)*(self.V**2)*temp_vec(q_arr)
        
        if normalize:
            return P/np.max(P)
        else:
            return P


class Disperse_Sphere:
    
    def __init__(self,R: float, PDI: float) -> None:
        
        self.R = R
        self.PDI = PDI
        
        return None
    
    
    def scattering(self, q_arr: np.ndarray, div: int=64):
                
        probability = np.linspace(start=0, stop=1, num=div + 1, dtype='f')
        probability[0] = 0.001
        probability[-1] = 0.999
        
        R = self.R
        PDI = self.PDI
        
        k = 1/PDI
        
        Zs = SZ_PPF(y=probability, k=k)
        Xs = SZ_avg(x_0=Zs[:-1], x_1=Zs[1:], k=k)
        Xs = Xs.astype('f')
        cXs = np.cbrt(Xs)

        storage = np.zeros((div, q_arr.size), dtype='f')
        
        threads = []

        for _ in range(div):
            threads.append(None)
        
        for i in range(div):
            
            params = (R*cXs[i], )
            args = (q_arr, i, storage, params)
            
            t_ = th.Thread(target=self.target, args=args)
            t_.start()
            
            threads[i] = t_
        
        for t in threads:
            t.join()
        
        I_q = np.zeros(shape=q_arr.shape, dtype='f')
        
        for i, I in enumerate(storage):
            I_q += Xs[i]*I*(probability[i + 1] - probability[i])
        
        return I_q/np.max(I_q)
    
    
    def target(
        self, 
        q_arr: np.ndarray, 
        id_: int, 
        storage: np.ndarray, 
        params: tuple, 
    ) -> None:
        
        R, = params            
        S_ = Sphere(R=R)
        
        storage[id_, :] += S_.scattering(q_arr=q_arr)
        
        return None


class Disperse_Spheroid:
    
    def __init__(
        self,
        R: float, 
        epsilon: float, 
        PDI: float
    ) -> None:
        
        self.R = R
        self.epsilon = epsilon
        self.PDI = PDI
    
    
    def scattering(self, q_arr: np.ndarray, div: int=64):
                
        probability = np.linspace(start=0, stop=1, num=div + 1, dtype='f')
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

        storage = np.zeros((div, q_arr.size), dtype='f')
        
        threads = []

        for _ in range(div):
            threads.append(None)
        
        for i in range(div):
            
            params = (R*cXs[i], epsilon)
            args = (q_arr, i, storage, params)
            
            t_ = th.Thread(target=self.target, args=args)
            t_.start()
            
            threads[i] = t_
        
        for t in threads:
            t.join()
        
        I_q = np.zeros(shape=q_arr.shape, dtype='f')
        
        for i, I in enumerate(storage):
            I_q += Xs[i]*I*(probability[i + 1] - probability[i])
        
        return I_q/np.max(I_q)

    
    def target(
        self, 
        q_arr: np.ndarray, 
        id_: int, 
        storage: np.ndarray, 
        params: tuple, 
    ) -> None:
        
        R, epsilon = params            
        S_ = Spheroid(R=R, epsilon=epsilon)
        
        storage[id_, :] += S_.scattering(q_arr=q_arr)
        
        return None


class Disperse_Cylinder:
    
    def __init__(self,R: float, L: float, PDI: float) -> None:
        
        self.R = R
        self.L = L
        self.PDI = PDI
        
        return None
    
    
    def scattering(self, q_arr: np.ndarray, div: int=64):
                
        probability = np.linspace(start=0, stop=1, num=div + 1, dtype='f')
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

        storage = np.zeros((div, q_arr.size), dtype='f')
        
        threads = []

        for _ in range(div):
            threads.append(None)
        
        for i in range(div):
            
            params = (R*cXs[i], L)
            args = (q_arr, i, storage, params)
            
            t_ = th.Thread(target=self.target, args=args)
            t_.start()
            
            threads[i] = t_
        
        for t in threads:
            t.join()
        
        I_q = np.zeros(shape=q_arr.shape, dtype='f')
        
        for i, I in enumerate(storage):
            I_q += Xs[i]*I*(probability[i + 1] - probability[i])
        
        return I_q/np.max(I_q)

    
    def target(
        self, 
        q_arr: np.ndarray, 
        id_: int, 
        storage: np.ndarray, 
        params: tuple, 
    ) -> None:
        
        R, L = params            
        S_ = Cylinder(R=R, L=L)
        
        storage[id_, :] += S_.scattering(q_arr=q_arr)
        
        return None


def generate_disperse_spheroid_scattering(
    R: float, 
    epsilon: float, 
    PDI: float, 
    q_arr: np.ndarray, 
) -> np.ndarray:
    return Disperse_Spheroid(R=R, epsilon=epsilon, PDI=PDI).scattering(q_arr=q_arr)


def generate_disperse_cylinder_scattering(
    R: float,
    L: float,
    PDI: float,
    q_arr: np.ndarray,
) -> np.ndarray:
    return Disperse_Cylinder(R=R, L=L, PDI=PDI).scattering(q_arr=q_arr)


def test() -> None:
    
    

    return None


def main(*args, **kwargs) -> int:
    
    test()
    
    return 0


if __name__ == '__main__':
    main()
