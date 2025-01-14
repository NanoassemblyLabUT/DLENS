import numpy as np

from scipy.special import gammainc
from scipy.stats import gamma


def SZ_CDF(x: np.ndarray, k: float) -> np.ndarray:
    """
    Returns the cumulative distribution function (CDF) for a Schulz-Zimm 
    distribution.
    
    Arguments:
        x: random variable; [0, inf)
        k: the shape parameter (k > 0); equals to 1/PDI
    
    This is essentially the lower incomplete gamma function.
    """
    return gammainc(k, k*x)


def SZ_PDF(x: np.ndarray, k: float) -> np.ndarray:
    """
    Returns the probability density function (PDF) for a Schulz-Zimm 
    distribution.
    
    Arguments:
        x: random variable; [0, inf)
        k: the shape parameter (k > 0); equals to 1/PDI
    
    This is essentially a gamma distribution with the scale parameter set to
    1/k, k being the shape parameter.
    """
    return gamma.pdf(x, k, scale=1/k)


def SZ_PPF(y: np.ndarray, k: float) -> np.ndarray:
    """
    Returns the percent point function (PPF) for a Schulz-Zimm 
    distribution.
    
    Arguments:
        y: desired output for Shulz-Zimm CDF
        k: the shape parameter (k > 0); equals to 1/PDI
    
    This returns the random variable, x, that corresponds to the desired output
    from the CDF of a specified Schulz-Zimm distribution.
    """
    return gamma.ppf(y, k, scale=1/k)


def SZ_avg(x_0: np.ndarray, x_1: np.ndarray, k: float) -> np.ndarray:
    """
    Returns the percent point function (PPF) for a Schulz-Zimm 
    distribution.
    
    Arguments:
        x: random variable; [0, inf)
        k: the shape parameter (k > 0); equals to 1/PDI
    
    This returns the average x value between x_1 and x_0 by dividing the result
    of integrating x*f(x) by the result of integrading f(x).
    """
    return (gammainc(k + 1, k*x_1) - gammainc(k + 1, k*x_0))/(gammainc(k, k*x_1) - gammainc(k, k*x_0))


def main(*args, **kwargs) -> int:
    return 0


if __name__ == '__main__':
    main()
