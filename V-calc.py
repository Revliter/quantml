import scipy.integrate as integrate
import numpy as np
import math

namelist = {"000011":0, "000012":1, "000029":2, "000030":3, "000039":4 }
k_list = [4.03679e-07, 1.60518e-07, 3.13979e-07, 1.83822e-07, 4.55493e-07]
b_list = [1.61144555e-07 ,4.23827577e-08 ,2.27322663e-07, 2.49116032e-07 ,1.72804347e-07]
E_list = [10534.58, 860883.32, -44487.24, 64534.30, -192740.35]
phi_list = [3e-07, 2e-07, 1e-07]

global sum
global totv

def V_init(tot):
    global sum
    global totv
    sum = 0
    totv = tot

def V_calc(phi, b, k, E, t, T, dt):
    global sum
    global totv
    gamma = (phi * 1.0 / k) ** 0.5
    def f(u):
        return np.sinh(gamma * (T - u)) / np.sinh(gamma * (T - t))
    
    ans = gamma * np.cosh(gamma * (T - t)) / np.sinh(gamma * (T - t)) * (totv - sum) - b * 1.0 / 2 / k * E * integrate.quad(f, t, T)[0]
    sum += ans * dt
    return math.floor(ans * dt)

# code: a string representing the stock code
# T: total transaction time (unit: second)
# dt: transaction interval (uint: second)
# phi_index: transaction pace (should be 0-fast/1-moderate/2-slow)
# Q: total transaction amount

# the funciton return an np.darray type containing T/dt elements,
# representing the ideal transaction amount for each interval
def Transaction(code, T, dt, phi_index, Q):
    index, res = namelist[str(code)], []
    b, k, phi, E = b_list[index], k_list[index], phi_list[phi_index], E_list[index]
    V_init(Q)
    for t in np.arange(0, T, dt):
        res.append(V_calc(phi, b, k, E, t, T, dt))
    return res