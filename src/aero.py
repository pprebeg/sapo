import numpy as np


def calc_reynolds(mu, rho_h, c_m, V_inf):
    Re = rho_h * V_inf * c_m / mu
    return Re
def calc_mach(V_inf, a_h):
    Ma = a_h / V_inf
    return Ma


def calc_air_density_speedsound_ICAO(h_m):
    # ICAO standardne vrijednosti na razini mora

    p0 = 101325  # Tlak na razini mora, Pa
    T0 = 288.15  # Temperatura na razini mora, K
    g = 9.80665  # Gravitacijska konstanta, m/s²
    R = 287.05  # Specifična plinska konstanta za suhi zrak, J/(kg·K)
    L = 0.0065  # Temperaturni gradijent, K/m
    S = 110.4 #Sutherlandova konstanta , K
    T_0 = 273 # Referentna temperatura , K
    mu_0 = 1.716 * 10**(-5) #viskoznoest zraka za T_0, Pa·s
    # Izračun temperature i tlaka na zadanoj visini
    T = T0 - L * h_m
    p = p0 * (1 - L * h_m / T0) ** (g / (R * L))

    # Izračun gustoće zraka kg/m³
    rho = p / (R * T)
    # Izračun brzine zvuka m/s
    a = 20.5*T**0.5
    #Sutherlandova formula za dinamičku viskoznost zraka Pa·s
    mu = mu_0 * ((T_0+S)/(T+S))*(T/T_0)**(3/2)
    return rho, a, mu
def calc_constants(Re,Ma,S,cr,ct,L0):

    return 0

def test_calc_constans():
    pass

if __name__ == "__main__":
    test_calc_constans()
