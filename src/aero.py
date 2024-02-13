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
def calc_friction_drag_constant(Re, Ma, S, cr, ct, L0):
    lambd =ct/cr
    #Izračun parazitnog otpora trenja
    first_term = 1 / (1 + 0.2 * Ma ** 2) ** 0.467
    second_term = 0.472 / (np.log10(Re * cr * (1 + lambd) / 2)) ** 2.58
    third_term = 1 - (((1 - lambd) ** 4) * (4.55 - 0.27 * np.log10(Re)) * cr) / 100
    Cf = first_term * second_term * third_term
    R_LS = 1.07 + 8/35*(Ma-0.25)-0.972*(1-np.cos(L0))**1.848
    Awet = 2*S
    Cdf = Cf*R_LS*Awet/S #postaviti odgovarajuci omjer Awet i Sref


    return Cdf

def test_calc_constans():
    pass

if __name__ == "__main__":
    test_calc_constans()
