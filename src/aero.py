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

def calc_friction_drag_constant(Re,Ma,cr, lambd, L0,A_wt_A_ref):
    #Izračun parazitnog otpora trenja
    first_term = 1 / (1 + 0.2 * Ma ** 2) ** 0.467
    second_term = 0.472 / (np.log10(Re * cr * (1 + lambd) / 2)) ** 2.58
    third_term = 1 - (((1 - lambd) ** 4) * (4.55 - 0.27 * np.log10(Re)) * cr) / 100
    Cf = first_term * second_term * third_term
    R_LS = 1.07 + 8/35*(Ma-0.25)-0.972*(1-np.cos(L0))**1.848
    Cdf = Cf*R_LS*A_wt_A_ref
    return Cdf

def wing_mass_MHS(Sw, c_mac, t_c_max, rho_mat, Kp, AR, n_ult, L_1_4, lambd):
    '''
    Wing weght estimation Mohhamad H Saddrey, Aircraft Designe ... 2012, EQ 10.3
    :param Sw:wing planform area
    :param c_mac: wing mean aerodynamic chord
    :param t_c_max: the maximum thickness-to-chord ratio
    :param rho_mat:density of construction material
    :param Kp:wing density factor Table 10.8
    :param AR:aspect ratio
    :param n_ult:ultimate load factor (usually nult = 1.5 · nmax)
    :param L_1_4:quarter chord sweep angle
    :param lambd:taper ratio
    :return: ww wing weight
    '''
    # region Tables
    # No. Aircraft – wing structural installation condition Kρ
    # 1 GA, no engine, no fuel tank in the wing 0.0011–0.0013
    # 2 GA, no engine on the wing, fuel tank in the wing 0.0014–0.0018
    # 3 GA, engine installed on the wing, no fuel tank in the wing 0.0025–0.003
    # 4 GA, engine installed on the wing, fuel tank in the wing 0.003–0.0035
    # 5 Home-built 0.0012–0.002
    # 6 Transport, cargo, airliner (engines attached to the wing) 0.0035–0.004
    # 7 Transport, cargo, airliner (engines not attached to the wing) 0.0025–0.003
    # 8 Supersonic fighter, few light stores under wing 0.004–0.006
    # 9 Supersonic fighter, several heavy stores under wing 0.009–0.012
    # 10 Remotely controlled model 0.001–0.0015

    # Table 10.6 Density of various aerospace materials
    # No. Engineering materials Density (kg/m3)
    # 1 Aerospace aluminum 2711
    # 2 Fiberglass/epoxy 1800–1850
    # 3 Graphite/epoxy 1520–1630
    # 4 Low-density foam 16–30
    # 5 High-density foam 50–80
    # 6 Steel alloys 7747
    # 7 Titanium alloys 4428
    # 8 Balsa wood 160
    # 9 Plastics (including Monokote) 900–1400
    # endregion
    ww = Sw*c_mac*t_c_max*rho_mat*Kp*(AR*n_ult/(np.cos(L_1_4)))**0.6*lambd**0.04
    return ww


def test_calc_constans():
    Cdf =calc_friction_drag_constant(3*10**6,0.2,10,1,1,0)
    print('Cdf = ', Cdf)

if __name__ == "__main__":

    test_calc_constans()
