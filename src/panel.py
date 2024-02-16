import numpy as np
from src.naca  import naca
import matplotlib.pyplot as plt

def get_points_for_CPACS_NACA_4_5(profNaca, n_points, n_decimals=5):
    x, y = naca(profNaca, n_points, False, True)
    x=np.array(x)
    y = np.array(y)
    x=x.round(n_decimals)
    y=y.round(n_decimals)
    return x[::-1],y[::-1]
def c1c2(xj, zj, deltaj, xk, zk, deltak, Sk):
    # C1C2.py - Pomoćna funkcija za program PANEL.py

    ksij = (xj - xk) * np.cos(deltak) + (zj - zk) * np.sin(deltak)
    etaj = -(xj - xk) * np.sin(deltak) + (zj - zk) * np.cos(deltak)

    b = -2 * ksij
    c = ksij**2 + etaj**2

    eps = 1e-20
    D = 4 * etaj**2
    if D < eps:
        F1 = 2 / b - 2 / (b + 2 * Sk)
    else:
        F1 = (2 / np.sqrt(D)) * (np.arctan((2 * Sk + b) / np.sqrt(D)) - np.arctan(b / np.sqrt(D)))
    F2 = (1 / 2) * np.log((c + b * Sk + Sk**2) / c) - (b / 2) * F1

    I1 = etaj * F1 - (etaj / Sk) * F2
    I2 = (etaj / Sk) * F2
    I3 = (ksij - c / Sk) * F1 + (ksij / Sk - 1) * F2 + 1
    I4 = (c / Sk) * F1 - (ksij / Sk) * F2 - 1

    Cn1 = I1 * np.sin(deltaj - deltak) + I3 * np.cos(deltaj - deltak)
    Cn2 = I2 * np.sin(deltaj - deltak) + I4 * np.cos(deltaj - deltak)

    Ct1 = I1 * np.cos(deltaj - deltak) - I3 * np.sin(deltaj - deltak)
    Ct2 = I2 * np.cos(deltaj - deltak) - I4 * np.sin(deltaj - deltak)

    return Cn1, Cn2, Ct1, Ct2


import numpy as np


def naca_4_5_code(M, P, XX, n):
    # NACA4 Geometrija NACA profila serije 4.
    # Funkcija NACA4 definira koordinate profila NACA serije 4 za zadanu
    # maksimalnu zakrivljenost M (u stotinama tetive), apcisa maksimalne
    # zakrivljenosti P (u desetinama tetive) te najvece debljine XX u stotinama
    # tetive). Te su veličine dane u relativnom odnosu prema tetivi (c=1).
    # Tako da prva tri ulazna podatka predstavljaju oznaku željenog NACA
    # profila: NACA MPXX.
    # Pri tome funkcija generira ukupno 2*N+1 točaka (N točaka na gornjaci i
    # N točaka na donjaci uz to da je točka na izlaznom rubu prva, a ujedno i
    # zadnja). Tocke su složene, počev od izlaznog ruba preko donjake pa nazad
    # preko gornjake do izlaznog ruba. U pozivu oblika:
    #
    #    [X,Z] = NACA4(M,P,XX,N)
    #
    # u vektoru X spremljene su x koordinate točaka, a z koordinate točaka u
    # vektoru Z.

    c = 1

    # Najveća zakrivljenost srednje linije (relativno prema duljini tetive)
    f = M / 100
    # Položaj najveće zakrivljenosti srednje linije (rel. prema duljini tetive)
    xf = P / 10
    # Najveća debljina (rel. prema duljini tetive)
    t = XX / 100

    # Raspodjela xc koordinate duž tetive
    beta = np.linspace(np.pi / 2, np.pi, n + 1)
    xc = c * (1 + np.cos(beta))  # cos-raspodjela: gušće oko LE

    # Promjena debljine = f(xc)
    zt = 5 * t * (0.2969 * np.sqrt(xc) - 0.126 * xc - 0.3537 * (xc) ** 2 + 0.2843 * (xc) ** 3 - 0.1015 * (xc) ** 4)

    # Srednja linija = f(xc) i nagib srednje linije = f(xc)
    x_g, z_g, x_d, z_d = [], [], [], []
    for i in range(n + 1):
        if f == 0 or xf == 0:
            # Simetričan profil!
            # Koordinate točaka na gornjaci
            x_g.append(xc[i])
            z_g.append(zt[i])
            # Koordinate točaka na donjaci
            x_d.append(xc[i])
            z_d.append(-zt[i])
        else:
            if xc[i] / c < xf:
                zc = c * f / xf ** 2 * (2 * xf * xc[i] / c - (xc[i] / c) ** 2)
                theta = np.arctan(2 * f / xf ** 2 * (xf - xc[i] / c))
            else:
                zc = c * f / (1 - xf) ** 2 * (1 - 2 * xf + 2 * xf * (xc[i] / c) - (xc[i] / c) ** 2)
                theta = np.arctan(2 * f / (1 - xf) ** 2 * (xf - xc[i] / c))

            # Koordinate točaka na gornjaci
            x_g.append(xc[i] - zt[i] * c * np.sin(theta))
            z_g.append(zc + zt[i] * c * np.cos(theta))
            # Koordinate točaka na donjaci
            x_d.append(xc[i] + zt[i] * c * np.sin(theta))
            z_d.append(zc - zt[i] * c * np.cos(theta))

    # Preslagivanje vektora x i z koordinata
    x = np.concatenate([x_d[0:n], x_g[::-1]])
    z = np.concatenate([z_d[0:n], z_g[::-1]])

    return x, z


def panel(naca_4_5_code, alfa, n):

    # PANEL program - 2D panelna metoda, profil u nestlačivoj struji.
    # funkcija PANEL određuje raspodjelu gustoce vrtloga na cvorovima
    # segmenata profila, uz pretpostavku linearne promjene gustoce duž
    # jednog segmenta. Temeljem tog rezultata određuje se i koeficijent
    # tlaka u segmentima (donjaci i gornjaci profila).
    # Program koristi funkcije:
    #   - NACA4.py za generiranje koordinata profila NACA serije 4,
    #   - C1C2.py pomoćna funkcija, generiranje matrica Cn1, Cn2, Ct1 i Ct2.

    m = n * 2  # ukupni broj segmenata na profilu
    # Profil

    #xk, zk = naca_4_5_code(2, 4, 8, n)  # Implementirajte funkciju naca4
    xk, zk =get_points_for_CPACS_NACA_4_5(naca_4_5_code,n)
    # Priprema varijabli
    Cn1 = np.zeros((m, m))
    Cn2 = np.zeros((m, m))
    Ct1 = np.zeros((m, m))
    Ct2 = np.zeros((m, m))
    An = np.zeros((m, m))
    At = np.zeros((m, m))
    B = np.zeros(m)
    D = np.zeros(m)

    # Kontrolne tocke
    xj = np.zeros(m)
    zj = np.zeros(m)
    for j in range(m):
        xj[j] = (xk[j] + xk[j + 1]) / 2
        zj[j] = (zk[j] + zk[j + 1]) / 2

    # Segmenti
    S = np.zeros(m)
    delta = np.zeros(m)
    for k in range(m):
        S[k] = np.sqrt((xk[k + 1] - xk[k]) ** 2 + (zk[k + 1] - zk[k]) ** 2)
        delta[k] = np.arctan2(zk[k + 1] - zk[k], xk[k + 1] - xk[k])

    # Matrice C
    for j in range(m):
        for k in range(m):
            Cn1[j, k], Cn2[j, k], Ct1[j, k], Ct2[j, k] = c1c2(xj[j], zj[j], delta[j], xk[k], zk[k], delta[k], S[k])
            if k == j:
                Cn1[j, k] = 1
                Cn2[j, k] = -1
                Ct1[j, k] = np.pi / 2
                Ct2[j, k] = np.pi / 2

    An[:, 0] = Cn1[:, 0] - Cn2[:, m - 1]
    At[:, 0] = Ct1[:, 0] - Ct2[:, m - 1]
    for k in range(1, m):
        An[:, k] = Cn1[:, k] + Cn2[:, k - 1]
        At[:, k] = Ct1[:, k] + Ct2[:, k - 1]

    for j in range(m):
        B[j] = np.sin(alfa - delta[j])
        D[j] = np.cos(delta[j] - alfa)

    # Vektor gustoce vrtloga u cvorovima segmenata
    Gp = np.linalg.inv(An) @ B

    # Vektori normirane brzine i koeficijenta tlaka u cvorovima segmenata
    Vc = D + At @ Gp
    Cp = 1 - Vc**2
    Gp = np.concatenate((Gp, [-Gp[0]]))
    dx = S * np.cos(delta)
    dz = S * np.sin(delta)
    dfz = Cp * dx
    dfx = -Cp * dz
    cz = - np.sum(dfz)
    cx = - np.sum(dfx)
    cl = cz * np.cos(alfa) - cx * np.sin(alfa)
    lw_c = np.sum(S)
    return cl, lw_c

def get_naca_4_5_airfoils_data(naca_4_5_code,n):
    cl0,lw_c = panel(naca_4_5_code,0,n)
    cl4, lw_c = panel(naca_4_5_code, np.radians(4), n)
    a = (cl4-cl0)/(np.radians(4))
    return cl0,a,lw_c
def plot_airfoils(airfoils, show_points):
    h = []
    label = []
    fig, ax = plt.subplots()
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('z')
    ax.grid(True)
    plotType = '-'
    if show_points:
        plotType += 'o'
    for key, data in airfoils.items():
        h1, = plt.plot(data[0], data[1], plotType, linewidth=1)
        h.append(h1)
        label.append(key)
    plt.axis((-0.02, 1.02) + plt.axis()[2:])
    ax.legend(h, label)
    plt.show()

if __name__ == "__main__":
    airfoils = {}
    m = 100
    xk, zk = naca_4_5_code(2, 4, 8, 50)
    airfoils['2415_jan'] = [xk, zk, naca_4_5_code, "NACA 4 digit airfoil with halfcosine spacing between points"]
    xk1, zk1 = get_points_for_CPACS_NACA_4_5('2408', m)
    airfoils['2415_pp'] = [xk1, zk1, naca_4_5_code, "NACA 4 digit airfoil with halfcosine spacing between points"]
    #xk, zk = naca_4_5_code(2, 4, 8, 50)
    #xk, zk = get_points_for_CPACS_NACA_4_5(naca_4_5_code, m)
    #airfoils[naca_4_5_code] = [xk, zk, naca_4_5_code, "NACA 4 digit airfoil with halfcosine spacing between points"]
    plot_airfoils(airfoils,True)
    alpha=8
    cl =panel('2408',alpha/57.3,50)
    #print('Gp =', Gp)
    #print('Cp = ', Cp)

    print('cl =', cl)
    print('alfa =', alpha)

