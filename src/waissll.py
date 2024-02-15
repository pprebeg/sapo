import numpy as np
import time


def get_waissll_points_trapezoidal(p_0, b, c0, ct, L0, alpha_pos, i_r, i_t, phei, a_r, a_t, m,sym='none'):
    i_values = np.arange(m)
    # Kontrolne tocke
    y_all = np.linspace(-b / 2, b / 2, m + 1)
    y1 = y_all[0:m]
    dy=(y_all[1]-y_all[0])/2.0
    ykt = y1 + dy
    #ykt = -b/2 + (i_values + 0.5) * b / m
    ckt = c0 - (c0 - ct) * np.abs(ykt) / (b / 2)
    a0 = ((2 * (a_t - a_r)) / b) * ykt + a_r
    h = (a0 * ckt) / (4 * np.pi)
    xkt = 0.25 * ckt + h + np.abs(ykt) * np.tan(L0)
    zkt = np.abs(ykt) * np.tan(phei)
    i = 2 * ykt * (i_t - i_r) / b + i_r
    nj = np.column_stack([np.sin(i + alpha_pos), np.ones(m) * (-np.sin(phei)), np.cos(i + alpha_pos) + np.cos(phei)])
    nj = nj/np.linalg.norm(nj,axis=1)[:, np.newaxis]

    # Hvatiste sile:

    yf = ykt
    cf = ckt
    xf = xkt - h
    zf = zkt

    # Lijevi vrh vrtloga:


    c1 = c0 - (c0 - ct) * abs(y1) / (b / 2)
    x1 = 0.25 * c1 + abs(y1) * np.tan(L0)
    z1 = np.abs(y1) * np.tan(phei)

    # Desni vrh vrtloga:
    y2 = y_all[1:m+1]
    c2 = c0 - (c0 - ct) * np.abs(y2) / (b / 2)
    x2 = 0.25 * c2 + np.abs(y2) * np.tan(L0)
    z2 = np.abs(y2) * np.tan(phei)

    p_kt = np.column_stack([xkt, ykt, zkt]) + p_0
    p_f = np.column_stack([xf, yf, zf]) + p_0
    p_1 = np.column_stack([x1, y1, z1]) + p_0
    p_2 = np.column_stack([x2, y2, z2]) + p_0

    if sym == 'none':
        pass # do nothing
    elif sym == 'x-z-plane':
        pass
    elif sym == 'x-y-plane':
        pass
    elif sym == 'y-z-plane':
        pass
    else:
        print ('Unknown symmetry type: {0}, exiting program!'.format(sym))
        exit()

    return p_kt, nj, p_f, p_1, p_2

def trag(p1, p2, pm):
    import numpy as np
    # PI-vortices locations
    ort = np.array([1, 0, 0])
    Vr1 = pm-p1
    Vr2 = pm-p2
    mr1 = np.linalg.norm(Vr1)
    mr2 = np.linalg.norm(Vr2)
    Vw1 = np.cross(ort, Vr1) / (mr1 * (mr1 - np.dot(ort, Vr1)))
    Vw2 = np.cross(ort, Vr2) / (mr2 * (mr2 - np.dot(ort, Vr2)))
    B = (Vw2 - Vw1) / (4 * np.pi)
    return B

def pivrtlog(p1, p2, pm):
    import numpy as np
    # PI-vortices locations
    ort = np.array([1, 0, 0])
    Vr1 = pm - p1
    Vr2 = pm - p2
    mr1 = np.linalg.norm(Vr1)
    mr2 = np.linalg.norm(Vr2)
    Vw1 = np.cross(ort, Vr1) / (mr1 * (mr1 - np.dot(ort, Vr1)))
    Vw2 = np.cross(ort, Vr2) / (mr2 * (mr2 - np.dot(ort, Vr2)))
    cross_product = np.cross(Vr1, Vr2)
    denominator = mr1 * mr2 * (mr1 * mr2 + np.dot(Vr1, Vr2))
    Vw0 = cross_product * (mr1 + mr2) / denominator
    B = (-Vw1 + Vw0 + Vw2) / (4 * np.pi)
    return B

def vectorized_pivrtlog(p1, p2, pm):
    ort = np.array([1, 0, 0])
    Vr1 = pm- p1
    Vr2 = pm - p2
    mr1 = np.linalg.norm(Vr1, axis=2)
    mr2 = np.linalg.norm(Vr2, axis=2)
    epsilon = 1e-10
    dot_Vr1_ort = np.sum(Vr1 * ort, axis=2)
    dot_Vr2_ort = np.sum(Vr2 * ort, axis=2)
    Vw1 = np.cross(ort, Vr1) / (mr1[..., np.newaxis] * (mr1 - dot_Vr1_ort)[..., np.newaxis] + epsilon)
    Vw2 = np.cross(ort, Vr2) / (mr2[..., np.newaxis] * (mr2 - dot_Vr2_ort)[..., np.newaxis] + epsilon)
    #B = (Vw2 - Vw1) / (4 * np.pi)
    cross_product = np.cross(Vr1, Vr2)
    denominator = mr1 * mr2 * (mr1 * mr2 + np.sum(Vr1 * Vr2, axis=2))
    Vw0 = cross_product * (mr1 + mr2)[..., np.newaxis] / denominator[..., np.newaxis]
    B = (-Vw1 + Vw0 + Vw2) / (4 * np.pi)
    return B

def vectorized_trag(p1, p2, pm):
    ort = np.array([1, 0, 0])
    Vr1 = pm - p1
    Vr2 = pm - p2
    mr1 = np.linalg.norm(Vr1, axis=2)
    mr2 = np.linalg.norm(Vr2, axis=2)
    epsilon = 1e-10
    dot_Vr1_ort = np.sum(Vr1 * ort, axis=2)
    dot_Vr2_ort = np.sum(Vr2 * ort, axis=2)
    Vw1 = np.cross(ort, Vr1) / (mr1[..., np.newaxis] * (mr1 - dot_Vr1_ort)[..., np.newaxis] + epsilon)
    Vw2 = np.cross(ort, Vr2) / (mr2[..., np.newaxis] * (mr2 - dot_Vr2_ort)[..., np.newaxis] + epsilon)
    B = (Vw2 - Vw1) / (4 * np.pi)
    return B

def calc_tapered_wing(p_kt, nj, p_f, p_1, p_2, S, A):
    m, _ = np.shape(p_kt)
    # import geometrija as ge
    # Determine unknown circulation distribution
    b = np.sqrt(A * S)
    # Determine unknown circulation distribution
    B = np.zeros((m, m))
    E = np.ones(m)
    D = np.zeros((m, m))
    g = np.zeros(m)
    start_time = time.time()
    # Rješavanje sustava:
    for j in range(m):
        for i in range(m):
            BV = pivrtlog(p_1[i], p_2[i], p_kt[j])
            B[j, i] = np.dot(BV, nj[j, :])
    G = -np.linalg.solve(B, E / (b / 2))

    # Izračun koeficijenta uzgona:
    CLa = A * np.sum(G) / m
    end_time = time.time()
    print("Time for original version CLa:{:.4f} seconds".format(end_time - start_time))
    print('Cla_orig=', CLa)
    # Vectorization
    start_time = time.time()
    p_kt_vec = p_kt[:, np.newaxis, :]
    p_f_vec = p_f[:, np.newaxis, :]
    BV_vec = vectorized_pivrtlog(p_1, p_2, p_kt_vec)
    B_vec = np.einsum('ijk,ik->ij', BV_vec, nj)

    G_vec = -np.linalg.solve(B_vec, E / (b / 2))
    CLa_vec = A * np.sum(G_vec) / m
    end_time = time.time()
    print("Time for vectorized version:{:.4f} seconds".format(end_time - start_time))
    print('Cla_vec=', CLa_vec)
    print('Test B_vec', np.allclose(B, B_vec, 1e-5))
    print('Test G_vec', np.allclose(G, G_vec, 1e-5))
    print('Test Cla_vec', np.allclose(CLa, CLa_vec, 1e-5))

    # Izračun koeficijenta induciranog otpora:
    start_time = time.time()
    for j in range(m):
        for i in range(m):
            if (p_1[i,1]+p_2[i,1])*p_f[j,1] > 0:
                DV = G[i] * (trag(p_1[i], p_2[i], p_f[j]))
                pass
            else:
                DV = G[i] * (pivrtlog(p_1[i], p_2[i], p_f[j]))
                pass
            D[j, i] = DV[2]
        g[j] = -(b / 2) * np.sum(D[j, :])
        pass
    suma = 0
    for j in range(m):
        suma = suma + g[j] * G[j]
    CDa2 = A * suma / m
    end_time = time.time()
    print("Time for original version CDa2:{:.4f} seconds".format(end_time - start_time))
    print('CDa2_orig=', CDa2)
    # Vectorization
    start_time = time.time()
    DV_vec = vectorized_trag(p_1, p_2, p_f_vec)
    DV_vec = np.einsum('ijk,j->ijk', DV_vec, G_vec)
    D_vec = DV_vec[..., 2]
    g_vec = -(b / 2) * np.sum(D_vec, axis=1)
    sum = np.sum(g_vec * G)
    CDa2_vec = A * sum / m
    end_time = time.time()
    print("Time for vectorized version CDa2:{:.4f} seconds".format(end_time - start_time))
    print('CDa2_vec=', CDa2_vec)
    print('Test D_vec', np.allclose(D, D_vec, 1e-5))
    print('Test g_vec', np.allclose(g, g_vec, 1e-5))
    print('Test CDa2_vec', np.allclose(CDa2, CDa2_vec, 1e-5))
    return CLa_vec,CDa2_vec

def calc_CLa_CDa2(p_kt, nj, p_f, p_1, p_2,A,S):
    m,_ =np.shape(p_kt)
    # Determine unknown circulation distribution
    b = np.sqrt(A * S)
    # Determine unknown circulation distribution
    E = np.ones(m)
    # Vectorization
    p_kt_vec = p_kt[:, np.newaxis, :]
    p_f_vec = p_f[:, np.newaxis, :]
    BV_vec = vectorized_pivrtlog(p_1, p_2, p_kt_vec)
    B_vec = np.einsum('ijk,ik->ij', BV_vec, nj)

    G_vec = -np.linalg.solve(B_vec, E / (b / 2))
    CLa_vec = A * np.sum(G_vec) / m

     # Vectorization
    DV_vec = vectorized_trag(p_1, p_2, p_f_vec)
    DV_vec = np.einsum('ijk,j->ijk', DV_vec, G_vec)
    D_vec = DV_vec[..., 2]
    g_vec = -(b / 2) * np.sum(D_vec, axis=1)
    sum = np.sum(g_vec * G_vec)
    CDa2_vec = A * sum / m

    ci = np.linalg.norm(p_kt-p_f,axis=1) #3/4c -1/4c = 1/2c
    ci= ci*2.0
    dbi = np.linalg.norm(p_2-p_1,axis=1)
    gama_i_db_i = G_vec*dbi* (b / 2)
    gama_i_db_i_ni = np.einsum('ij,i->ij', nj, gama_i_db_i)

    return CLa_vec,CDa2_vec,gama_i_db_i_ni
def test_matlab_geometry():
    alpha = 4/57.3
    vinf = 35
    rho = 1.225
    p_0 =np.zeros(3)
    b = 10.0
    c_r = 2.0  # root chord
    c_t = 2.0  # tip chord
    L0 = 45/ 57.3
    alpha_pos = 0 / 57.3  # postavni kut krila
    i_r = 0 / 57.3  # kut uvijanja u korjenu krila
    i_t = 0 / 57.3  # kut uvijanja u vrhu krila
    a0_r = 2*np.pi # NACA 2415, korijen krila
    a0_t = 2*np.pi # NACA 2408, vrh krila
    phei = 0 / 57.3  # dihedral
    m= 8

    sref = b*(c_r+c_t)/2.0
    A=b**2/sref
    p_kt, nj, p_f, p_1, p_2 = get_waissll_points_trapezoidal(p_0, b, c_r, c_t, L0, alpha_pos, i_r, i_t, phei, a0_r, a0_t, m,sym='none')
    CLa_vec,CDa2_vec= calc_tapered_wing(p_kt, nj, p_f, p_1, p_2, sref, A)
    L=0.5*rho*vinf**2*CLa_vec*alpha
    D = 0.5*rho*vinf**2*CDa2_vec*alpha**2
    print('L =',L)
    print('D =',D)
if __name__ == "__main__":
    test_matlab_geometry()

