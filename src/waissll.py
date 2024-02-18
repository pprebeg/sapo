import nt

import numpy as np
import time



def get_unit_vector_mx3(vec):
    unit_vec = vec / np.linalg.norm(vec, axis=1)[:, np.newaxis]
    return unit_vec

def get_quantity_distribution_across_x(q_r,q_t,x_r,x_t,x):
    if np.isclose(q_t,q_r) or np.isclose(x_t,x_r):
        q= np.full_like(x,q_r)
    else:
        q= (q_t-q_r)/(x_t-x_r)*x + q_r
    return q

def get_waissll_geometry_segment(p_0, b, c0, ct, L0,i_r, i_t, phei, a_r, a_t, m, sym=False, xrot=0.0):
    # Kontrolne tocke
    y_all_sym = np.linspace(0.0, b, m + 1)
    c_all = c0 - (c0 - ct) * np.abs(y_all_sym) /b
    x_all = 0.25 * c_all + np.abs(y_all_sym) * np.tan(L0)
    z_all = np.abs(y_all_sym) * np.tan(phei)
    # Lijevi vrh vrtloga:
    y1 = y_all_sym[0:m]
    x1 = x_all[0:m]
    z1 = z_all[0:m]
    # Desni vrh vrtloga:
    y2 = y_all_sym[1:m+1]
    x2 = x_all[1:m+1]
    z2 = z_all[1:m+1]
    # Kontrolne točke
    dy=(y_all_sym[1]-y_all_sym[0])/2.0
    ykt = y1 + dy
    ckt = c0 - (c0 - ct) * np.abs(ykt) / b
    a0 = ((a_t - a_r) / b) * ykt + a_r
    h = (a0 * ckt) / (4 * np.pi)
    xkt = 0.25 * ckt + h + np.abs(ykt) * np.tan(L0)
    zkt = np.abs(ykt) * np.tan(phei)
    i = ykt * (i_t - i_r) / b + i_r
    nj = np.column_stack([np.sin(i), np.ones(m) * (-np.sin(phei)), np.cos(i) + np.cos(phei)])
    nj = get_unit_vector_mx3(nj)
    c_i = np.column_stack([np.cos(i), np.zeros(m), np.sin(i)])
    c_i_e = get_unit_vector_mx3(c_i)

    # Hvatiste sile:
    yf = ykt
    xf = xkt - h
    zf = zkt

    # Compact coordinates into point vectors
    p_kt = np.column_stack([xkt, ykt, zkt]) + p_0
    p_f = np.column_stack([xf, yf, zf]) + p_0
    p_1 = np.column_stack([x1, y1, z1]) + p_0
    p_2 = np.column_stack([x2, y2, z2]) + p_0

    c_i = ckt

    if sym :
        sym_matrix = np.array([1, -1, 1])
        p_kt_sym = p_kt*sym_matrix
        p_f_sym = p_f*sym_matrix
        p_1_sym = p_2*sym_matrix
        p_2_sym = p_1*sym_matrix
        nj_sym = nj*sym_matrix


        p_kt = np.concatenate((p_kt_sym, p_kt))
        p_f = np.concatenate((p_f_sym, p_f))
        p_1 = np.concatenate((p_1_sym, p_1))
        p_2 = np.concatenate((p_2_sym, p_2))
        nj = np.concatenate((nj_sym,nj))
        c_i_e = np.concatenate((c_i_e, c_i_e))
        c_i = np.concatenate((c_i, c_i)) #chord

    e_b_i = get_unit_vector_mx3(np.cross(nj,c_i_e))
    db_i = np.einsum('ij,ij->i',p_2 - p_1, e_b_i)

    if not np.isclose(xrot,0.0) :
        pass # rotate

    return p_kt, nj, p_f, p_1, p_2, db_i,c_i

def get_waissll_points_trapezoidal(b, c0, ct, L0, i_r, i_t, phei, a_r, a_t, m):
    # Kontrolne tocke
    y_all = np.linspace(-b / 2, b / 2, m + 1)
    c_all = c0 - (c0 - ct) * np.abs(y_all) / (b / 2)
    x_all = 0.25 * c_all + np.abs(y_all) * np.tan(L0)
    z_all = np.abs(y_all) * np.tan(phei)
    # Lijevi vrh vrtloga:
    y1 = y_all[0:m]
    x1 = x_all[0:m]
    z1 = z_all[0:m]
    # Desni vrh vrtloga:
    y2 = y_all[1:m+1]
    x2 = x_all[1:m+1]
    z2 = z_all[1:m+1]
    # Kontrolne točke
    dy=(y_all[1]-y_all[0])/2.0
    ykt = y1 + dy
    ckt = c0 - (c0 - ct) * np.abs(ykt) / (b / 2)
    a0 = ((2 * (a_t - a_r)) / b) * ykt + a_r
    h = (a0 * ckt) / (4 * np.pi)
    xkt = 0.25 * ckt + h + np.abs(ykt) * np.tan(L0)
    zkt = np.abs(ykt) * np.tan(phei)
    i = 2 * ykt * (i_t - i_r) / b + i_r
    nj = np.column_stack([np.sin(i), np.ones(m) * (-np.sin(phei)), np.cos(i) + np.cos(phei)])
    nj = get_unit_vector_mx3(nj)
    c_i = np.column_stack([np.cos(i), np.zeros(m), np.sin(i)])
    c_i_e = get_unit_vector_mx3(c_i)

    # Hvatiste sile:
    yf = ykt
    xf = xkt - h
    zf = zkt

    c_i = ckt

    p_kt = np.column_stack([xkt, ykt, zkt])
    p_f = np.column_stack([xf, yf, zf])
    p_1 = np.column_stack([x1, y1, z1])
    p_2 = np.column_stack([x2, y2, z2])

    e_b_i = get_unit_vector_mx3(np.cross(nj, c_i_e))
    db_i = np.einsum('ij,ij->i', p_2 - p_1, e_b_i)

    return p_kt, nj, p_f, p_1, p_2, db_i,c_i

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
    Vw0 = cross_product * (mr1 + mr2)[..., np.newaxis] / (denominator[..., np.newaxis]+ epsilon)
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

def calc_tapered_wing(p_kt, nj, p_f, p_1, p_2,db_i, S, A,V_vec):
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
    #Circulation -
    rhs = np.einsum('j,ij->i', V_vec, nj)
    Gama_i = -np.linalg.solve(B_vec, rhs)
    Gama_i_e = get_unit_vector_mx3(np.cross(nj, V_vec))
    Gama_db_i_vec = np.einsum('i,ij->ij', Gama_i*db_i, Gama_i_e)
    L_i_e = get_unit_vector_mx3(np.cross(V_vec,Gama_i_e))



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
    cond_LR =(p_1[:, np.newaxis, 1] + p_2[:, np.newaxis, 1]) * p_f[np.newaxis, :, 1] > 0
    Dijk = np.where(cond_LR[..., np.newaxis],
                     vectorized_trag(p_1, p_2, p_f_vec),
                     vectorized_pivrtlog(p_1, p_2, p_f_vec))
    DV_vec = np.einsum('ijk,j->ijk', Dijk, G_vec)
    D_vec = DV_vec[..., 2]
    g_vec = -(b / 2) * np.sum(D_vec, axis=1)
    sum = np.sum(g_vec * G)
    CDa2_vec = A * sum / m
    end_time = time.time()
    #Kutta Joukowski induced drag
    w_ijk=np.einsum('ijk,j->ijk', Dijk, Gama_i)
    w_ij = w_ijk[..., 2]
    w_i = np.sum(w_ij, axis=1)
    w_i_e = - L_i_e
    #D_i_e = get_unit_vector_mx3(np.cross(w_i_e,Gama_i_e))
    w_i_vec = np.einsum('i,ij->ij', w_i, w_i_e)
    print("Time for vectorized version CDa2:{:.4f} seconds".format(end_time - start_time))
    print('CDa2_vec=', CDa2_vec)
    print('Test D_vec', np.allclose(D, D_vec, 1e-5))
    print('Test g_vec', np.allclose(g, g_vec, 1e-5))
    print('Test CDa2_vec', np.allclose(CDa2, CDa2_vec, 1e-5))
    return CLa_vec,CDa2_vec,Gama_db_i_vec,w_i_vec

def calc_joukowski_forces_weissinger_lifting_line(p_kt, nj, p_f, p_1, p_2, db_i, V_vec, rho):
    m, _ = np.shape(p_kt)
    # Determine unknown circulation distribution
    p_kt_vec = p_kt[:, np.newaxis, :]
    p_f_vec = p_f[:, np.newaxis, :]
    BV_vec = vectorized_pivrtlog(p_1, p_2, p_kt_vec)
    B_vec = np.einsum('ijk,ik->ij', BV_vec, nj)
    #Circulation Gama
    rhs = np.einsum('j,ij->i', V_vec, nj)
    Gama_i = -np.linalg.solve(B_vec, rhs)
    Gama_i_e = get_unit_vector_mx3(np.cross(nj, V_vec))
    Gama_db_i_vec = np.einsum('i,ij->ij', Gama_i*db_i, Gama_i_e)
    L_i_e = get_unit_vector_mx3(np.cross(V_vec,Gama_i_e))

    # Induced velocity w
    cond_LR =(p_1[:, np.newaxis, 1] + p_2[:, np.newaxis, 1]) * p_f[np.newaxis, :, 1] > 0
    Dijk = np.where(cond_LR[..., np.newaxis],
                     vectorized_trag(p_1, p_2, p_f_vec),
                     vectorized_pivrtlog(p_1, p_2, p_f_vec))
    w_ijk=np.einsum('ijk,j->ijk', Dijk, Gama_i)
    w_ij = w_ijk[..., 2]
    w_i = np.sum(w_ij, axis=1)
    e_w_i = - L_i_e
    w_i_vec = np.einsum('i,ij->ij', w_i, e_w_i)

    # Calculate Kutta Joukowski forces vector in each segment

    L_i_vec = rho * np.cross(V_vec, Gama_db_i_vec)
    D_i_vec = rho * np.cross(w_i_vec, Gama_db_i_vec)

    return L_i_vec, D_i_vec

def plot_planform(p_kt, p_f, p_1, p_2):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(p_kt[:, 0], p_kt[:, 1], p_kt[:, 2],'+', label='kt')
    ax.plot(p_f[:, 0], p_f[:, 1], p_f[:, 2],'.', label='f')
    ax.plot(p_1[:, 0], p_1[:, 1], p_1[:, 2], '.', label='1')
    ax.plot(p_2[:, 0], p_2[:, 1], p_2[:, 2], '.', label='2')
    ax.set_xlabel('x, m')
    ax.set_ylabel('y, m')
    ax.set_zlabel('z, m')
    ax.legend()
    plt.show()

def test_matlab_geometry():
    alpha = 4/57.3
    V_inf = 15
    V_inf_x = V_inf * np.cos(alpha)
    V_inf_z = V_inf * np.sin(alpha)
    V_inf_vec = np.array([V_inf_x, 0, V_inf_z])
    rho = 1.225
    p_0 =np.zeros(3)
    b = 10.0
    c_r = 2.0  # root chord
    c_t = 2.0  # tip chord
    L0 = 45/ 57.3
    i_r = 0 / 57.3  # postavni kut u korjenu krila
    i_t = 0 / 57.3  # postavni kut u vrhu krila
    a0_r = 2*np.pi # NACA 2415, korijen krila
    a0_t = 2*np.pi # NACA 2408, vrh krila
    phei = 0 / 57.3  # dihedral
    m= 8

    sref = b*(c_r+c_t)/2.0
    A=b**2/sref
    p_kt_i, e_n_kt_i, p_f_i, p_1_i, p_2_i,db_i,c_i = get_waissll_points_trapezoidal(b, c_r, c_t, L0, i_r, i_t, phei, a0_r, a0_t, m)
    CLa_vec,CDa2_vec,Gama_db_i_vec,w_i_vec= calc_tapered_wing(p_kt_i, e_n_kt_i, p_f_i, p_1_i, p_2_i,db_i, sref, A,V_inf_vec)
    L=0.5*rho*V_inf**2*CLa_vec*alpha*sref
    D = 0.5*rho*V_inf**2*CDa2_vec*alpha**2*sref
    FL = rho* np.cross(V_inf_vec,Gama_db_i_vec)
    FD = rho * np.cross(w_i_vec, Gama_db_i_vec)
    FLsum = np.sum(FL,axis=0)
    FDsum = np.sum(FD, axis=0)
    print('L =',L)
    print('D =',D)
    print('FL =', FLsum)
    print('FL_norm =', np.linalg.norm(FLsum))
    print('FD =', FDsum)
    print('FD_norm =', np.linalg.norm(FDsum))


def test_symetric_segment_geometry():
    alpha = 4/57.3
    V_inf = 15
    V_inf_x = V_inf * np.cos(alpha)
    V_inf_z = V_inf * np.sin(alpha)
    V_inf_vec = np.array([V_inf_x, 0, V_inf_z])
    rho = 1.225
    p_0 =np.zeros(3)
    b = 2.5
    c_r = 2.0  # root chord
    c_t = 2.0  # tip chord
    L0 = 45/ 57.3
    i_r = 0 / 57.3  # postavni kut u korjenu krila
    i_t = 0 / 57.3  # postavni kut u vrhu krila
    a0_r = 2*np.pi  # NACA 2415, korijen krila
    a0_t = 2*np.pi  # NACA 2408, vrh krila
    phei = 0 / 57.3  # dihedral
    m = 40

    sref = b*(c_r+c_t)/2.0
    A = b**2/sref


    #First segent
    p_kt_i, e_n_kt_i, p_f_i, p_1_i, p_2_i,db_i,c_i = get_waissll_geometry_segment(p_0, b, c_r, c_t, L0, i_r, i_t, phei, a0_r, a0_t, m, True)
    # Second segment
    p_0 = np.array([b*np.tan(L0), b, 0])
    p_kt_i_, e_n_kt_i_, p_f_i_, p_1_i_, p_2_i_,db_i_,c_i_ = get_waissll_geometry_segment(p_0, b, c_r, c_t, L0, i_r, i_t, phei, a0_r, a0_t, m, True)

    p_kt_i = np.concatenate((p_kt_i, p_kt_i_))
    p_f_i = np.concatenate((p_f_i, p_f_i_))
    p_1_i = np.concatenate((p_1_i, p_1_i_))
    p_2_i = np.concatenate((p_2_i, p_2_i_))
    e_n_kt_i = np.concatenate((e_n_kt_i, e_n_kt_i_))
    db_i = np.concatenate((db_i, db_i_))
    c_i = np.concatenate((c_i, c_i_))

    plot_planform(p_kt_i,p_f_i,p_1_i,p_2_i)
    CLa_vec,CDa2_vec,Gama_db_i_vec,w_i_vec= calc_tapered_wing(p_kt_i, e_n_kt_i, p_f_i, p_1_i, p_2_i,db_i, sref, A,V_inf_vec)
    L = 0.5*rho*V_inf**2*CLa_vec*alpha*sref
    D = 0.5*rho*V_inf**2*CDa2_vec*alpha**2*sref
    FL = rho * np.cross(V_inf_vec, Gama_db_i_vec)
    FD = rho * np.cross(w_i_vec, Gama_db_i_vec)
    FLsum = np.sum(FL,axis=0)
    FDsum = np.sum(FD, axis=0)
    print('Lsym =', L)
    print('Dsym =', D)
    print('FLsym =', FLsum)
    print('FL_sym =', np.linalg.norm(FLsum))
    print('FDsym =', FDsum)
    print('FD_sym =', np.linalg.norm(FDsum))
    print('calc_joukowski_forces_weisinger_lifting_line')
    FL,FD = calc_joukowski_forces_weissinger_lifting_line(p_kt_i, e_n_kt_i, p_f_i, p_1_i, p_2_i, db_i, V_inf_vec, rho)
    FLsum = np.sum(FL,axis=0)
    FDsum = np.sum(FD, axis=0)
    print('FL =', FLsum)
    print('FL_norm =', np.linalg.norm(FLsum))
    print('FD =', FDsum)
    print('FD_norm =', np.linalg.norm(FDsum))


if __name__ == "__main__":
    test_matlab_geometry()
    test_symetric_segment_geometry()

