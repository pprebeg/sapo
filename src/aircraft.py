from typing import List
import numpy as np
from src.waissll import get_waissll_points_trapezoidal, calc_CLa_CDa2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from enum import Enum
from src.aero import calc_air_density_speedsound_ICAO, calc_mach, calc_reynolds, calc_friction_drag_constant
from src.panel import get_naca_4_5_airfoils_data
from typing import Dict

class Symmetry(Enum):
    NO_SYMMETRY = 'none'
    INHERITS = 'inherits'
    X_Z_PLANE = 'x-z-plane'
    X_Y_PLANE = 'x-y-plane'
    Y_Z_PLANE = 'y-z-plane'

def getMinMax(numList,inMin = None, inMax = None):
    if inMin == None:
        xmax=numList[0]
        xmin = numList[0]
    else:
        xmax = inMax
        xmin = inMin

    for x in numList:
        if  x > xmax:
            xmax=x
        if  x < xmin:
            xmin=x

    return xmin,xmax

class FlightCondition:
    def __init__(self,v_ms,h_m,alpha_deg,wf=1.0):
        self._V_inf = v_ms # m/s
        self._h_m = h_m # m
        self._alpha_deg = alpha_deg  # °
        self._rho_h, self._a_h, self._mu =  calc_air_density_speedsound_ICAO(h_m)
        self._dyn_press = 0.5 * self._rho_h * self._V_inf ** 2.0
        self._wf = wf #weight factor of flight condition
        self._L = 0.0
        self._D = 0.0
        self._N = 0.0
        self._sina = np.sin(self.alpha_rad)
        self._cosa = np.cos(self.alpha_rad)

        self._force_vectors:Dict[str,np.ndarray] = {}
        self._force_locations:Dict[str,np.ndarray]  = {}
        self._moment_vectors:Dict[str,np.ndarray]  = {}

    @property
    def wf(self):
        return self._wf
    @property
    def alpha_deg(self):
        return self._alpha_deg

    @alpha_deg.setter
    def alpha_deg(self, value):
        self._alpha_deg = value
    @property
    def alpha_rad(self):
        return np.radians(self._alpha_deg)

    @property
    def V_inf(self):
        return  self._V_inf

    @property
    def V_inf_vec(self):
        V_inf_x = self.V_inf * self._cosa
        V_inf_z = self.V_inf * self._sina
        V_inf_vec = np.array([V_inf_x, 0, V_inf_z])
        return V_inf_vec

    def calc_reynolds(self,c):
        Re = self._rho_h * self._V_inf *c/self._mu
        return Re

    def clear_forces_and_moments(self):
        self._force_vectors.clear()
        self._force_locations.clear()
        self._moment_vectors.clear()

    def add_force_vectors_and_locations(self,key,force_vectors,force_location):
        self._force_vectors[key] = force_vectors
        self._force_location[key] = force_vectors


    def calc_aero_forces(self,wing:Wing, CLa, CDa2):
        CD0w = wing.calc_friction_drag_constant(self._mu,self._rho_h, self._V_inf,self.Ma)
        CL0w = wing.calc_CL0()
        dp_S = self._dyn_press * wing.Sref
        C_L = CL0w + CLa*self.alpha_rad
        C_D = CD0w + CDa2*self.alpha_rad**2.0
        self._L = dp_S*C_L
        self._D = dp_S*C_D
        self._N = self._L * np.cos(self.alpha_rad) * np.cos(wing.phei)

    def calc_distributed_aero_forces(self,Sref,CLa,CL0,CDa2,CD0,phei,Gama_i,w_i_Gamai,L_unit_vec):
        dp_S = self._dyn_press * Sref
        CL0=0.0
        C_L = CL0 + CLa*self.alpha_rad
        C_D = CD0 + CDa2*self.alpha_rad**2.0
        L = dp_S*C_L
        D = dp_S*C_D
        N = L * np.cos(self.alpha_rad) * np.cos(phei)


        Lift_vec= self._rho_h*self.V_inf*Gama_i
        Lift=np.sum(Lift_vec,axis=0)
        Drag_vec = self._rho_h*w_i_Gamai
        Drag = np.sum(Drag_vec, axis=0)
        pass


    @property
    def D(self):
        return self._D

    @property
    def L(self):
        return self._L

    @property
    def N(self):
        return self._N

    @property
    def Ma(self):
        return calc_mach(self.V_inf,self._a_h)

    @property
    def Re(self):
        return calc_reynolds(self._mu, self._rho_h,self._c, self._V_inf)
    def get_info(self):
        msg = 'FC: Vcr = {0:.2f} m/s; h = {1:.0f} m; alpha = {2:.2f} °; '.format(self._V_inf, self._h_m, self._alpha_deg)
        msg += 'L = {0:.2f} N; D = {1:.2f} N;'.format(self._L, self._D)
        return msg

class Segment:
    def __init__(self, b, c0, ct, L0, alpha_pos, i_r, i_t, phei, naca_r, naca_t,p_0 = np.zeros(3),sym = Symmetry.NO_SYMMETRY):
        # all angels are in radians
        self._b = b
        self._c0 = c0
        self._ct = ct
        self._L0 = L0
        self._alpha_pos = alpha_pos
        self._i_r = i_r
        self._i_t = i_t
        self._phei = phei
        self._cl0_r, self._a_r, self._lw_c_r = get_naca_4_5_airfoils_data(naca_r,50)
        self._cl0_t, self._a_t, self._lw_c_t = get_naca_4_5_airfoils_data(naca_t,50)
        self._p_0 = p_0
        self._sym = sym

    @property
    def lw_c(self):
        return (self._lw_c_r+self._lw_c_t)/2.0

    @property
    def CL_0(self):
        return (self._cl0_r + self._cl0_t) / 2.0
    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        self._b = value
    @property
    def c0(self):
        return self._c0

    @c0.setter
    def c0(self, value):
        self._c0 = value

    @property
    def ct(self):
        return self._ct

    @ct.setter
    def ct(self, value):
        self._ct = value
    @property
    def L0(self):
        return self._L0

    @L0.setter
    def L0(self, value):
        self._L0 = value
    @property
    def alpha_pos (self):
        return self._alpha_pos

    @alpha_pos.setter
    def alpha_pos(self, value):
        self._alpha_pos = value
    @property
    def i_r(self):
        return self._i_r

    @i_r.setter
    def i_r(self, value):
        self._i_r = value

    @property
    def i_t(self):
        return self._i_t

    @i_t.setter
    def i_t(self, value):
        self._i_t = value

    @property
    def phei(self):
        return self._phei

    @phei.setter
    def phei(self, value):
        self._phei = value

    @property
    def p_0(self):
        return self._p_0

    @p_0.setter
    def p_0(self, value):
        self._p_0 = value

    def get_waissll_points(self,m):
        wll_pts = get_waissll_points_trapezoidal(
            self._p_0, self._b, self._c0, self._ct, self._L0, self._alpha_pos,
            self._i_r, self._i_t, self._phei, self._a_r, self._a_t, m)
        return wll_pts

    @property
    def Sref(self):
        sref = self._b*(self._c0+self._ct)/2.0
        return sref

    @property
    def A(self):
        return self._b**2/self.Sref

    def get_segment_points(self):
        vRootLE = self._p_0
        b_2 = self._b /2
        cr = self._c0
        ct = self. _ct
        sweep = self._L0
        dihedral = self._phei
        xrot = 0.0 #?

        xLEt_1 = np.tan(sweep) * b_2
        zLEt_1 = np.tan(dihedral) * b_2
        dvLERootTip = np.array([xLEt_1, b_2, zLEt_1])
        #l_LE = np.linalg.norm(dvLERootTip)
        vTipLE = np.add(vRootLE, dvLERootTip)
        dvRootLETE = np.array([cr, 0, 0])
        dvTipLETE = np.array([ct, 0, 0])
        vRootTE = np.add(vRootLE, dvRootLETE)
        vTipTE = np.add(vTipLE, dvTipLETE)
        if xrot != 0.0:
            sifi = np.sin(xrot)
            cofi = np.cos(xrot)
            rotMat = np.zeros((3, 3))
            rotMat[0, 0] = 1
            rotMat[1, 1] = cofi
            rotMat[1, 2] = -sifi
            rotMat[2, 1] = sifi
            rotMat[2, 2] = cofi
            vTipLE = rotMat.dot(vTipLE)
            vTipTE = rotMat.dot(vTipTE)
            vRootLE = rotMat.dot(vRootLE)
            vRootTE = rotMat.dot(vRootTE)

        x = []
        y = []
        z = []
        x.append(vRootLE[0])
        y.append(vRootLE[1])
        z.append(vRootLE[2])
        x.append(vRootTE[0])
        y.append(vRootTE[1])
        z.append(vRootTE[2])
        x.append(vTipTE[0])
        y.append(vTipTE[1])
        z.append(vTipTE[2])
        x.append(vTipLE[0])
        y.append(vTipLE[1])
        z.append(vTipLE[2])
        x.append(vRootLE[0])
        y.append(vRootLE[1])
        z.append(vRootLE[2])

        return x, y, z

    def get_segment_tip_LE_p(self):
        vRootLE = self._p_0
        b_2 = self._b
        sweep = self._L0
        dihedral = self._phei
        xrot = 0.0  # ?

        xLEt_1 = np.tan(sweep) * b_2
        zLEt_1 = np.tan(dihedral) * b_2
        dvLERootTip = np.array([xLEt_1, b_2, zLEt_1])
        p_t_LE = np.add(vRootLE, dvLERootTip)
        if xrot != 0.0:
            sifi = np.sin(xrot)
            cofi = np.cos(xrot)
            rotMat = np.zeros((3, 3))
            rotMat[0, 0] = 1
            rotMat[1, 1] = cofi
            rotMat[1, 2] = -sifi
            rotMat[2, 1] = sifi
            rotMat[2, 2] = cofi
            p_t_LE = rotMat.dot(p_t_LE)
        return p_t_LE

class LiftingBody:
    def __init__(self,name):
        self._segments:List[Segment] = []
        self._m = 80
        self._name = name

    def add_segment(self,seg):
        self._segments.append(seg)

    def clear_segments(self):
        self._segments.clear()

    def enforce_segment_LE_connection(self):
        p_t_LE = None
        for seg in self._segments:
            if p_t_LE is not None:
                seg.p_0 = p_t_LE
            p_t_LE = seg.get_segment_tip_LE_p()

    @property
    def name(self):
        return self._name

    @property
    def segments(self):
        return self._segments

    def get_waissll_points(self):
        m = self._m
        p_kt = np.empty((0,3))
        n_kt = np.empty((0, 3))
        p_f = np.empty((0,3))
        p_1 = np.empty((0,3))
        p_2 = np.empty((0,3))
        for seg in self._segments:
            seg_wll_pts = seg.get_waissll_points(m)
            s_p_kt, s_nj, s_p_f, s_p_1, s_p_2 = seg_wll_pts
            p_kt = np.concatenate((p_kt, s_p_kt))
            p_f = np.concatenate((p_f, s_p_f))
            p_1 = np.concatenate((p_1, s_p_1))
            p_2 = np.concatenate((p_2, s_p_2))
            n_kt = np.concatenate((n_kt, s_nj))
        return p_kt, n_kt, p_f, p_1, p_2

    def calculate_CLa_CDa2(self):
        p_kt, n_kt, p_f, p_1, p_2 = self.get_waissll_points()
        A = self.A
        S = self.Sref
        CLa,CDa2,Gama_i,w_i_Gamai,L_unit_vec = calc_CLa_CDa2(p_kt, n_kt, p_f, p_1, p_2,A,S,np.ones(3))
        return CLa,CDa2

    def calculate_discretized_forces(self,V_vec):
        p_kt, n_kt, p_f, p_1, p_2 = self.get_waissll_points()
        A = self.A
        S = self.Sref
        CLa,CDa2,Gama_i,w_i_Gamai,L_unit_vec = calc_CLa_CDa2(p_kt, n_kt, p_f, p_1, p_2,A,S,V_vec)
        return CLa,CDa2,Gama_i,w_i_Gamai,L_unit_vec

    def calculate_aero_forces(self,fc:FlightCondition):
        pass
    
    @property
    def Sref(self):
        sref = 0.0
        for seg in self._segments:
            srefi = seg.Sref
            sref+= srefi
        return sref

    @property
    def A(self):
        return self.b ** 2.0 / self.Sref

    @property
    def b(self):
        bb = 0.0
        for seg in self._segments:
            bi = seg.b
            bb += bi
        return bb

    @property
    def phei(self):
        fi = 0.0
        for seg in self._segments:
            fi += seg.phei
        return fi

    def get_additional_aero(self):
        cr = self.wing.segments[0].c0
        ct = self.wing.segments[0].ct
        L0 = self.wing.segments[0].L0

        CL0 = (self.wing.segments[0]._cl0_r + self.wing.segments[0]._cl0_t )/2
        return CL0,

    def calc_friction_drag_constant(self,mu,rho_h,V_inf, Ma):
        Cf = 0
        for seg in self.segments:
            Re = calc_reynolds(mu, rho_h, (seg.c0 + seg.ct) / 2.0, V_inf)
            Cfi = calc_friction_drag_constant(Re,Ma,seg.Sref,seg.c0,seg.ct,seg.L0,seg.lw_c)
            Cf +=Cfi*seg.Sref
        return Cf/self.Sref
    def calc_CL0(self):
        CL0 = 0
        for seg in self.segments:
            CL0 += seg.CL_0*seg.Sref
        return CL0/self.Sref
    def calculate_aero(self,fc):

class Wing(LiftingBody):
    def __init__(self):
        super().__init__('wing')


class HorizontalTail(LiftingBody):
    def __init__(self):
        super().__init__('htail')




class Aircraft:
    def __init__(self):
        super().__init__()
        self._wing:Wing = None
        self._htail: HorizontalTail = None
        self._W_TO = 0.0 # N
        self._flight_conditions:List[FlightCondition] = []

    def add_flight_condition(self,fc):
        self._flight_conditions.append(fc)

    def clear_flight_conditions(self):
        self._flight_conditions.clear()

    @property
    def flight_conditions(self):
        return self._flight_conditions

    @property
    def W_TO(self):
        return self._W_TO
    @W_TO.setter
    def W_TO(self, value):
        self._W_TO = value

    @property
    def wing(self):
        return self._wing

    @wing.setter
    def wing(self, value):
        self._wing = value

    def run_aero(self):
        Sref = self.wing.Sref
        phei = self.wing.segments[0].phei

        CL0 = 0.1
        CD0 = 0.02
 #       CLa,CDa2 = self.wing.calculate_CLa_CDa2()
 #       for fc in self._flight_conditions:
 #           fc.calc_aero_forces(Sref,CLa,CL0,CDa2,CD0,phei)

        for fc in self._flight_conditions:
            CLa,CDa2,Gama_i,w_i_Gamai,L_unit_vec = self.wing.calculate_discretized_forces(fc.V_inf_vec)
            fc.calc_distributed_aero_forces(Sref,CLa,CL0,CDa2,CD0,phei,Gama_i,w_i_Gamai,L_unit_vec)

    def get_info(self):
        msg = 'W_TO = {0:.2f}'.format(self._W_TO)+'\n'
        for fc in self._flight_conditions:
            msg += fc.get_info()+'\n'
        return msg

    def plot_planform(self,do_plot_wllp = True):
        xplot = []
        yplot = []
        zplot = []
        labelplot = []
        axMin = None
        axMax = None

        i_seg = 0
        for seg in self._wing.segments:
            i_seg +=1
            x, y, z = seg.get_segment_points()
            xplot.append(x)
            yplot.append(y)
            zplot.append(z)
            labelplot.append('{0} seg {1}'.format(self._wing.name,i_seg))
            axMin, axMax = getMinMax(x, axMin, axMax)
            axMin, axMax = getMinMax(y, axMin, axMax)
            axMin, axMax = getMinMax(z, axMin, axMax)



        if len(xplot) > 0:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            for i in range(len(xplot)):
                verts = [list(zip(xplot[i], yplot[i], zplot[i]))]
                poly = Poly3DCollection(verts, alpha=0.5)
                ax.add_collection3d(poly)
                ax.plot(xplot[i], yplot[i], zplot[i], label=labelplot[i])
            if do_plot_wllp:
                p_kt, n_kt, p_f, p_1, p_2 = self.wing.get_waissll_points()
                ax.plot(p_kt[:, 0], p_kt[:, 1], p_kt[:, 2],'+', label='kt')
                ax.plot(p_f[:, 0], p_f[:, 1], p_f[:, 2],'.', label='f')
                ax.plot(p_1[:, 0], p_1[:, 1], p_1[:, 2], '.', label='1')
                ax.plot(p_2[:, 0], p_2[:, 1], p_2[:, 2], '.', label='2')
            ax.set_xlabel('x, m')
            ax.set_ylabel('y, m')
            ax.set_zlabel('z, m')
            ax.legend()
            ax.auto_scale_xyz([axMin, axMax], [axMin, axMax], [axMin, axMax])
            plt.show()



def create_one_segment_wing():
    test = Aircraft()
    test.W_TO = 45 * 9.81 # N
    wing = Wing()
    test.wing = wing
    #prepare segment
    b = 10.0
    c_r = 2.0  # root chord
    c_t = 1.5  # tip chord
    c_m =(c_r+c_t)/2
    L0 = 45/ 57.3
    alpha_pos = 0 / 57.3  # postavni kut krila
    i_r = 0 / 57.3  # kut uvijanja u korjenu krila
    i_t = 0 / 57.3  # kut uvijanja u vrhu krila
    a0_r = 6.9586 # NACA 2415, korijen krila
    a0_r = (0.604461 -0.236203)/(np.radians(4.0)-np.radians(0.0))
    a0_t = 6.6649 # NACA 2408, vrh krila
    a0_r = 2*np.pi #
    a0_t = 2*np.pi #
    phei = 0.0 / 57.3  # dihedral
    seg = Segment(b,c_r,c_t,L0,alpha_pos,i_r,i_t,phei,'2415','2408')
    wing.add_segment(seg)
    # Add Flight conditions
    fc = FlightCondition(15.0,0.0,2.0)
    test.add_flight_condition(fc)
    test.run_aero()
    return test



if __name__ == "__main__":
    ac = create_one_segment_wing()
    ac.run_aero()
    print(ac.get_info())
    ac.plot_planform()