from typing import List,Dict
import numpy as np
from src.waissll import get_waissll_geometry_segment,calc_joukowski_forces_weissinger_lifting_line,get_unit_vector_mx3
from src.waissll import get_quantity_distribution_across_x
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from enum import Enum
from src.aero import calc_air_density_speedsound_ICAO, calc_mach, calc_reynolds, calc_friction_drag_constant
from src.aero import wing_mass_MHS
from src.panel import get_naca_4_5_airfoils_data


class Symmetry(Enum):
    NO_SYMMETRY = 'none'
    INHERITS = 'inherits'
    X_Z_PLANE = 'x-z-plane'
    X_Y_PLANE = 'x-y-plane'
    Y_Z_PLANE = 'y-z-plane'


class ForceType(Enum):
    LIFT = 1
    DRAG = 2
    THRUST = 3
    INERTIA = 4
    OTHER = 0

class CL0_AproxType(Enum):
    NONE = 1
    NACA_PANEL = 2

class CD0_AproxType(Enum):
    NONE = 1
    ADAPDT = 2

def get_min_max(numList, inMin = None, inMax = None):
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


class ForcesAndMoments:
    def __init__(self):
        self._force_vectors:Dict[str,np.ndarray] = {}
        self._force_types: Dict[str, ForceType] = {}
        self._force_location_keys: Dict[str, str] = {}
        self._force_locations:Dict[str,np.ndarray]  = {}
        self._moment_vectors:Dict[str,np.ndarray]  = {}

    @property
    def is_empty(self):
        return (self.num_force_vectors == 0 and self.num_moment_vectors == 0)

    @property
    def num_force_vectors(self):
        return len(self._force_vectors)

    @property
    def num_moment_vectors(self):
        return len(self._moment_vectors)

    def clear_all(self):
        self._force_vectors.clear()
        self._force_types.clear()
        self._force_location_keys.clear()
        self._force_locations.clear()
        self._moment_vectors.clear()

    def add_force_vectors_and_locations(self,key,force_vectors,location_key,force_type):
        self._force_vectors[key] = force_vectors
        self._force_types[key] = force_type
        self._force_location_keys[key] = location_key

    def add_scalar_weight(self, key:str,weight:float,location:np.ndarray,load_factor = 1.0):
        weight_vec = np.array([[0.0,0.0,- weight*load_factor]])
        self._force_vectors[key] = weight_vec
        self._force_types[key] = ForceType.INERTIA
        self._force_location_keys[key] = key
        self._force_locations[key] = location

    def add_force_locations(self,key,force_location):
        self._force_locations[key] = force_location

    def get_force_vector_sum(self,key):
        sum = np.sum(self._force_vectors[key], axis = 0)
        return sum

    def get_force_vector_sum_norm(self,key):
        sum = np.sum(self._force_vectors[key], axis = 0)
        return np.linalg.norm(sum)

    def get_drag(self):
        drag = 0.0
        for key in self._force_vectors.keys():
            force_type = self._force_types[key]
            if force_type == ForceType.DRAG:
                drag+= self.get_force_vector_sum_norm(key)
        return drag

    def get_lift(self):
        lift = 0.0
        for key in self._force_vectors.keys():
            force_type = self._force_types[key]
            if force_type == ForceType.LIFT:
                lift+= self.get_force_vector_sum_norm(key)
        return lift

    def get_lift_drag_ratio(self):
        lift = self.get_lift()
        drag = self.get_drag()
        return lift/drag

    def _get_total_sum_in_direction(self,dir):
        fdir = 0.0
        for key in self._force_vectors.keys():
            f = self.get_force_vector_sum(key)
            fdir+= f[dir]
        return fdir

    def get_total_sum_in_z(self):
        fz= self._get_total_sum_in_direction(2)
        return fz

    def get_info(self):
        np.set_printoptions(precision=3)
        msg = '\tForce vectors:{0}; Moment vectors:{1} \n'.format(self.num_force_vectors, self.num_moment_vectors)
        if self.num_force_vectors > 0:
            msg+= '\tForce vectors:\n'
        for key,force in self._force_vectors.items():
            msg+= '\t\t{0}, {1}, num vectors: {2}, vector sum: {3}, resultant: {4:.2f} \n'.format(
                key,str(self._force_types[key]),len(force),self.get_force_vector_sum(key),self.get_force_vector_sum_norm(key))
        msg +='\t\t------------------------------------------------------------------------------------\n'
        msg += '\t\tTotal sum in directions: x:{0:.3f}, y:{1:.3f},z: {2:.3f}\n'.format(
            self._get_total_sum_in_direction(0),self._get_total_sum_in_direction(1),self._get_total_sum_in_direction(2))
        return msg


class FlightCondition:
    def __init__(self,v_ms,h_m,alpha_deg,load_factor = 1.0,wf=1.0):
        self._V_inf = v_ms # m/s
        self._h_m = h_m # m
        self._alpha_deg = alpha_deg  # °
        self._load_factor = load_factor
        self._rho_h, self._a_h, self._mu =  calc_air_density_speedsound_ICAO(h_m)
        self._dyn_press = 0.5 * self._rho_h * self._V_inf ** 2.0
        self._wf = wf #weight factor of flight condition
        self._sina = np.sin(self.alpha_rad)
        self._cosa = np.cos(self.alpha_rad)

        self._forces = ForcesAndMoments()
        self._forces_Sref = 0.0

    #region Get Set properties


    @property
    def forces_Sref(self):
        return self._forces_Sref

    @forces_Sref.setter
    def forces_Sref(self, value):
        self._forces_Sref = value

    @property
    def load_factor(self):
        return self._load_factor

    @load_factor.setter
    def load_factor(self, value):
        self._load_factor = value

    @property
    def alpha_deg(self):
        return self._alpha_deg

    @alpha_deg.setter
    def alpha_deg(self, value):
        self._alpha_deg = value

    @property
    def V_inf(self):
        return self._V_inf

    @V_inf.setter
    def V_inf(self, value):
        self._V_inf = value
        self._dyn_press = 0.5 * self._rho_h * self._V_inf ** 2.0

    # endregion

    #region Get only properties


    @property
    def forces(self)->ForcesAndMoments:
        return self._forces

    @property
    def g(self):
        return 9.80665

    @property
    def dyn_press(self):
        return self._dyn_press

    @property
    def wf(self):
        return self._wf

    @property
    def alpha_rad(self):
        return np.radians(self._alpha_deg)

    @property
    def V_inf_vec(self):
        V_inf_x = self.V_inf * self._cosa
        V_inf_z = self.V_inf * self._sina
        V_inf_vec = np.array([V_inf_x, 0, V_inf_z])
        return V_inf_vec

    @property
    def rho(self):
        return self._rho_h

    @property
    def mu(self):
        return self._mu

    @property
    def Ma(self):
        return calc_mach(self.V_inf,self._a_h)
    #endregion

    #region Functions
    def calc_reynolds(self,c):
        Re = self._rho_h * self._V_inf *c/self._mu
        return Re

    def get_info(self):
        msg = '\tVcr = {0:.3f} m/s; h = {1:.0f} m; alpha = {2:.3f}; n = {3:.2f}\n'.format(
            self._V_inf, self._h_m, self._alpha_deg, self.load_factor)
        msg += '\t'+('-' * (len(msg)+2))+'\n'
        if not self.forces.is_empty:
            msg+= self.forces.get_info()
            msg+= '\tLift and drag coefficients: CL = {0:.3f}; CD = {1:.4f}; CL/CD = {2:.2f}; CL(3/2)/CD = {3:.2f}\n'.format(
                self.get_CL(),self.get_CD(),self.get_CL_CD_ratio(),self.get_CL_3_2_CD_ratio())
        else:
            msg+='\n'
        return msg

    def get_CL(self):
        L= self.forces.get_lift()
        CL= L/(self.dyn_press*self.forces_Sref)
        return CL

    def get_CD(self):
        D= self.forces.get_drag()
        CD= D/(self.dyn_press*self.forces_Sref)
        return CD

    def get_CL_CD_ratio(self):
        CL = self.get_CL()
        CD = self.get_CD()
        return CL/CD

    def get_CL_3_2_CD_ratio(self):
        CL = self.get_CL()
        CD = self.get_CD()
        return CL**(1.5) / CD
    #endregion


class Segment:
    def __init__(self, b, cr, ct, sweep_le, inc_r, inc_slope, dihedral, naca_r, naca_t,
                 num_vll_seg = 20, p_0 = np.zeros(3), sym = Symmetry.NO_SYMMETRY):
        # all angels are in degrees
        self._b = b
        self._c_r = cr
        self._c_t = ct
        self._sweep_le = sweep_le
        self._incidence_r = inc_r
        self._incidence_slope = inc_slope
        self._dihedral = dihedral
        self._num_vll_seg = num_vll_seg
        self._naca_r = naca_r
        self._naca_t = naca_t
        if naca_r == 'PLATE':
            self._cl0_r, self._a_r, self._lw_c_r,t_c_max_r = (0.0, 2*np.pi,2.0,0.0)
        else:
            self._cl0_r, self._a_r, self._lw_c_r,t_c_max_r = get_naca_4_5_airfoils_data(naca_r,50)
        if naca_t == 'PLATE':
            self._cl0_t, self._a_t, self._lw_c_t,t_c_max_t = (0.0, 2*np.pi,2.0,0.0)
        else:
            self._cl0_t, self._a_t, self._lw_c_t,t_c_max_t = get_naca_4_5_airfoils_data(naca_t,50)
        self._t_c_max = max(t_c_max_r,t_c_max_t)
        self._p_0 = p_0
        self._sym = sym
        self._xrot = 0.0

    #region Get only properties
    @property
    def sym(self):
        return self._sym
    @property
    def lw_c_r(self):
        return self._lw_c_r

    @property
    def lw_c_t(self):
        return self._lw_c_t

    @property
    def cl0_r(self):
        return self._cl0_r

    @property
    def cl0_t(self):
        return self._cl0_t

    @property
    def Sref(self):
        sref = self._b * (self._c_r + self._c_t) / 2.0
        return sref

    @property
    def A(self):
        return self._b**2/self.Sref

    @property
    def taper(self):
        return self._c_t / self._c_r

    @property
    def t_c_max(self):
        return self._t_c_max

    @property
    def sweep_025(self):
        L0 = np.arctan(np.tan(np.radians(self.sweep_le)) - (self.c_r - self.c_t) / (4 * self.b))
        return np.degrees(L0)

    @property
    def mac(self):
        la = self.taper
        return 2/3*self.c_r*(1+la+la**2)/(1+la)

    #endregion

    #region Get Set properties

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        self._b = value

    @property
    def c_r(self):
        return self._c_r

    @c_r.setter
    def c_r(self, value):
        self._c_r = value

    @property
    def c_t(self):
        return self._c_t

    @c_t.setter
    def c_t(self, value):
        self._c_t = value

    @property
    def sweep_le(self):
        return self._sweep_le

    @sweep_le.setter
    def sweep_le(self, value):
        self._sweep_le = value

    @property
    def incidence_r(self):
        return self._incidence_r

    @incidence_r.setter
    def incidence_r(self, value):
        self._incidence_r = value

    @property
    def incidence_slope(self):
        return self._incidence_slope

    @incidence_slope.setter
    def incidence_slope(self, value):
        self._incidence_slope = value

    @property
    def incidence_t(self):
        return self._incidence_r+self._incidence_slope*self.b


    @property
    def dihedral(self):
        return self._dihedral

    @dihedral.setter
    def dihedral(self, value):
        self._dihedral = value

    @property
    def p_0(self):
        return self._p_0

    @p_0.setter
    def p_0(self, value):
        self._p_0 = value
    #endregion

    #region Functions
    def get_waissll_geometry(self):
        m = self._num_vll_seg
        sym = False
        if self._sym == Symmetry.X_Z_PLANE:
            sym = True
        # Transform all angles to radians befoe using get_waissll_geometry_segment
        wll_geo = get_waissll_geometry_segment(
            self._p_0, self._b, self._c_r, self._c_t, np.radians(self.sweep_le),
            np.radians(self._incidence_r), np.radians(self.incidence_t), np.radians(self._dihedral),
            self._a_r, self._a_t, m,sym,np.radians(self._xrot))
        return wll_geo

    def get_segment_points(self):
        vRootLE = self._p_0
        b_2 = self._b
        cr = self._c_r
        ct = self. _c_t
        sweep = np.radians(self._sweep_le)
        dihedral = np.radians(self._dihedral)
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
        sweep = np.radians(self._sweep_le)
        dihedral = np.radians(self._dihedral)
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

    def get_info(self):
        msg = '\t\tb = {0:.3f}; c_r = {1:.3f}; c_t = {2:.3f}; SweepLE = {3:.2f}; dihedral = {4:.2f}\n'.format(
            self.b, self.c_r, self.c_t, self.sweep_le,self.dihedral)
        msg += '\t\tinc_r = {0:.3f}; inc_t = {1:.3f}; NACA_r = {2}; NACA_t = {3}\n'.format(
            self.incidence_r, self.incidence_t, self._naca_r, self._naca_t)
        return msg
    #endregion

class AircraftComponent():
    def __init__(self, uid:str):
        self._uid: str = uid

    @property
    def uid(self)->str:
        return self._uid

    def calculate_and_set_inertia_forces(self, fc:FlightCondition):
        mass = self.calc_mass()
        cg = self.calc_CG()
        fc.forces.add_scalar_weight('W_' + self.uid, mass * fc.g, cg)

    #region Abstract Functions
    def calc_mass(self):
        return 0.0

    def calc_CG(self):
        return np.zeros(3)

    def get_info(self):
        msg = '\tuid: {0}\n'.format(self.uid)
        return msg
    #endregion

class FixedComponent(AircraftComponent):
    def __init__(self,uid,mass:float,cg:np.ndarray=np.zeros(3)):
        super().__init__(uid)
        self._mass = mass
        self._cg = cg

    #region Owerloaded Abstract Functions
    def calc_mass(self):
        return self._mass

    def calc_CG(self):
        return self._cg
    #endregion

class WettedNonLiftingComponent(AircraftComponent):
    def __init__(self,uid,mass:float,cg:np.ndarray=np.zeros(3)):
        super().__init__(uid)

    def calculate_and_set_drag_forces(self, fc:FlightCondition):
        pass

class FixedWettedComponent(WettedNonLiftingComponent):
    def __init__(self,uid,mass:float,cd0,cda,cg:np.ndarray=np.zeros(3),cp:np.ndarray=np.zeros(3)):
        super().__init__(uid)
        self._mass = mass
        self._cg = cg
        self._cd0 = cd0
        self._cda = cda
        self._cp = cp
    def calculate_and_set_drag_forces(self, fc:FlightCondition):
        pass #TODO
    #region Owerloaded Abstract Functions
    def calc_mass(self):
        return self._mass

    def calc_CG(self):
        return self._cg
    #endregion

class LiftingBody(AircraftComponent):
    def __init__(self,uid):
        super().__init__(uid)
        self._segments:List[Segment] = []
        self._cl0_aprox_type = CL0_AproxType.NONE
        self._cd0_aprox_type = CD0_AproxType.NONE

    #region Get only properties

    @property
    def cl0_aprox_type(self):
        return self._cl0_aprox_type

    @property
    def cd0_aprox_type(self):
        return self._cd0_aprox_type

    @property
    def segments(self):
        return self._segments

    @property
    def Sref(self):
        sref = 0.0
        for seg in self._segments:
            if seg.sym == Symmetry.NO_SYMMETRY:
                srefi = seg.Sref
            else:
                srefi = 2*seg.Sref
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
            if seg.sym == Symmetry.NO_SYMMETRY:
                bb += bi
            else:
                bb += 2*bi
        return bb

    @property
    def c_r(self):
        cr = self._segments[0].c_r
        return cr

    @property
    def c_t(self):
        ct = self._segments[-1].c_t
        return ct

    @property
    def t_c_max(self):
        t_c_max = 0.0
        for seg in self._segments:
            t_c_max = max(t_c_max,seg.t_c_max)
        return t_c_max

    @property
    def taper(self):
        return self.c_t / self.c_r

    @property
    def sweep_le(self):
        sweep = 0.0
        for seg in self._segments:
            sweepi = seg.sweep_le * seg.Sref
            sweep += sweepi
        return sweep/self.Sref

    @property
    def sweep_025(self):
        b=self.b
        c_r = self.c_r
        c_t = self.c_t
        y = self.b * (c_r + 2 * c_t) / (3 * (c_r + c_t))
        return np.degrees(np.arctan(np.tan(np.radians(self.sweep_le)) - 4 * y / b * (1 - y / b)))

    @property
    def mac(self):
        la = self.taper
        return 2 / 3 * self.c_r * (1 + la + la ** 2) / (1 + la)

    @property
    def num_segments(self):
        return len(self._segments)

    #endregion

    #region Modify Functions
    def add_segment(self,seg):
        self._segments.append(seg)

    def clear_segments(self):
        self._segments.clear()

    def enforce_segments_LE_connection(self):
        p_t_LE = None
        for seg in self._segments:
            if p_t_LE is not None:
                seg.p_0 = p_t_LE
            p_t_LE = seg.get_segment_tip_LE_p()

    def set_cl0_cd0_aprox_types(self,cl0_aprox_type,cd0_aprox_type):
        self._cl0_aprox_type = cl0_aprox_type
        self._cd0_aprox_type = cd0_aprox_type
    #endregion

    #region Calculation Functions
    def get_waissll_geo_and_aero_data(self,fc:FlightCondition):
        p_kt_i = np.empty((0,3))
        e_n_kt_i = np.empty((0, 3))
        p_f_i = np.empty((0,3))
        p_1_i = np.empty((0,3))
        p_2_i = np.empty((0,3))
        db_i = np.empty((0))
        c_i = np.empty((0))
        #aero
        cl0_i = np.empty((0))
        cd0_i = np.empty((0))

        for seg in self._segments:
            seg_wll_geo = seg.get_waissll_geometry()
            s_p_kt_i, s_e_n_kt_i, s_p_f_i, s_p_1_i, s_p_2_i,s_db_i,s_c_i = seg_wll_geo
            p_kt_i = np.concatenate((p_kt_i, s_p_kt_i))
            p_f_i = np.concatenate((p_f_i, s_p_f_i))
            p_1_i = np.concatenate((p_1_i, s_p_1_i))
            p_2_i = np.concatenate((p_2_i, s_p_2_i))
            e_n_kt_i = np.concatenate((e_n_kt_i, s_e_n_kt_i))
            db_i= np.concatenate((db_i, s_db_i))
            c_i = np.concatenate((c_i, s_c_i))
            #aero
            ##lift
            if self.cl0_aprox_type == CL0_AproxType.NACA_PANEL:
                s_cl0_i = get_quantity_distribution_across_x(seg.cl0_r,seg.cl0_t,seg.c_r,seg.c_t,s_c_i)
                cl0_i = np.concatenate((cl0_i, s_cl0_i))
            ##drag
            if self.cd0_aprox_type == CD0_AproxType.ADAPDT:
                Re_i_r = fc.calc_reynolds(seg.c_r)
                Re_i_t = fc.calc_reynolds(seg.c_t)
                lambd=seg.c_t/seg.c_r
                s_cd0_r = calc_friction_drag_constant(Re_i_r, fc.Ma, seg.c_r, lambd, np.radians(self.sweep_le), seg.lw_c_r)
                s_cd0_t = calc_friction_drag_constant(Re_i_t, fc.Ma, seg.c_t, lambd, np.radians(seg.sweep_le), seg.lw_c_t)
                s_cd0_i = get_quantity_distribution_across_x(s_cd0_r, s_cd0_t, seg.c_r, seg.c_t, s_c_i)
                cd0_i = np.concatenate((cd0_i, s_cd0_i))

        if cd0_i.size==0:
            cd0_i = None
        if cl0_i.size==0:
            cl0_i = None
        return p_kt_i, e_n_kt_i, p_f_i, p_1_i, p_2_i,db_i,c_i,cl0_i,cd0_i

    def calculate_and_set_aero_forces(self, fc:FlightCondition):
        p_kt_i, e_n_kt_i, p_f_i, p_1_i, p_2_i,db_i,c_i,cl0_i,cd0_i = self.get_waissll_geo_and_aero_data(fc)
        L_i_vec,D_i_vec = calc_joukowski_forces_weissinger_lifting_line(p_kt_i, e_n_kt_i, p_f_i, p_1_i, p_2_i, db_i,
                                                                          fc.V_inf_vec, fc.rho)
        fc.forces.add_force_locations('wll_i_f', p_f_i)
        fc.forces.add_force_vectors_and_locations('L_i_vec', L_i_vec, 'wll_i_f', ForceType.LIFT)
        fc.forces.add_force_vectors_and_locations('D_i_vec', D_i_vec, 'wll_i_f', ForceType.DRAG)

        e_L = get_unit_vector_mx3(L_i_vec)
        e_D = get_unit_vector_mx3(D_i_vec)
        dp_db_c = fc.dyn_press*db_i*c_i
        if not (cl0_i is None):
            L_0_i= dp_db_c*cl0_i
            L_0_i_vec = np.einsum('i,ij->ij', L_0_i, e_L)
            fc.forces.add_force_vectors_and_locations('L_0_i_vec', L_0_i_vec, 'wll_i_f', ForceType.LIFT)
        if not (cd0_i is None):
            D_0_i =  dp_db_c * cd0_i
            D_0_i_vec = np.einsum('i,ij->ij', D_0_i, e_D)
            fc.forces.add_force_vectors_and_locations('D_0_i_vec', D_0_i_vec, 'wll_i_f',ForceType.DRAG)

        return 0



    #endregion

    #region Info Functions
    def get_info(self):
        msg = '\tuid: {0}; num segments: {1}\n'.format(self.uid,self.num_segments)
        iseg=1
        for seg in self.segments:
            msg += '\tSegment {0}:\n'.format(iseg)
            msg += seg.get_info()
            iseg+=1
        return msg
    #endregion

class Wing(LiftingBody):
    def __init__(self, uid):
        super().__init__(uid)
        self._hms_mass_data = None

    def set_mass_calc_data(self,rho_mat,n_ult, Kp):
        self._hms_mass_data = (rho_mat,n_ult, Kp)

    #region Overloaded Functions
    def calc_mass(self):
        if self._hms_mass_data is None:
            return 0.0
        else:
            rho_mat, n_ult, Kp = self._hms_mass_data
            mass = wing_mass_MHS(self.Sref, self.mac, self.t_c_max, rho_mat, Kp, self.A, n_ult,
                                 np.radians(self.sweep_025), self.taper)
            return mass
        return mass

    def calc_CG(self):
        return np.zeros(3)

    #endregion
    #region Info Functions
    def get_info(self):
        msg = super().get_info()
        msg += '\tWing: b = {0:0.3f}; A = {1:0.3f}:\n'.format(self.b,self.A)
        return msg
    #endregion
class Tail(LiftingBody):
    def __init__(self, uid):
        super().__init__(uid)

class HorizontalTail(Tail):
    def __init__(self, uid):
        super().__init__(uid)


    #region Overloaded Functions
    def calc_mass(self):
        mass = 0.0
        return mass

    def calc_CG(self):
        return np.zeros(3)
    #endregion

class Aircraft:
    def __init__(self,name):
        self._name = name
        self._flight_conditions:List[FlightCondition] = []
        self._components = {}

    #region Get only properties
    @property
    def components(self)->Dict[str,AircraftComponent]:
        return self._components

    @property
    def lifting_components(self) -> Dict[str, LiftingBody]:
        lifting= {}
        for key,value in self._components.items():
            if isinstance(value,LiftingBody):
                lifting[key]=value
        return lifting

    @property
    def drag_components(self) -> Dict[str, WettedNonLiftingComponent]:
        dragcomps = {}
        for key, value in self._components.items():
            if isinstance(value, WettedNonLiftingComponent):
                dragcomps[key] = value
        return dragcomps

    @property
    def flight_conditions(self)->List[FlightCondition]:
        return self._flight_conditions

    @property
    def num_comp(self):
        return len(self._components)

    @property
    def num_flight_cond(self):
        return len(self._flight_conditions)
    @property
    def name(self):
        return self._name

    #endregion

    #region Modify object functions

    def add_flight_condition(self,fc:FlightCondition):
        self._flight_conditions.append(fc)

    def clear_flight_conditions(self):
        self._flight_conditions.clear()

    def add_component(self,component:AircraftComponent):
        self._components[component.uid] = component

    def clear_flight_conditions(self):
        self._flight_conditions.clear()
    #endregion

    #region Calculation functions
    def calculate_forces(self):
        for fc in self._flight_conditions:
            fc.forces.clear_all()
        self.calculate_aero_forces()
        self.calculate_inertia_forces()

    def calculate_aero_forces(self):
        for fc in self._flight_conditions:
            for l_body in self.lifting_components.values():
                l_body.calculate_and_set_aero_forces(fc)
                if isinstance(l_body,Wing):
                    fc.forces_Sref = l_body.Sref

            for drag_body in self.drag_components.values():
                drag_body.calculate_and_set_drag_forces(fc)

    def calculate_inertia_forces(self):
        for fc in self._flight_conditions:
            for component in self.components.values():
                component.calculate_and_set_inertia_forces(fc)

    #endregion

    #region Ploting and Printing functions
    def get_info(self):
        msg = 'Name: {0}'.format(self.name)+'\n'
        msg += 'Num Components: {0}, Num Flight Conditions {1}'.format(self.num_comp,self.num_flight_cond) + '\n'
        msg += 'Components:' + '\n'
        for comp in self.components.values():
            msg += comp.get_info()
        msg += 'Flight Conditions:' + '\n'
        iseg=1
        for fc in self.flight_conditions:
            msg += '\t{0}'.format(iseg)+fc.get_info()+'\n'
            iseg+=1
        return msg

    def plot_planform(self,do_plot_wllp = True):
        xplot = []
        yplot = []
        zplot = []
        labelplot = []
        axMin = None
        axMax = None

        for l_body in self.lifting_components.values():
            i_seg = 0
            for seg in l_body.segments:
                i_seg +=1
                x, y, z = seg.get_segment_points()
                xplot.append(x)
                yplot.append(y)
                zplot.append(z)
                labelplot.append('{0} seg {1}'.format(l_body.uid, i_seg))
                axMin, axMax = get_min_max(x, axMin, axMax)
                axMin, axMax = get_min_max(y, axMin, axMax)
                axMin, axMax = get_min_max(z, axMin, axMax)

        if len(xplot) > 0:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            for i in range(len(xplot)):
                verts = [list(zip(xplot[i], yplot[i], zplot[i]))]
                poly = Poly3DCollection(verts, alpha=0.5)
                ax.add_collection3d(poly)
                ax.plot(xplot[i], yplot[i], zplot[i], label=labelplot[i])

            if do_plot_wllp:
                for l_body in self.lifting_components.values():
                    p_kt, n_kt, p_f, p_1, p_2,db_i,c_i,cl0_i,cd0_i = l_body.get_waissll_geo_and_aero_data(self.flight_conditions[0])
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
    #endregion