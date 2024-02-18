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

    def add_scalar_weight(self, key:str,weight:float,location:np.ndarray):
        weight_vec = np.array([0.0,0.0,weight])
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
            f = self.get_force_vector_sum_norm(key)
            fdir+= f[dir]
        return fdir

    def get_total_sum_in_z(self):
        fz= self._get_total_sum_in_direction(2)
        return fz


class FlightCondition:
    def __init__(self,v_ms,h_m,alpha_deg,wf=1.0):
        self._V_inf = v_ms # m/s
        self._h_m = h_m # m
        self._alpha_deg = alpha_deg  # °
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
    def forces(self):
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
        msg = 'FC: Vcr = {0:.2f} m/s; h = {1:.0f} m; alpha = {2:.2f} °; '.format(self._V_inf, self._h_m, self._alpha_deg)
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
    def __init__(self, b, cr, ct, sweep_le, inc_r, inc_t, dihedral, naca_r, naca_t,
                 num_vll_seg = 20, p_0 = np.zeros(3), sym = Symmetry.NO_SYMMETRY):
        # all angels are in degrees
        self._b = b
        self._c_r = cr
        self._c_t = ct
        self._sweep_le = sweep_le
        self._incidence_r = inc_r
        self._incidence_t = inc_t
        self._dihedral = dihedral
        self._num_vll_seg = num_vll_seg
        if naca_r == 'PLATE':
            self._cl0_r, self._a_r, self._lw_c_r = (0.0, 2*np.pi,2.0)
        else:
            self._cl0_r, self._a_r, self._lw_c_r = get_naca_4_5_airfoils_data(naca_r,50)
        if naca_t == 'PLATE':
            self._cl0_t, self._a_t, self._lw_c_t = (0.0, 2*np.pi,2.0)
        else:
            self._cl0_t, self._a_t, self._lw_c_t = get_naca_4_5_airfoils_data(naca_t,50)
        self._p_0 = p_0
        self._sym = sym
        self._xrot = 0.0

    #region Get only properties
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
    def sweep_LE(self):
        return self._sweep_le

    @sweep_LE.setter
    def sweep_LE(self, value):
        self._sweep_le = value

    @property
    def incidence_r(self):
        return self._incidence_r

    @incidence_r.setter
    def incidence_r(self, value):
        self._incidence_r = value

    @property
    def incidence_t(self):
        return self._incidence_t

    @incidence_t.setter
    def incidence_t(self, value):
        self._incidence_t = value

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
            self._p_0, self._b, self._c_r, self._c_t, np.radians(self._sweep_le),
            np.radians(self._incidence_r), np.radians(self._incidence_t), np.radians(self._dihedral),
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
        msg = 'uid: {0}'.format(self.uid)
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

class LiftingBody(AircraftComponent):
    def __init__(self,uid):
        super().__init__(uid)
        self._segments:List[Segment] = []

    #region Get only properties

    @property
    def segments(self):
        return self._segments

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
            s_cl0_i = get_quantity_distribution_across_x(seg.cl0_r,seg.cl0_t,seg.c_r,seg.c_t,s_c_i)
            cl0_i = np.concatenate((cl0_i, s_cl0_i))
            ##drag
            Re_i_r = fc.calc_reynolds(seg.c_r)
            Re_i_t = fc.calc_reynolds(seg.c_t)
            lambd=seg.c_t/seg.c_r
            s_cd0_r = calc_friction_drag_constant(Re_i_r, fc.Ma, seg.c_r, lambd, seg.sweep_LE, seg.lw_c_r)
            s_cd0_t = calc_friction_drag_constant(Re_i_t, fc.Ma, seg.c_t, lambd, seg.sweep_LE, seg.lw_c_t)
            s_cd0_i = get_quantity_distribution_across_x(s_cd0_r, s_cd0_t, seg.c_r, seg.c_t, s_c_i)
            cd0_i = np.concatenate((cd0_i, s_cd0_i))
        return p_kt_i, e_n_kt_i, p_f_i, p_1_i, p_2_i,db_i,c_i,cl0_i,cd0_i

    def calculate_and_set_aero_forces(self, fc:FlightCondition):
        p_kt_i, e_n_kt_i, p_f_i, p_1_i, p_2_i,db_i,c_i,cl0_i,cd0_i = self.get_waissll_geo_and_aero_data(fc)
        L_i_vec,D_i_vec = calc_joukowski_forces_weissinger_lifting_line(p_kt_i, e_n_kt_i, p_f_i, p_1_i, p_2_i, db_i,
                                                                          fc.V_inf_vec, fc.rho)
        e_L = get_unit_vector_mx3(L_i_vec)
        e_D = get_unit_vector_mx3(D_i_vec)
        dp_db_c = fc.dyn_press*db_i*c_i
        L_0_i= dp_db_c*cl0_i
        D_0_i =  dp_db_c * cd0_i
        L_0_i_vec = np.einsum('i,ij->ij', L_0_i, e_L)
        D_0_i_vec = np.einsum('i,ij->ij', D_0_i, e_D)
        fc.forces.add_force_locations('wll_i_f',p_f_i)
        fc.forces.add_force_vectors_and_locations('L_i_vec',L_i_vec,'wll_i_f',ForceType.LIFT)
        fc.forces.add_force_vectors_and_locations('D_i_vec', D_i_vec, 'wll_i_f',ForceType.DRAG)
        fc.forces.add_force_vectors_and_locations('L_0_i_vec', L_0_i_vec, 'wll_i_f',ForceType.LIFT)
        fc.forces.add_force_vectors_and_locations('D_0_i_vec', D_0_i_vec, 'wll_i_f',ForceType.DRAG)
        return 0



    #endregion

    #region Info Functions
    def get_info(self):
        msg = 'uid: {0}; num segments: {1}'.format(self.uid,self.num_segments)
        return msg
    #endregion



class Wing(LiftingBody):
    def __init__(self, uid):
        super().__init__(uid)

    #region Overloaded Functions
    def calc_mass(self):
        Sw = 0.0
        c_mac = 0.0
        t_c_max = 0.0
        rho_mat = 0.0
        Kp = 0.0
        AR = 0.0
        n_ult = 0.0
        L_1_4 = 0.0
        lambd = 0.0
        mass = wing_mass_MHS(Sw, c_mac, t_c_max, rho_mat, Kp, AR, n_ult, L_1_4, lambd)
        return mass

    def calc_CG(self):
        return np.zeros(3)

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
            for l_body in self.lifting_components:
                l_body.calculate_and_set_aero_forces(fc)
                if isinstance(l_body,Wing):
                    fc.forces_Sref = self.l_body.Sref

    def calculate_inertia_forces(self):
        for fc in self._flight_conditions:
            for component in self.components:
                component.calculate_and_set_inertia_forces(fc)

    #endregion

    #region Ploting and Printing functions
    def get_info(self):
        msg = 'Name: {0}'.format(self.name)+'\n'
        msg += 'Num Components: {0}, Num Flight Conditions {1}'.format(self.num_comp,self.num_flight_cond) + '\n'
        msg += 'Components:' + '\n'
        for comp in self.components.values():
            msg += comp.get_info()+'\n'
        msg += 'Flight Conditions:' + '\n'
        for fc in self.flight_conditions:
            msg += fc.get_info()+'\n'
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