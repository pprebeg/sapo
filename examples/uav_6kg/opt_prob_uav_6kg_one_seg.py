import numpy as np

try:
    from moobench.optbase import *
except ImportError:
    pass
from src.aircraft import Aircraft,FlightCondition,Segment,Wing

class CallbackPropertyGetSetConnector(BasicGetSetConnector):
    def __init__(self,instance,prop):
        self._instance=instance
        self._property= prop
    @property
    def value(self):
        return self._property.fget( self._instance )

    @value.setter
    def value(self,value):
        self._property.fset(self._instance, value)

class CallbackGetPropRatioConnector(BasicGetConnector):
    def __init__(self,num,denum):
        self._num =num
        self._denum = denum
    @property
    def value(self):
        return self._num.value/self._denum.value

class Aircraft_AnMod(AnalysisExecutor):
    def __init__(self,gen_aircraft_func):
        super().__init__()
        self._ac:Aircraft= gen_aircraft_func()
        self._ac.calculate_forces()

    @property
    def aircraft(self):
        return self._ac

    def analyze(self):
        self._ac.calculate_forces()
        return AnalysisResultType.OK

    def objfun_cd_mean(self):
        cd=0.0
        wfsum = 0.0
        for fc in self.aircraft.flight_conditions:
            cd+=fc.get_CD()*fc.wf
            wfsum += fc.wf
        cd_mean = cd/wfsum
        return cd_mean

    def objfun_cl_cd_mean(self):
        cl_cd=0.0
        wfsum = 0.0
        for fc in self.aircraft.flight_conditions:
            cl_cd+=fc.get_CL_CD_ratio()*fc.wf
            wfsum += fc.wf
        cl_cd_mean = cl_cd/wfsum
        return - cl_cd_mean

    def objfun_cl_3_2_cd_mean(self):
        cl_3_2_cd=0.0
        wfsum = 0.0
        for fc in self.aircraft.flight_conditions:
            cl_3_2_cd+=fc.get_CL_3_2_CD_ratio()*fc.wf
            wfsum += fc.wf
        cl_3_2_cd_mean = cl_3_2_cd/wfsum
        return - cl_3_2_cd_mean


class UAV_MinCD_OptProb(OptimizationProblem):
    def __init__(self,gen_aircraft_func,name='',):
        if name == '':
            name = 'Aircraf_Opt_Problem'
        super().__init__(name)
        am = Aircraft_AnMod(gen_aircraft_func)
        self._ac = am.aircraft

        ifc=0
        for fc in am.aircraft.flight_conditions:
            ifc+=1
            #fca_conn = CallbackPropertyGetSetConnector(fc,FlightCondition.alpha_deg)
            #self.add_design_variable(DesignVariable('xfc_{}'.format(ifc), fca_conn, 0.0,5.0))
            fcZ_conn = CallbackGetConnector(fc.forces.get_total_sum_in_z)
            self.add_constraint(DesignConstraint('g_Fz_fc_{}'.format(ifc), fcZ_conn, 0.0, ConstrType.EQ))
        iws = 1
        for lbody in self.aircraft.lifting_components.values():
            for seg in lbody.segments:
                ws_b_con = CallbackPropertyGetSetConnector(seg,Segment.b)
                self.add_design_variable(DesignVariable('x_b_ws_{}'.format(iws), ws_b_con, 0.5, 3.0))
                ws_cr_con = CallbackPropertyGetSetConnector(seg, Segment.c_r)
                self.add_design_variable(DesignVariable('x_c0_ws_{}'.format(iws), ws_cr_con, .01, 0.5))
                ws_ct_con = CallbackPropertyGetSetConnector(seg, Segment.c_t)
                self.add_design_variable(DesignVariable('x_ct_ws_{}'.format(iws), ws_ct_con, 0.01, 0.5))
                # add ratio constraint
                ws_g_ct_cr = CallbackGetPropRatioConnector(ws_ct_con,ws_cr_con)
                self.add_constraint(DesignConstraint('g_ws_cr_ct', ws_g_ct_cr, 1.0, ConstrType.LT))

                ws_sweep_con = CallbackPropertyGetSetConnector(seg, Segment.sweep_le)
                self.add_design_variable(DesignVariable('x_sweep_ws_{}'.format(iws), ws_sweep_con, 0, 45))
                ws_i_r_con = CallbackPropertyGetSetConnector(seg, Segment.incidence_r)
                self.add_design_variable(DesignVariable('x_inc_r_ws_{}'.format(iws), ws_i_r_con, -3 , 5))
                ws_i_slope_con = CallbackPropertyGetSetConnector(seg, Segment.incidence_slope)
                self.add_design_variable(DesignVariable('x_inc_slope_ws_{}'.format(iws), ws_i_slope_con, -7, 0))
                ws_dihedral_con = CallbackPropertyGetSetConnector(seg, Segment.dihedral)
                self.add_design_variable(DesignVariable('x_dih_{}'.format(iws), ws_dihedral_con, 0.00, 20))
                iws+=1
            if isinstance(lbody,Wing):
                w_A_con = CallbackPropertyGetSetConnector(lbody, Wing.A)
                self.add_constraint(DesignConstraint('g_wing_A', w_A_con, 10.0, ConstrType.LT))
        objd = CallbackGetConnector(am.objfun_cl_cd_mean)
        self.add_objective(DesignObjective('obj_CD', objd))
        self.add_analysis_executor(am)

    @property
    def aircraft(self):
        return self._ac

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value