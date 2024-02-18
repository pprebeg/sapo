import numpy as np

try:
    from moobench.optbase import *
except ImportError:
    pass
from src.aircraft import Aircraft,FlightCondition,Segment

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

    @property
    def aircraft(self):
        return self._ac

    def analyze(self):
        pass
        self._ac.calculate_aero_forces()
        return AnalysisResultType.OK

    def objfun_drag(self):
        D=0.0
        for fc in self.aircraft.flight_conditions:
            D+=fc.D*fc.wf
        return D


class Aircraft_OptProb(OptimizationProblem):
    def __init__(self,gen_aircraft_func,name=''):
        if name == '':
            name = 'Aircraf_Opt_Problem'
        super().__init__(name)
        am = Aircraft_AnMod(gen_aircraft_func)
        ifc=0
        for fc in am.aircraft.flight_conditions:
            ifc+=1
            fca_conn = CallbackPropertyGetSetConnector(fc,FlightCondition.alpha_deg)
            self.add_design_variable(DesignVariable('xfc_{}'.format(ifc), fca_conn, 0.0,5.0))
            fcN_conn = CallbackPropertyGetSetConnector(fc, FlightCondition.N)
            self.add_constraint(DesignConstraint('gWLfc_{}'.format(ifc), fcN_conn, am.aircraft.W_TO, ConstrType.EQ))
        iws = 0
        for seg in am.aircraft.wing.segments:
            iws+=1
            ws_b_con = CallbackPropertyGetSetConnector(seg,Segment.b)
            self.add_design_variable(DesignVariable('xb_ws_{}'.format(iws), ws_b_con, 1.0, 15.0))
            ws_c0_con = CallbackPropertyGetSetConnector(seg, Segment.c_r)
            self.add_design_variable(DesignVariable('xc0_ws_{}'.format(iws), ws_c0_con, .01, 7.0))
            ws_ct_con = CallbackPropertyGetSetConnector(seg, Segment.c_t)
            self.add_design_variable(DesignVariable('xct_ws_{}'.format(iws), ws_ct_con, 0.01, 6.0))
            ws_L0_con = CallbackPropertyGetSetConnector(seg, Segment.sweep_LE)
            self.add_design_variable(DesignVariable('xL0_ws_{}'.format(iws), ws_L0_con, 5/57.3, 60/57.3))
            ws_alpha_pos_con = CallbackPropertyGetSetConnector(seg, Segment.alpha_pos)
            self.add_design_variable(DesignVariable('xalpha_pos_ws_{}'.format(iws), ws_alpha_pos_con, 0.01 / 57.3, 20/57.3))
            ws_i_r_con = CallbackPropertyGetSetConnector(seg, Segment.incidence_r)
            self.add_design_variable(DesignVariable('xi_rws_{}'.format(iws), ws_i_r_con, -2 / 57.3, 7 / 57.3))
            ws_i_t_con = CallbackPropertyGetSetConnector(seg, Segment.incidence_t)
            self.add_design_variable(DesignVariable('xi_t_ws_{}'.format(iws), ws_i_t_con, 0/ 57.3, 15 / 57.3))
            ws_phei_con = CallbackPropertyGetSetConnector(seg, Segment.dihedral)
            self.add_design_variable(DesignVariable('xphei_{}'.format(iws), ws_phei_con, 0.01 / 57.3, 20 / 57.3))
        objd = CallbackGetConnector(am.objfun_drag)
        self.add_objective(DesignObjective('obj', objd))
        self.add_analysis_executor(am)