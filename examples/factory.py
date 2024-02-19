import src.aircraft as ac

def get_verification_aircraft_aerodyn_class_one_segment():
    a = ac.Aircraft('Verification_1')
    #Prepare wing
    wing = ac.Wing('wing')
    #prepare wing segments
    alpha = 4
    V_inf = 15
    b = 5 # each segment
    c_r = 2.0  # root chord
    c_t = 2.0  # tip chord
    sweep_LE = 45.0
    i_r = 0.0  # kut uvijanja u korjenu krila
    i_t = 0.0  # kut uvijanja u vrhu krila
    i_slope = (i_t - i_r) / b
    dihedral = 0.0# dihedral
    seg = ac.Segment(b,c_r,c_t,sweep_LE,i_r,i_slope,dihedral,'PLATE','PLATE',sym=ac.Symmetry.X_Z_PLANE)
    wing.add_segment(seg)
    a.add_component(wing)
    # Add Flight conditions
    fc = ac.FlightCondition(V_inf,0.0,alpha)
    a.add_flight_condition(fc)
    return a

def get_verification_aircraft_aerodyn_class_two_segments():
    a = ac.Aircraft('Verification_2')
    #Prepare wing
    wing = ac.Wing('wing')
    #prepare wing segments
    b = 2.5 # each segment
    c_r = 2.0  # root chord
    c_t = 2.0  # tip chord
    sweep_LE = 45.0
    i_r = 0.0  # kut uvijanja u korjenu krila
    i_t = 0.0  # kut uvijanja u vrhu krila
    i_slope = (i_t - i_r) / (2*b)
    dihedral = 0.0# dihedral
    # First segment
    seg = ac.Segment(b,c_r,c_t,sweep_LE,i_r,i_slope,dihedral,'PLATE','PLATE',sym=ac.Symmetry.X_Z_PLANE)
    wing.add_segment(seg)
    # Second segment
    seg = ac.Segment(b, c_r, c_t, sweep_LE, i_r, i_slope, dihedral, 'PLATE', 'PLATE', sym=ac.Symmetry.X_Z_PLANE)
    wing.add_segment(seg)
    wing.enforce_segments_LE_connection()
    a.add_component(wing)
    # Add Flight conditions
    alpha = 2
    V_inf = 16
    fc = ac.FlightCondition(V_inf,0.0,alpha)
    a.add_flight_condition(fc)
    return a


def get_uav_6kg_one_seg_one_fc():
    zero_angle = 1e-6
    a = ac.Aircraft('UAV_6kg_one_seg')
    #Prepare wing
    wing = ac.Wing('wing')
    #prepare wing segments

    b = 1.1 # each segment
    c_r = 0.43  # root chord
    c_t = 0.13  # tip chord
    sweep_LE = 10.0
    i_r = 0.0  # kut uvijanja u korjenu krila
    i_t = 0.0  # kut uvijanja u vrhu krila
    i_slope = (i_t-i_r)/b
    dihedral = 4.0# dihedral
    naca_root = '63412'
    naca_tip  = '63410'
    seg = ac.Segment(b,c_r,c_t,sweep_LE,i_r,i_slope,dihedral,naca_root,naca_tip,sym=ac.Symmetry.X_Z_PLANE)
    wing.add_segment(seg)
    wing.set_mass_calc_data(1850, 1.5 * 2, 0.0014)
    a.add_component(wing)
    #Prepare other
    all_other = ac.FixedComponent('All_other_parts',6.0)
    a.add_component(all_other)
    # Add Flight conditions
    alpha = zero_angle
    V_inf = 14
    fc = ac.FlightCondition(V_inf, 0.0, alpha)
    a.add_flight_condition(fc)
    return a