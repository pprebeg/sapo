from factory import *
if False:
    a1 = get_verification_aircraft_aerodyn_class_one_segment()
    a1.calculate_forces()
    # a1.plot_planform(True)
    print(a1.get_info())
if False:
    a2 = get_verification_aircraft_aerodyn_class_two_segments()
    a2.calculate_forces()
    print(a2.get_info())
    #a2.plot_planform(True)

if True:
    uav_1 = get_test_survailance_uav_one_segment()
    uav_1.calculate_forces()
    print(uav_1.get_info())
    uav_1.plot_planform()