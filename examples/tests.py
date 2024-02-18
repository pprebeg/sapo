from factory import *
a1 = get_verification_aircraft_aerodyn_class_one_segment()
a2 = get_verification_aircraft_aerodyn_class_two_segments()

print(a1.get_info())
#a1.plot_planform(True)
print(a2.get_info())
a2.plot_planform(True)