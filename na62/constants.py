# Detector related constants
lkr_position = 241093
muv3_position = 246800
clock_period = 24.951059536 # ns

# Particle properties
photon_mass = 0
electron_mass = 0.511
muon_mass = 105.7
kaon_charged_mass = 493.677
pion_charged_mass = 139.57039
pion_neutral_mass = 134.9768
proton_mass = 938.27208816
kaon_br_map = {"k3pi": 0.05583, "ke3": 0.0507, "kmu2": 0.6356, "k2pi": 0.2067, "kmu3": 0.03352}
c = 299792458 # m/s

# Definitions
rich_hypothesis_map = {"bckg": 0, "e": 1, "mu": 2, "pi": 3, "k": 4, "mult": 99}
"""
Human-readable string to conventional integer value used in dataframe for 'rich_hypothesis' variable
"""
event_type_map = {"k3pi": 1, "ke3": 2, "kmu2": 3, "k2pi": 4, "kmu3": 5, "background": 6}
"""
Human-readable string to conventional integer value used in dataframe for 'event_type' variable
"""
