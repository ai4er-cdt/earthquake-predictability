"""
Created on Wed Jan 11 10:46:23 2023

@author: vinco
"""


def set_param(exp):
    if exp == "b698":
        parameters = {
            # Time interval to load from the experiment (don't load the begining
            # as the slip stick cycle does not start immediately and don't load
            # the end of experiment as we do not include friction evolution 
            # effects in the model)
            "t0": 11100.0, # index of initial time to load from the experiment
            "tend": 11300.0, # index of final time to load from the experiment?
            "Nheaders": 2, # Number of header lines in the data file (name, units)
            "dir_data": "gtc_quakes_data/labquakes/", # Path lab data repository
            "case_study": "MeleVeeduetal2020/b698", # Path to experiment data
            "data_type": "lab",
            "struct_type": "MeleVeeduetal2020",
            "file_format": "txt",
            "downsample_factor": 1, # FIXME: what is this?
            "vl": 10, # FIXME: what is this?
            "segment": None,
            "obs_unit": "MPa", # Units for shear stress
            "time_unit": "s", # Units for time
        }
        # Add labels appropriate for the units
        parameters["obs_label"] = r"$\tau_f$ [" + parameters["obs_unit"] + "]"
        parameters["time_label"] = r"Time [" + parameters["time_unit"] + "]"
    elif exp == "b726":
        parameters = {
            "t0": 9450.0,
            "tend": 9650.0,
            "Nheaders": 2,
            "dir_data": "gtc_quakes_data/labquakes/",
            "case_study": "MeleVeeduetal2020/b726",
            "data_type": "lab",
            "struct_type": "MeleVeeduetal2020",
            "file_format": "txt",
            "downsample_factor": 1,
            "vl": 10,
            "segment": None,
            "obs_unit": "MPa",
            "time_unit": "s",
        }
        parameters["obs_label"] = r"$\tau_f$ [" + parameters["obs_unit"] + "]"
        parameters["time_label"] = (
            r"$\tau_f$ [" + parameters["time_unit"] + "]"
        )
    elif exp == "i417":
        parameters = {
            "t0": 3650.0,
            "tend": 3850.0,
            "Nheaders": 2,
            "dir_data": "gtc_quakes_data/labquakes/",
            "case_study": "MeleVeeduetal2020/i417",
            "data_type": "lab",
            "struct_type": "MeleVeeduetal2020",
            "file_format": "txt",
            "downsample_factor": 1,
            "vl": 10,
            "segment": None,
            "obs_unit": "MPa",
            "time_unit": "s",
        }
        parameters["obs_label"] = r"$\tau_f$ [" + parameters["obs_unit"] + "]"
        parameters["time_label"] = (
            r"$\tau_f$ [" + parameters["time_unit"] + "]"
        )
    elif exp == "p4679":
        parameters = {
            "t0": 4233.28,  # 4229.0
            "tend": 4535.0,
            "Nheaders": 2,
            "dir_data": "gtc_quakes_data/labquakes/",
            "case_study": "Marone/p4679",
            "data_type": "lab",
            "struct_type": "Marone",
            "file_format": "txt",
            "downsample_factor": 1,
            "vl": None,
            "segment": None,
            "obs_unit": "MPa",
            "time_unit": "s",
        }
        parameters["obs_label"] = r"$\tau_f$ [" + parameters["obs_unit"] + "]"
        parameters["time_label"] = r"Time [" + parameters["time_unit"] + "]"
    elif exp == "p4581":
        parameters = {
            "t0": 2325.0,  # 2075.0,
            "tend": 2525.0,  # 2275.0,
            "Nheaders": 5,
            "var4peaks": "ShearStress",
            "peaks_dist": 300,
            "peaks_height": 0.6,
            "dir_data": "gtc_quakes_data/labquakes/",
            "case_study": "Marone/p4581",
            "struct_type": "Marone_p4581",
            "data_type": "lab",
            "file_format": "txt",
            "vl": None,
        }
    elif exp == "sim_b698":
        parameters = {
            "t0": 0.0,
            "tend": 200.0,
            "Nheaders": 2,
            "dir_data": "gtc_quakes_data/synquakes/",
            "case_study": "Gualandietal2023/b698",
            "data_type": "synthetic",
            "struct_type": None,
            "file_format": "txt",
            "downsample_factor": 1,
            "vl": None,
            "segment": None,
            "obs_unit": "MPa",
            "time_unit": "s",
        }
        parameters["obs_label"] = r"$\tau_f$ [" + parameters["obs_unit"] + "]"
        parameters["time_label"] = r"Time [" + parameters["time_unit"] + "]"
    elif exp == "sim_b726":
        parameters = {
            "t0": 0.0,
            "tend": 200.0,
            "Nheaders": 2,
            "dir_data": "gtc_quakes_data/synquakes/",
            "case_study": "Gualandietal2023/b726",
            "data_type": "synthetic",
            "struct_type": None,
            "file_format": "txt",
            "downsample_factor": 1,
            "vl": None,
            "segment": None,
            "obs_unit": "MPa",
            "time_unit": "s",
        }
        parameters["obs_label"] = r"$\tau_f$ [" + parameters["obs_unit"] + "]"
        parameters["time_label"] = r"Time [" + parameters["time_unit"] + "]"
    elif exp == "sim_i417":
        parameters = {
            "t0": 0.0,
            "tend": 200.0,
            "Nheaders": 2,
            "dir_data": "gtc_quakes_data/synquakes/",
            "case_study": "Gualandietal2023/i417",
            "data_type": "synthetic",
            "struct_type": None,
            "file_format": "txt",
            "downsample_factor": 1,
            "vl": None,
            "segment": None,
            "obs_unit": "MPa",
            "time_unit": "s",
        }
        parameters["obs_label"] = r"$\tau_f$ [" + parameters["obs_unit"] + "]"
        parameters["time_label"] = r"Time [" + parameters["time_unit"] + "]"
    elif exp == "cascadia":
        parameters = {
            "t0": 2000.0,
            "tend": 2030.0,
            "Nheaders": None,
            "dir_data": "gtc_quakes_data/slowquakes/",
            "case_study": "Micheletal2019a/dynamical_system_variables_rough",
            "data_type": "nature",
            "struct_type": None,
            "file_format": "mat",
            "downsample_factor": 1,
            "vl": 40e-3,
            "segment": 1,
            "obs_unit": r"m$^3$",
            "time_unit": "yr",
        }
        parameters["obs_label"] = r"$P$ [" + parameters["obs_unit"] + "]"
        parameters["time_label"] = r"Time [" + parameters["time_unit"] + "]"
    else:
        # pdb.set_trace()
        print("`exp` not recognized.")
    return parameters
