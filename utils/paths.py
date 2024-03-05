# credit: Camilla Billari (cgb47@cam.ac.uk)

import getpass
import os
import socket

# Data directories
MAIN_DIRECTORY = REL_DATA_DIR = GTC_DATA_DIR = None

# Retrieve the hostname of the current system to determine the environment
hostname = socket.gethostname()
# Retrieve the current user's login name
username = getpass.getuser()

# Check if the hostname indicates the system is part of the ac.uk domain
# Else check if the username is present to set up a local development environment
if "jasmin.ac.uk" in hostname or "jc.rl.ac.uk" in hostname:
    # Set the base directory path for JASMIN users, assuming the script runs within the JASMIN infrastructure
    current_dir = "/gws/nopw/j04/ai4er/users/"
    # List specific user directories within the JASMIN environment
    directories = ["cgbill", "pn341", "jpoff", "trr26", "ashine"]
    # Determine which of these directories the current user has write access to
    writable_directories = [
        d
        for d in directories
        if os.access(os.path.join(current_dir, d), os.W_OK)
    ]
    # Assume the first writable directory belongs to the JASMIN user running the script
    jasmin_user = writable_directories[0]
    # Define the main directory path for earthquake predictability
    MAIN_DIRECTORY = (
        f"/gws/nopw/j04/ai4er/users/{jasmin_user}/earthquake-predictability"
    )
    # Specify the relative path to the data directory within the main directory
    REL_DATA_DIR = "data"
    REL_PLOTS_DIR = "plots"
    REL_RESULTS_DIR = "results"
else:
    # Specify the relative path to the local data directory
    REL_DATA_DIR = "data_local"
    REL_PLOTS_DIR = "plots_local"
    REL_RESULTS_DIR = "results_local"
    if "camilla" in username:
        # Define the main directory on Camilla's local machine for earthquake predictability
        MAIN_DIRECTORY = "/Users/camillagiuliabillari/Desktop/github-repositories/cambridge/earthquake-predictability"
    if "tom" in username:
        # Define the main directory on Tom's local machine for earthquake predictability
        MAIN_DIRECTORY = (
            "/home/tom-ratsakatika/VSCode/earthquake-predictability"
        )
    if "new_user" in username:
        MAIN_DIRECTORY = "new_user_working_directory"

GTC_DATA_DIR = f"{MAIN_DIRECTORY}/{REL_DATA_DIR}/gtc_quakes_data"

# Saving plots and models
PLOTS_DIR = f"{MAIN_DIRECTORY}/{REL_PLOTS_DIR}"
RESULTS_DIR = f"{MAIN_DIRECTORY}/{REL_RESULTS_DIR}"

# Labquakes
LABQUAKES_DATA_DIR = f"{GTC_DATA_DIR}/labquakes"
MARONE_LAB_DATA_DIR = f"{LABQUAKES_DATA_DIR}/Marone"
MELEVEEDU_LAB_DATA_DIR = f"{LABQUAKES_DATA_DIR}/MeleVeeduetal2020"

# Slow earthquakes (Cascadia)
SLOWQUAKES_DATA_DIR = f"{GTC_DATA_DIR}/slowquakes"
MICHEL_SLOW_DATA_DIR = f"{SLOWQUAKES_DATA_DIR}/Micheletal2019a"

# Simulation earthquakes (Cascadia)
SYNQUAKES_DATA_DIR = f"{GTC_DATA_DIR}/synquakes"
DALZILIO_SYN_DATA_DIR = f"{SYNQUAKES_DATA_DIR}/DalZilioetal2020"
GUALANDI_SYN_DATA_DIR = f"{SYNQUAKES_DATA_DIR}/Gualandietal2023"
