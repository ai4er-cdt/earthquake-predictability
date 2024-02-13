# credit: Camilla Billari (cgb47@cam.ac.uk)

import getpass
import os
import socket

# data directories
MAIN_DIRECTORY = REL_DATA_DIR = GTC_DATA_DIR = None

# retrieve the hostname of the current system to determine the environment
hostname = socket.gethostname()
# retrieve the current user's login name
username = getpass.getuser()

# check if the hostname indicates the system is part of the ac.uk domain
# else check if the username is present to set up a local development environment
if "ac.uk" in hostname:
    # set the base directory path for JASMIN users, assuming the script runs within the JASMIN infrastructure
    current_dir = "/gws/nopw/j04/ai4er/users/"
    # list specific user directories within the JASMIN environment
    directories = ["cgbill", "pn341", "jpoff", "trr26", "ashine"]
    # determine which of these directories the current user has write access to
    writable_directories = [
        d
        for d in directories
        if os.access(os.path.join(current_dir, d), os.W_OK)
    ]
    # assume the first writable directory belongs to the JASMIN user running the script
    jasmin_user = writable_directories[0]
    # define the main directory path for earthquake predictability
    MAIN_DIRECTORY = (
        f"/gws/nopw/j04/ai4er/users/{jasmin_user}/earthquake-predictability"
    )
    # specify the relative path to the data directory within the main directory
    REL_DATA_DIR = "data"
else:
    # specify the relative path to the local data directory
    REL_DATA_DIR = "data_local"
    if "camilla" in username:
        # define the main directory on Camilla's local machine for earthquake predictability
        MAIN_DIRECTORY = "/Users/camillagiuliabillari/Desktop/github-repositories/cambridge/earthquake-predictability"

GTC_DATA_DIR = f"{MAIN_DIRECTORY}/{REL_DATA_DIR}/gtc_quakes_data"

# labquakes
LABQUAKES_DATA_DIR = f"{GTC_DATA_DIR}/labquakes"
MARONE_LAB_DATA_DIR = f"{LABQUAKES_DATA_DIR}/Marone"
MELEVEEDU_LAB_DATA_DIR = f"{LABQUAKES_DATA_DIR}/MeleVeeduetal2020"

# slow earthquakes (Cascadia)
SLOWQUAKES_DATA_DIR = f"{GTC_DATA_DIR}/slowquakes"
MICHEL_SLOW_DATA_DIR = f"{SLOWQUAKES_DATA_DIR}/Micheletal2019a"

# simulation earthquakes (Cascadia)
SYNQUAKES_DATA_DIR = f"{GTC_DATA_DIR}/synquakes"
DALZILIO_SYN_DATA_DIR = f"{SYNQUAKES_DATA_DIR}/DalZilioetal2020"
GUALANDI_SYN_DATA_DIR = f"{SYNQUAKES_DATA_DIR}/Gualandietal2023"
