"""
Created on Wed Jan 11 10:46:28 2023

@author: vinco
"""

import h5py
import numpy as np


def load_data(exp, dirs, params):
    """
    data_type : 'lab', 'nature', 'synthetic'
    struct_type : 'MeleVeeduetal2020', 'Marone_p4581', 'Marone'
    """

    if params["data_type"] == "lab":
        filename = (
            dirs["data"]
            + params["case_study"]
            + "/"
            + exp
            + "."
            + params["file_format"]
        )

        data = import_data(dirs, filename, params)

        # fields = ['Time (s)', 'Loading Point Displacement (microm)', \
        # 	   'Layer Thickness (mm)', 'Shear Stress (MPa)', \
        # 	   'Normal Stress (MPa)', 'Elastic Corrected Displacement (mm)', \
        # 	   'Friction coefficient mu', 'Shear Strain']

        # observed shear stress
        ShearStressobs = data["ShearStress"]
        # observed normal stress
        NormalStressobs = data["NormStress"]
        # observed displacement
        try:
            ecDispobs = data["ecDisp"]
        except Exception:
            ecDispobs = np.nan * np.ones(ShearStressobs.shape)
        # observed shear strain
        try:
            ShearStrainobs = data["ShearStrain"]
        except Exception:
            ShearStrainobs = np.nan * np.ones(ShearStressobs.shape)

        # number of recorded time stamps
        n_samples = ShearStressobs.shape[0]

        # observed time stamps (referenced to first)
        if params["struct_type"] == "MeleVeeduetal2020":
            t = data["Time"] - data["Time"][0]
        elif params["struct_type"] == "Marone_p4581":
            t = np.linspace(0.0, n_samples * 0.001, n_samples)
        elif params["struct_type"] == "Marone":
            # t = data['Time']
            t = data["Time"] - data["Time"][0]

        # detrend shear stress
        p = np.polyfit(t, ShearStressobs, deg=1)
        ShearStressobs_det = ShearStressobs - (p[0] * t + p[1])
        del p
        # detrend normal stress
        p = np.polyfit(t, NormalStressobs, deg=1)
        # NormalStressobs_det = NormalStressobs - (p[0] * t + p[1])
        del p
        # detrend observed displacement
        # p = np.polyfit(t, ecDispobs, deg=1)
        # # ecDispobs_det = ecDispobs - (p[0] * t + p[1])
        # del p
        # detrend shear strain
        # p = np.polyfit(t, ShearStrainobs, deg=2)
        # # ShearStrainobs_det = ShearStrainobs - (p[0] * t**2 + p[1] * t + p[2])
        # del p

        # observed data
        X = np.array([ShearStressobs_det]).T

        # observed time step
        dt = t[1] - t[0]

        vl = params["vl"]
        if vl is None:
            # estimate loading velocity in microm / s
            LPDispobs = data["LPDisp"]
            p = np.polyfit(t, LPDispobs, deg=1)
            vl = p[0]
            del p

        # Y = np.array([ShearStressobs_det, NormalStressobs_det]).T
        Y = np.array(
            [ShearStressobs, NormalStressobs, ecDispobs, ShearStrainobs]
        ).T
    elif params["data_type"] == "nature":
        f = h5py.File(
            dirs["data"] + params["case_study"] + "." + params["file_format"],
            "r",
        )
        upotrough_asp_ref = f.get("upotrough_asp")
        t_data = f.get("timeline_u")
        t = t_data[()][:, 0]

        dt = t[1] - t[0]

        n_samples = t.shape[0]
        n_segments = upotrough_asp_ref.shape[1]

        upot = dict()
        Upot_raw = np.empty((n_samples, n_segments))
        # Upot = np.empty((n_samples,n_segments))

        if params["segment"] is None:
            segment = 0
        else:
            segment = params["segment"]

        x_data = f[upotrough_asp_ref[0][segment]]
        upot["seg" + str(segment)] = x_data[()]
        Upot_raw[:, segment] = np.sum(upot["seg" + str(segment)], axis=1)

        Y = upot["seg" + str(segment)]
        X = -Upot_raw[:, segment]
        X = X[..., np.newaxis]

        vl = params["vl"]
        if vl is None:
            p = np.polyfit(t, X[:, 0], deg=1)
            vl = p[0]
            del p

    elif params["data_type"] == "synthetic":
        sol_t = np.loadtxt(dirs["data"] + params["case_study"] + "/sol_t.txt")
        sol_u = np.loadtxt(dirs["data"] + params["case_study"] + "/sol_u.txt")
        a = np.loadtxt(dirs["data"] + params["case_study"] + "/a.txt")
        sigman0 = np.loadtxt(
            dirs["data"] + params["case_study"] + "/sigman0.txt"
        )
        vl = params["vl"]
        if vl is None:
            vl = np.loadtxt(dirs["data"] + params["case_study"] + "/v0.txt")
        L1 = np.loadtxt(dirs["data"] + params["case_study"] + "/L1.txt")
        tau0 = np.loadtxt(dirs["data"] + params["case_study"] + "/tau0.txt")
        mu0 = tau0 / sigman0

        Dt = 200
        DT = Dt * vl / L1
        Tend = sol_t[-1]
        Tmin = Tend - DT

        sigman = sigman0 / 1e6

        # % OBSERVED DATA
        vl = vl * 1e6
        t = sol_t[sol_t > Tmin] * L1 / (vl * 1e-6)
        t = t - t[0]
        n_samples = t.shape[0]

        ShearStressobs = np.zeros((n_samples,))
        ShearStressobs = (tau0 + a * sigman0 * sol_u[sol_t > Tmin, 1]) / 1e6
        NormStressobs = sigman + (sigman * a / mu0) * sol_u[sol_t > Tmin, 2]

        p = np.polyfit(t, ShearStressobs, deg=1)
        ShearStressobs_det = ShearStressobs - (p[0] * t + p[1])
        del p

        p = np.polyfit(t, NormStressobs, deg=1)
        NormStressobs_det = NormStressobs - (p[0] * t + p[1])
        del p

        # observed data
        X = np.array([ShearStressobs_det]).T

        # observed time step
        dt = t[1] - t[0]

        Y = np.array([ShearStressobs_det, NormStressobs_det]).T

    elif params["data_type"] == "gnss":
        data_e = np.loadtxt(
            dirs["data"] + params["case_study"] + "pgc5_E_wtrend.data"
        )
        data_n = np.loadtxt(
            dirs["data"] + params["case_study"] + "pgc5_N_wtrend.data"
        )

        te = data_e[:, 0]
        tn = data_n[:, 0]

        t = np.intersect1d(te, tn)
        inde = np.arange(te.shape[0])[np.in1d(te, t)]
        indn = np.arange(tn.shape[0])[np.in1d(tn, t)]

        e = data_e[inde, 1]
        se = data_e[inde, 2]
        n = data_n[indn, 1]
        sn = data_n[indn, 2]

        dt = t[1] - t[0]

        Y = np.array([t, e, n, se, sn])
        X = -0.5 * (e + n)
        X = X[..., np.newaxis]

        vl = params["vl"]
        if vl is None:
            p = np.polyfit(t, X[:, 0], deg=1)
            vl = p[0]
            del p

    return X, Y, t, dt, vl


def import_data(dirs, filename, parameters):
    struct = parameters["struct_type"]
    file_format = parameters["file_format"]

    if struct == "MeleVeeduetal2020":
        if file_format == "txt":
            f = open(filename, "r")
            L = 0
            for line in f:
                L += 1

            f.close()

            Nheaders = parameters["Nheaders"]
            L = L - Nheaders
            Rec = np.empty([L, 1])
            LPDisp = np.empty([L, 1])
            LayerThick = np.empty([L, 1])
            ShearStress = np.empty([L, 1])
            NormStress = np.empty([L, 1])
            OnBoard = np.empty([L, 1])
            Time = np.empty([L, 1])
            Rec_float = np.empty([L, 1])
            TimeOnBoard = np.empty([L, 1])
            ecDisp = np.empty([L, 1])
            mu = np.empty([L, 1])
            ShearStrain = np.empty([L, 1])
            slip_velocity = np.empty([L, 1])
            ll = -1
            tt = -1
            f = open(filename, "r")
            for line in f:
                ll += 1
                columns = line.split()
                if ll > Nheaders - 1:
                    tt += 1
                    Rec[tt] = int(columns[0])
                    LPDisp[tt] = float(columns[1])
                    LayerThick[tt] = float(columns[2])
                    ShearStress[tt] = float(columns[3])
                    NormStress[tt] = float(columns[4])
                    OnBoard[tt] = float(columns[5])
                    Time[tt] = float(columns[6])
                    Rec_float[tt] = float(columns[7])
                    TimeOnBoard[tt] = float(columns[8])
                    ecDisp[tt] = float(columns[9])
                    mu[tt] = float(columns[10])
                    ShearStrain[tt] = float(columns[11])
                    slip_velocity[tt] = float(columns[12])

            f.close()

            ind_keep = [
                i
                for i, x in enumerate(Time[:, 0])
                if x >= parameters["t0"] and x <= parameters["tend"]
            ]

            data_output = {
                "Rec": Rec[ind_keep, 0],
                "LPDisp": LPDisp[ind_keep, 0],
                "LayerThick": LayerThick[ind_keep, 0],
                "ShearStress": ShearStress[ind_keep, 0],
                "NormStress": NormStress[ind_keep, 0],
                "OnBoard": OnBoard[ind_keep, 0],
                "Time": Time[ind_keep, 0],
                "Rec_float": Rec_float[ind_keep, 0],
                "TimeOnBoard": TimeOnBoard[ind_keep, 0],
                "ecDisp": ecDisp[ind_keep, 0],
                "mu": mu[ind_keep, 0],
                "ShearStrain": ShearStrain[ind_keep, 0],
                "OnBoarddot": slip_velocity[ind_keep, 0],
            }
    elif struct == "Marone":
        if file_format == "txt":
            f = open(filename, "r")
            L = 0
            for line in f:
                L += 1

            f.close()

            Nheaders = parameters["Nheaders"]
            L = L - Nheaders
            Rec = np.empty([L, 1])
            LPDisp = np.empty([L, 1])
            ShearStress = np.empty([L, 1])
            NormDisp = np.empty([L, 1])
            NormStress = np.empty([L, 1])
            Time = np.empty([L, 1])
            mu = np.empty([L, 1])
            LayerThick = np.empty([L, 1])
            ecDisp = np.empty([L, 1])
            ll = -1
            tt = -1
            f = open(filename, "r")
            for line in f:
                ll += 1
                columns = line.split()
                if ll > Nheaders - 1:
                    tt += 1
                    Rec[tt] = int(columns[0][:-1])
                    LPDisp[tt] = float(columns[1][:-1])
                    ShearStress[tt] = float(columns[2][:-1])
                    NormDisp[tt] = float(columns[3][:-1])
                    NormStress[tt] = float(columns[4][:-1])
                    Time[tt] = float(columns[5][:-1])
                    mu[tt] = float(columns[6][:-1])
                    LayerThick[tt] = float(columns[7][:-1])
                    ecDisp[tt] = float(columns[8])

            f.close()

            ind_keep = [
                i
                for i, x in enumerate(Time[:, 0])
                if x >= parameters["t0"] and x <= parameters["tend"]
            ]

            data_output = {
                "Rec": Rec[ind_keep, 0],
                "LPDisp": LPDisp[ind_keep, 0],
                "ShearStress": ShearStress[ind_keep, 0],
                "NormDisp": NormDisp[ind_keep, 0],
                "NormStress": NormStress[ind_keep, 0],
                "Time": Time[ind_keep, 0],
                "mu": mu[ind_keep, 0],
                "LayerThick": LayerThick[ind_keep, 0],
                "ecDisp": ecDisp[ind_keep, 0],
            }
    elif struct == "Marone_p4581":
        if file_format == "txt":
            f = open(filename, "r")
            L = 0
            for line in f:
                L += 1

            f.close()

            Nheaders = parameters["Nheaders"]
            L = L - Nheaders
            Rec = np.empty([L, 1])
            LPDisp = np.empty([L, 1])
            ShearStress = np.empty([L, 1])
            NormDisp = np.empty([L, 1])
            NormStress = np.empty([L, 1])
            Time = np.empty([L, 1])
            mu = np.empty([L, 1])
            LayerThick = np.empty([L, 1])
            ecDisp = np.empty([L, 1])
            ll = -1
            tt = -1
            f = open(filename, "r")
            for line in f:
                ll += 1
                columns = line.split()
                if ll > Nheaders - 1:
                    tt += 1
                    # pdb.set_trace()
                    Rec[tt] = int(columns[0])
                    LPDisp[tt] = float(columns[1])
                    ShearStress[tt] = float(columns[2])
                    NormDisp[tt] = float(columns[3])
                    NormStress[tt] = float(columns[4])
                    Time[tt] = float(columns[5])

            f.close()

            ind_keep = [
                i
                for i, x in enumerate(Time[:, 0])
                if x >= parameters["t0"] and x <= parameters["tend"]
            ]

            data_output = {
                "Rec": Rec[ind_keep, 0],
                "LPDisp": LPDisp[ind_keep, 0],
                "ShearStress": ShearStress[ind_keep, 0],
                "NormDisp": NormDisp[ind_keep, 0],
                "NormStress": NormStress[ind_keep, 0],
                "Time": Time[ind_keep, 0],
            }
    return data_output


def add_noise(X, A, rand_seed=None):
    """
    Add white noise to data.
    Parameters
    ----------
    X : 2-dim numpy array
            Input data.
            Format: n_samples x n_var, with n_samples number of epochs,
            n_var number of time series.
    A : scalar
            Noise amplitude.
    rand_seed : scalar (default : None)
            Random seed for noise generation.
    Returns
    -------
    X + A*np.random.randn(n_samples,n_var)
    """
    if rand_seed is not None:
        np.random.rand(rand_seed)
    return X + np.random.normal(loc=0.0, scale=A, size=X.shape)
