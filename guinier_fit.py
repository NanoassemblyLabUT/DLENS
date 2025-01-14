import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from math import sqrt, exp


def root_mean_squared_error(X: np.ndarray, Y: np.ndarray) -> float:
    return np.sqrt(np.average(np.square(X - Y))), np.sqrt(np.std(np.square(X - Y), ddof=1))


def mean_absolute_percentage_error(X: np.ndarray, Y: np.ndarray) -> float:
    return 100*np.average(np.absolute((X - Y)/X)), 100*np.std(np.absolute((X - Y)/X), ddof=1)


def get_data(shape: str) -> tuple[np.ndarray]:
    
    cwd = os.getcwd()
    file_dir = os.path.join(cwd, 'Data')

    if shape == 'sphere':
        name = '2024_09_12_disperse_spheroid_core_shell_yj5782_0.npy'
    elif shape == 'cylinder':
        name = '2024_09_11_disperse_cylinder_core_shell_yj5782_0.npy'
    else:
        pass
    
    file_name = os.path.join(file_dir, name)
    
    data = np.load(file_name)
    
    d = []

    for i in range(data.shape[0]):
        if True in np.isnan(data[i, :]):
            d.append(i)

    np.delete(data, d, 0)
    
    X = data[:, 4:]
    y = data[:, 0:4]
        
    return X, y


def linear_least_square(X: np.ndarray, Y: np.ndarray) -> tuple[float]:
    
    a_11 = np.sum(np.square(X))
    a_12 = a_21 = np.sum(X)
    a_22 = X.size
    
    b_1 = np.sum(X*Y)
    b_2 = np.sum(Y)
    
    A = np.array(((a_11, a_12), 
                  (a_21, a_22)))
    b = np.array((b_1, b_2)).reshape(-1, 1)
    
    v = np.matmul(np.linalg.inv(A), b)
    
    return v[0, 0], v[1, 0]


def quadratic_least_square(X: np.ndarray, Y: np.ndarray) -> tuple[float]:
    
    a_11 = np.sum(np.power(X, 4))
    a_12 = a_21 = np.sum(np.power(X, 3))
    a_13 = a_22 = a_31 = np.sum(np.square(X))
    a_23 = a_32 = np.sum(X)
    a_33 = X.size
    
    b_1 = np.sum(np.square(X)*Y)
    b_2 = np.sum(X*Y)
    b_3 = np.sum(Y)
    
    A = np.array(((a_11, a_12, a_13), 
                  (a_21, a_22, a_23), 
                  (a_31, a_32, a_33)))
    b = np.array((b_1, b_2, b_3)).reshape(-1, 1)
    
    v = np.matmul(np.linalg.inv(A), b)
    
    return v[0, 0], v[1, 0], v[2, 0]


def Lennard_Jones_fitting(X: np.ndarray, Y: np.ndarray) -> tuple[float]:
    
    c_0 = np.sum(np.power(X, -24))
    c_1 = np.sum(np.power(X, -18))
    c_2 = np.sum(np.power(X, -12))
    c_3 = np.sum(np.power(X, -6))
    c_4 = X.size
    
    b_1 = np.sum(Y*np.power(X, -12))
    b_2 = np.sum(Y*np.power(X, -6))
    b_3 = np.sum(Y)

    A = np.array(((c_0, -c_1, c_2), 
                  (c_1, -c_2, c_3), 
                  (c_2, -c_3, c_4)))
    b = np.array((b_1, b_2, b_3)).reshape(-1, 1)
    
    v = np.matmul(np.linalg.inv(A), b)
    
    return v[0, 0], v[1, 0], v[2, 0]


def data_processing(q: np.ndarray, I: np.ndarray) -> tuple[float]:
    
    X = np.square(q)
    Y = np.log(I)
    
    m, b = linear_least_square(X=X, Y=Y)
    R_g = sqrt(-3*m)
    I_0 = exp(b)
    
    return R_g, I_0


def cutoff(q: np.ndarray, I: np.ndarray) -> tuple[np.ndarray]:
    
    cwd = os.getcwd()
    model_dir = os.path.join(cwd, 'Models')
    model_name = '2024_11_18_SCNN_qr_0.keras'
    
    model_path = os.path.join(model_dir, model_name)
    model = tf.keras.models.load_model(model_path, compile=False)
    
    X = I
    X[X <= 0] = np.min(X[X >= 0])
    X = 1 + np.log10(X)/2
    X = np.tanh(X)
    X = X[np.newaxis, :, np.newaxis]
    
    r = 256*model.predict(X)[0]
    qr = q*r
        
    q_new = q[qr <= 3]
    I_new = I[qr <= 3]

    return q_new, I_new


def guinier_fit(q: np.ndarray, I: np.ndarray) -> tuple[float]:
    
    q, I = cutoff(q=q, I=I)
    R_g, I_0 = data_processing(q=q, I=I)
    
    return R_g, I_0


def compare(*args, **kwargs) -> None:
    
    q_log_arr = np.arange(-2, 0, np.true_divide(1, 128))
    q = np.power(10, q_log_arr - 2*np.log10(2))
    
    shape = 'sphere'
    
    I, y = get_data(shape=shape)
    
    X = I
    X[X <= 0] = np.min(X[X >= 0])
    X = 1 + np.log10(X)/2
    X = np.tanh(X)
    X = X[:, :, np.newaxis]
    
    cwd = os.getcwd()
    model_dir = os.path.join(cwd, 'Models')
    
    file_name_sr = '2024_11_18_sphere_CPNN_Radius_0.keras'
    file_name_se = '2024_11_18_sphere_CPNN_AspectRatio_0.keras'
    file_name_sd = '2024_12_10_sphere_CPNN_PDI_0.keras'

    file_path_sr = os.path.join(model_dir, file_name_sr)
    file_path_se = os.path.join(model_dir, file_name_se)
    file_path_sd = os.path.join(model_dir, file_name_sd)

    model_sr = tf.keras.models.load_model(file_path_sr, compile=False)
    model_se = tf.keras.models.load_model(file_path_se, compile=False)
    model_sd = tf.keras.models.load_model(file_path_sd, compile=False)

    r_0s = np.zeros(shape=(X.shape[0], ))
    r_1s = np.zeros(shape=(X.shape[0], ))
    r_2s = np.zeros(shape=(X.shape[0], ))
    
    rs = y[:, 0]
    es = y[:, 1]
    ds = y[:, 2]
    
    As = Bs = rs
    Cs = es*rs
    r_0s += np.sqrt((np.square(As) + np.square(Bs) + np.square(Cs))/5)
    
    for i in range(rs.size):

        r_g = r_0s[i]
        
        qr = q*r_g
        
        q_temp = q[qr <= 1.3]
        I_temp = I[i, :]
        I_temp = I_temp[qr <= 1.3]
                
        r_g, _ = data_processing(q=q_temp, I=I_temp)
        r_2s[i] += r_g
    
    r_pred = model_sr.predict(X)
    e_pred = model_se.predict(X)
    d_pred = model_sd.predict(X)
    
    Rs = 256*r_pred[:, 0]
    Es = 2*e_pred[:, 0]
    Ds = np.power(10, -4*d_pred[:, 0])/2
    
    gamma = 1 + 0.08*Ds/0.5
    
    As = Bs = Rs
    Cs = Es*Rs
    r_1s += np.sqrt((np.square(As) + np.square(Bs) + np.square(Cs))/5)
    r_1s = r_1s/gamma
    
    RMSE_DL = root_mean_squared_error(X=r_0s, Y=r_1s)
    RMSE_GN = root_mean_squared_error(X=r_0s, Y=r_2s)
    
    MAPE_DL = mean_absolute_percentage_error(X=r_0s, Y=r_1s)
    MAPE_GN = mean_absolute_percentage_error(X=r_0s, Y=r_2s)
    
    print(RMSE_DL, MAPE_DL)
    print(RMSE_GN, MAPE_GN)
    
    RMSEs = [RMSE_DL, RMSE_GN]
    MAPEs = [MAPE_DL, MAPE_GN]
    
    methods = ['Deep Learning', 'Guinier Fit']
    color = ['blue', 'red']
    
    fig, axes = plt.subplots(1, 2, figsize=(6, 4))
    ax_0, ax_1 = axes[0], axes[1]
    
    ax_0.bar(methods, RMSEs, color=color)
    ax_0.set_xlabel('(a)')
    ax_0.set_ylabel('Root-Mean-Squared Error (Å)')
    
    ax_1.bar(methods, MAPEs, color=color)
    ax_1.set_xlabel('(b)')
    ax_1.set_ylabel('Mean Absolute Percentage Error (%)')
    
    plot_dir = os.path.join(cwd, 'assets')
    name = os.path.join(plot_dir, 'guinier.png')
    fig.savefig(name, bbox_inches='tight')
    
    plt.show()
    
    plt.figure()
    
    plt.scatter(ds, 100*(r_1s - r_0s)/r_0s, s=0.5, label='Deep Learning')
    plt.scatter(ds, 100*(r_2s - r_0s)/r_0s, s=0.5, label='Guinier Fit')
    plt.xscale('log')
    plt.xlabel('PDI')
    plt.ylabel('Percentage Error (%)')
    plt.legend()

    plt.show()
    
    num = 64
    
    PDIs = np.linspace(np.log10(1e-3), np.log10(5e-1), num)
    
    XD = np.power(10, PDIs + (np.log10(5e-1) - np.log10(1e-3))/128)[:-1]
    PDIs = np.power(10, PDIs)
    eds = np.zeros(shape=(num - 1, ))
    egs = np.zeros(shape=(num - 1, ))
    
    for i in range(num - 1):
        truths = np.logical_and(ds >= PDIs[i], ds < PDIs[i + 1])
        eds[i] += np.average((r_1s[truths] - r_0s[truths])/r_0s[truths])
        egs[i] += np.average((r_2s[truths] - r_0s[truths])/r_0s[truths])
        
    a, b, c = quadratic_least_square(X=XD, Y=eds)
    d, e = linear_least_square(X=XD, Y=egs)
    
    temp_0 = a*np.square(XD) + b*XD + c
    # temp_1 = d*XD + e

    plt.figure()
    
    plt.plot(XD, eds, label='Deep Learning')
    plt.plot(XD, temp_0, label='Mock 1')
    # plt.plot(XD, egs, label='Guinier Fit')
    # plt.plot(XD, temp_1, label='Mock 2')
    # plt.xscale('log')
    plt.xlabel('PDI')
    plt.ylabel('Percentage Error (%)')
    plt.legend()
    
    plt.show()
    
    corr_0 = a*np.square(Ds) + b*Ds + c + 1
    # corr_1 = d*Ds + e + 1
    
    plt.figure()
    
    plt.scatter(ds, 100*(r_1s - r_0s)/r_0s, s=0.5, label='Deep Learning')
    plt.scatter(ds, 100*(r_1s/corr_0 - r_0s)/r_0s, s=0.5, label='Corrected Deep Learning')
    plt.scatter(ds, 100*(r_2s - r_0s)/r_0s, s=0.5, label='Guinier Fit')
    plt.xscale('log')
    plt.xlabel('PDI')
    plt.ylabel('Percentage Error (%)')
    plt.legend()

    plt.show()
    
    RMSE_DL = root_mean_squared_error(X=r_0s, Y=r_1s)
    RMSE_CR = root_mean_squared_error(X=r_0s, Y=r_1s/corr_0)
    RMSE_GN = root_mean_squared_error(X=r_0s, Y=r_2s)
    
    MAPE_DL = mean_absolute_percentage_error(X=r_0s, Y=r_1s)
    MAPE_CR = mean_absolute_percentage_error(X=r_0s, Y=r_1s/corr_0)
    MAPE_GN = mean_absolute_percentage_error(X=r_0s, Y=r_2s)
    
    print(RMSE_DL, RMSE_CR, RMSE_GN)
    print(MAPE_DL, MAPE_CR, MAPE_GN)
    
    RMSEs = [RMSE_DL, RMSE_CR, RMSE_GN]
    MAPEs = [MAPE_DL, MAPE_CR, MAPE_GN]
    
    methods = ['Deep\nLearning', 'Corrected\nDeep\nLearning', 'Guinier\nFit']
    color = ['blue', 'green', 'red']
    
    fig, axes = plt.subplots(1, 2, figsize=(6, 4))
    ax_0, ax_1 = axes[0], axes[1]
    
    ax_0.bar(methods, RMSEs, color=color)
    ax_0.set_xlabel('(a)')
    ax_0.set_ylabel('Root-Mean-Squared Error (Å)')
    
    ax_1.bar(methods, MAPEs, color=color)
    ax_1.set_xlabel('(b)')
    ax_1.set_ylabel('Mean Absolute Percentage Error (%)')
    
    plt.show()

    return None


def sphere_correction(*args, **kwargs) -> None:
    
    q_log_arr = np.arange(-2, 0, np.true_divide(1, 128))
    q = np.power(10, q_log_arr - 2*np.log10(2))
    
    shape = 'sphere'
    
    I, y = get_data(shape=shape)
    
    X = I
    X[X <= 0] = np.min(X[X >= 0])
    X = 1 + np.log10(X)/2
    X = np.tanh(X)
    X = X[:, :, np.newaxis]
    
    cwd = os.getcwd()
    model_dir = os.path.join(cwd, 'Models')
    
    file_name_sr = '2024_11_18_sphere_CPNN_Radius_0.keras'
    file_name_se = '2024_11_18_sphere_CPNN_AspectRatio_0.keras'
    file_name_sd = '2024_12_10_sphere_CPNN_PDI_0.keras'

    file_path_sr = os.path.join(model_dir, file_name_sr)
    file_path_se = os.path.join(model_dir, file_name_se)
    file_path_sd = os.path.join(model_dir, file_name_sd)

    model_sr = tf.keras.models.load_model(file_path_sr, compile=False)
    model_se = tf.keras.models.load_model(file_path_se, compile=False)
    model_sd = tf.keras.models.load_model(file_path_sd, compile=False)

    r_0s = np.zeros(shape=(X.shape[0], ))
    r_1s = np.zeros(shape=(X.shape[0], ))
    r_2s = np.zeros(shape=(X.shape[0], ))
    
    rs = y[:, 0]
    es = y[:, 1]
    ds = y[:, 2]
    
    As = Bs = rs
    Cs = es*rs
    r_0s += np.sqrt((np.square(As) + np.square(Bs) + np.square(Cs))/5)
    
    for i in range(rs.size):

        r_g = r_0s[i]
        
        qr = q*r_g
        
        q_temp = q[qr <= 1.3]
        I_temp = I[i, :]
        I_temp = I_temp[qr <= 1.3]
                
        r_g, _ = data_processing(q=q_temp, I=I_temp)
        r_2s[i] += r_g
    
    r_pred = model_sr.predict(X)
    e_pred = model_se.predict(X)
    d_pred = model_sd.predict(X)
    
    Rs = 256*r_pred[:, 0]
    Es = 2*e_pred[:, 0]
    Ds = np.power(10, -4*d_pred[:, 0])/2
        
    As = Bs = Rs
    Cs = Es*Rs
    r_1s += np.sqrt((np.square(As) + np.square(Bs) + np.square(Cs))/5)
    
    RMSE_DL, _ = root_mean_squared_error(X=r_0s, Y=r_1s)
    RMSE_GN, _ = root_mean_squared_error(X=r_0s, Y=r_2s)
    
    MAPE_DL, _ = mean_absolute_percentage_error(X=r_0s, Y=r_1s)
    MAPE_GN, _ = mean_absolute_percentage_error(X=r_0s, Y=r_2s)
    
    print(RMSE_DL, MAPE_DL)
    print(RMSE_GN, MAPE_GN)
    
    RMSEs = [RMSE_DL, RMSE_GN]
    MAPEs = [MAPE_DL, MAPE_GN]
    
    methods = ['Deep Learning', 'Guinier Fit']
    color = ['blue', 'red']
    
    fig, axes = plt.subplots(1, 2, figsize=(6, 4))
    ax_0, ax_1 = axes[0], axes[1]
    
    ax_0.bar(methods, RMSEs, color=color)
    ax_0.set_xlabel('(a)')
    ax_0.set_ylabel('Root-Mean-Squared Error (Å)')
    
    ax_1.bar(methods, MAPEs, color=color)
    ax_1.set_xlabel('(b)')
    ax_1.set_ylabel('Mean Absolute Percentage Error (%)')
    
    plot_dir = os.path.join(cwd, 'assets')
    name = os.path.join(plot_dir, 'guinier.png')
    fig.savefig(name, bbox_inches='tight')
    
    plt.show()
    
    plt.figure()
    
    plt.scatter(ds, 100*(r_1s - r_0s)/r_0s, s=0.5, label='Deep Learning')
    plt.scatter(ds, 100*(r_2s - r_0s)/r_0s, s=0.5, label='Guinier Fit')
    plt.xscale('log')
    plt.xlabel('PDI')
    plt.ylabel('Percentage Error (%)')
    plt.legend()

    plt.show()
    
    num = 64
    
    PDIs = np.linspace(np.log10(1e-3), np.log10(5e-1), num)
    
    XD = np.power(10, PDIs + (np.log10(5e-1) - np.log10(1e-3))/128)[:-1]
    PDIs = np.power(10, PDIs)
    eds = np.zeros(shape=(num - 1, ))
    egs = np.zeros(shape=(num - 1, ))
    
    for i in range(num - 1):
        truths = np.logical_and(ds >= PDIs[i], ds < PDIs[i + 1])
        eds[i] += np.average((r_1s[truths] - r_0s[truths])/r_0s[truths])
        egs[i] += np.average((r_2s[truths] - r_0s[truths])/r_0s[truths])
        
    a, b, c = quadratic_least_square(X=XD, Y=eds)
    
    print(a, b, c)
    
    temp_0 = a*np.square(XD) + b*XD + c

    plt.figure()
    
    plt.scatter(XD, 100*eds, label='Deep Learning Error')
    plt.plot(XD, 100*temp_0, label='Approximation', color='k', linestyle='dotted')
    plt.xlabel('PDI')
    plt.ylabel('Percentage Error (%)')
    plt.legend()
    
    plt.show()
    
    corr_0 = a*np.square(Ds) + b*Ds + c + 1
    
    RMSE_DL, STD_RD = root_mean_squared_error(X=r_0s, Y=r_1s)
    RMSE_CR, STD_RC = root_mean_squared_error(X=r_0s, Y=r_1s/corr_0)
    RMSE_GN, STD_RG = root_mean_squared_error(X=r_0s, Y=r_2s)
    
    MAPE_DL, STD_MD = mean_absolute_percentage_error(X=r_0s, Y=r_1s)
    MAPE_CR, STD_MC = mean_absolute_percentage_error(X=r_0s, Y=r_1s/corr_0)
    MAPE_GN, STD_MG = mean_absolute_percentage_error(X=r_0s, Y=r_2s)
    
    print(RMSE_DL, RMSE_CR, RMSE_GN)
    print(MAPE_DL, MAPE_CR, MAPE_GN)
    
    RMSEs = [RMSE_DL, RMSE_CR, RMSE_GN]
    STD_R = [STD_RD, STD_RC, STD_RG]
    MAPEs = [MAPE_DL, MAPE_CR, MAPE_GN]
    STD_M = [STD_MD, STD_MC, STD_MG]

    methods = ['Deep\nLearning', 'Corrected\nDeep\nLearning', 'Guinier\nFit']
    color = ['blue', 'green', 'red']
    
    fig, axes = plt.subplots(1, 2, figsize=(6, 4))
    ax_0, ax_1 = axes[0], axes[1]
    
    ax_0.bar(methods, RMSEs, color=color)
    ax_0.set_xlabel('(a)')
    ax_0.set_ylabel('Root-Mean-Squared Error (Å)')

    ax_1.bar(methods, MAPEs, color=color)
    ax_1.set_xlabel('(b)')
    ax_1.set_ylabel('Mean Absolute Percentage Error (%)')

    plt.show()

    return None


def sphere_check(*args, **kwargs) -> None:
    
    shape = 'sphere'
    
    I, y = get_data(shape=shape)
    
    X = I
    X[X <= 0] = np.min(X[X >= 0])
    X = 1 + np.log10(X)/2
    X = np.tanh(X)
    X = X[:, :, np.newaxis]
    
    cwd = os.getcwd()
    model_dir = os.path.join(cwd, 'Models')
    
    file_name_sr = '2024_11_18_sphere_CPNN_Radius_0.keras'
    file_name_se = '2024_11_18_sphere_CPNN_AspectRatio_0.keras'
    file_name_sd = '2024_12_10_sphere_CPNN_PDI_0.keras'

    file_path_sr = os.path.join(model_dir, file_name_sr)
    file_path_se = os.path.join(model_dir, file_name_se)
    file_path_sd = os.path.join(model_dir, file_name_sd)

    model_sr = tf.keras.models.load_model(file_path_sr, compile=False)
    model_se = tf.keras.models.load_model(file_path_se, compile=False)
    model_sd = tf.keras.models.load_model(file_path_sd, compile=False)
    
    rs = y[:, 0]
    es = y[:, 1]
    ds = y[:, 2]
    
    r_pred = model_sr.predict(X)
    e_pred = model_se.predict(X)
    d_pred = model_sd.predict(X)
    
    Rs = 256*r_pred[:, 0]
    Es = 2*e_pred[:, 0]
    Ds = np.power(10, -4*d_pred[:, 0])/2
    
    num = 64
    
    PDIs = np.linspace(np.log10(1e-3), np.log10(5e-1), num)
    
    XD = np.power(10, PDIs + (np.log10(5e-1) - np.log10(1e-3))/128)[:-1]
    PDIs = np.power(10, PDIs)
    e_rs = np.zeros(shape=(num - 1, ))
    e_es = np.zeros(shape=(num - 1, ))
    e_ds = np.zeros(shape=(num - 1, ))
    
    for i in range(num - 1):
        truths = np.logical_and(ds >= PDIs[i], ds < PDIs[i + 1])
        e_rs[i] += np.average((Rs[truths] - rs[truths])/rs[truths])
        e_es[i] += np.average((Es[truths] - es[truths])/es[truths])
        e_ds[i] += np.average((Ds[truths] - ds[truths])/ds[truths])
        
    a_0, b_0, c_0 = np.polyfit(x=XD, y=e_rs, deg=2)
    a_1, b_1, c_1 = np.polyfit(x=XD, y=e_es, deg=2)
    a_2, b_2, c_2, d_2 = np.polyfit(x=np.log10(XD), y=e_ds, deg=3)
    
    temp_0 = a_0*np.square(XD) + b_0*XD + c_0
    temp_1 = a_1*np.square(XD) + b_1*XD + c_1
    temp_2 = a_2*np.power(np.log10(XD), 3) + b_2*np.square(np.log10(XD)) + c_2*np.log10(XD) + d_2
        
    plt.figure()
    
    # plt.scatter(ds, 100*(Rs - rs)/rs, s=0.5, label='Radius')
    plt.scatter(XD, e_rs, label='Average', s=1.0)
    plt.plot(XD, temp_0, label='Approximation', linestyle='dotted', color='k')
    # plt.xscale('log')
    plt.xlabel('PDI')
    plt.ylabel('Percentage Error (%)')
    plt.legend()

    plt.show()
    
    plt.figure()
    
    # plt.scatter(ds, 100*(Es - es)/es, s=0.5, label='Aspect Ratio')
    plt.scatter(XD, e_es, label='Average', s=1.0)
    plt.plot(XD, temp_1, label='Approximation', linestyle='dotted', color='k')
    # plt.xscale('log')
    plt.xlabel('PDI')
    plt.ylabel('Percentage Error (%)')
    plt.legend()

    plt.show()
    
    plt.figure()
    
    # plt.scatter(ds, 100*(Ds - ds)/ds, s=0.5, label='PDI')
    plt.scatter(XD, e_ds, label='Average', s=1.0)
    plt.plot(XD, temp_2, label='Approximation', linestyle='dotted', color='k')
    # plt.xscale('log')
    plt.xlabel('PDI')
    plt.ylabel('Percentage Error (%)')
    plt.legend()

    plt.show()
    
    corr_2 = a_2*np.power(np.log10(Ds), 3) + b_2*np.square(np.log10(Ds)) + c_2*np.log10(Ds) + d_2 + 1
    Ds_corr = Ds/corr_2
        
    corr_0 = a_0*np.square(Ds_corr) + b_0*Ds_corr + c_0 + 1
    corr_1 = a_1*np.square(Ds_corr) + b_1*Ds_corr + c_1 + 1
    
    Rs_corr = Rs/corr_0
    Es_corr = Es/corr_1
    
    RMSE_RR, _ = root_mean_squared_error(X=rs, Y=Rs)
    RMSE_RC, _ = root_mean_squared_error(X=rs, Y=Rs_corr)
    RMSE_ER, _ = root_mean_squared_error(X=es, Y=Es)
    RMSE_EC, _ = root_mean_squared_error(X=es, Y=Es_corr)
    RMSE_DR, _ = root_mean_squared_error(X=ds, Y=Ds)
    RMSE_DC, _ = root_mean_squared_error(X=ds, Y=Ds_corr)
    
    MAPE_RR, _ = mean_absolute_percentage_error(X=rs, Y=Rs)
    MAPE_RC, _ = mean_absolute_percentage_error(X=rs, Y=Rs_corr)
    MAPE_ER, _ = mean_absolute_percentage_error(X=es, Y=Es)
    MAPE_EC, _ = mean_absolute_percentage_error(X=es, Y=Es_corr)
    MAPE_DR, _ = mean_absolute_percentage_error(X=ds, Y=Ds)
    MAPE_DC, _ = mean_absolute_percentage_error(X=ds, Y=Ds_corr)
    
    print(RMSE_RR, RMSE_RC)
    print(RMSE_ER, RMSE_EC)
    print(RMSE_DR, RMSE_DC)

    print(MAPE_RR, MAPE_RC)
    print(MAPE_ER, MAPE_EC)
    print(MAPE_DR, MAPE_DC)
    
    RMSEs = [RMSE_RR, RMSE_RC, RMSE_ER, RMSE_EC, RMSE_DR, RMSE_DC]
    MAPEs = [MAPE_RR, MAPE_RC, MAPE_ER, MAPE_EC, MAPE_DR, MAPE_DC]

    params = ['Radius\nOriginal', 'Radius\nCorrected', 
               'Ratio\nOriginal', 'Ratio\nCorrected', 
               'PDI\nOriginal', 'PDI\nCorrected']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax_0, ax_1 = axes[0], axes[1]
    
    ax_0.bar(params, RMSEs)
    ax_0.set_xlabel('(a)')
    ax_0.set_ylabel('Root-Mean-Squared Error (Å)')

    ax_1.bar(params, MAPEs)
    ax_1.set_xlabel('(b)')
    ax_1.set_ylabel('Mean Absolute Percentage Error (%)')

    plt.show()
    
    return None


def cylinder_correction(*args, **kwargs) -> None:
    
    q_log_arr = np.arange(-2, 0, np.true_divide(1, 128))
    q = np.power(10, q_log_arr - 2*np.log10(2))
    
    shape = 'cylinder'
    
    I, y = get_data(shape=shape)
    
    X = I
    X[X <= 0] = np.min(X[X >= 0])
    X = 1 + np.log10(X)/2
    X = np.tanh(X)
    X = X[:, :, np.newaxis]
    
    cwd = os.getcwd()
    model_dir = os.path.join(cwd, 'Models')
    
    file_name_sr = '2024_11_18_cylinder_CPNN_Radius_0.keras'
    file_name_se = '2024_11_18_cylinder_CPNN_AspectRatio_0.keras'
    file_name_sd = '2024_12_10_cylinder_CPNN_PDI_0.keras'

    file_path_sr = os.path.join(model_dir, file_name_sr)
    file_path_se = os.path.join(model_dir, file_name_se)
    file_path_sd = os.path.join(model_dir, file_name_sd)

    model_sr = tf.keras.models.load_model(file_path_sr, compile=False)
    model_se = tf.keras.models.load_model(file_path_se, compile=False)
    model_sd = tf.keras.models.load_model(file_path_sd, compile=False)
    
    rs = y[:, 0]
    ls = y[:, 1]
    ds = y[:, 2]
    
    r_pred = model_sr.predict(X)
    l_pred = model_se.predict(X)
    d_pred = model_sd.predict(X)
    
    Rs = 256*r_pred[:, 0]
    Ls = 16*l_pred[:, 0]*Rs
    Ds = np.power(10, -4*d_pred[:, 0])/2
    
    r_0s = np.sqrt(np.square(rs)/2 + np.square(ls)/12)
    r_1s = np.sqrt(np.square(Rs)/2 + np.square(Ls)/12)
    r_2s = np.zeros(shape=(X.shape[0], ))

    for i in range(rs.size):

        r_g = r_0s[i]
        
        qr = q*r_g
        
        q_temp = q[qr <= 1.0]
        I_temp = I[i, :]
        I_temp = I_temp[qr <= 1.0]
        
        if q_temp.size < 8:
            q_temp = q[:8]
            I_temp = I[i, :]
            I_temp = I_temp[:8]
                
        r_g, _ = data_processing(q=q_temp, I=I_temp)
        r_2s[i] += r_g

    RMSE_DL, _ = root_mean_squared_error(X=r_0s, Y=r_1s)
    RMSE_GN, _ = root_mean_squared_error(X=r_0s, Y=r_2s)
    
    MAPE_DL, _ = mean_absolute_percentage_error(X=r_0s, Y=r_1s)
    MAPE_GN, _ = mean_absolute_percentage_error(X=r_0s, Y=r_2s)
    
    print(RMSE_DL, RMSE_GN)
    print(MAPE_DL, MAPE_GN)
    
    RMSEs = [RMSE_DL, RMSE_GN]
    MAPEs = [MAPE_DL, MAPE_GN]
    
    methods = ['Deep Learning', 'Guinier Fit']
    color = ['blue', 'red']
    
    fig, axes = plt.subplots(1, 2, figsize=(6, 4))
    ax_0, ax_1 = axes[0], axes[1]
    
    ax_0.bar(methods, RMSEs, color=color)
    ax_0.set_xlabel('(a)')
    ax_0.set_ylabel('Root-Mean-Squared Error (Å)')
    
    ax_1.bar(methods, MAPEs, color=color)
    ax_1.set_xlabel('(b)')
    ax_1.set_ylabel('Mean Absolute Percentage Error (%)')
    
    plt.show()
    
    plt.figure()
    
    plt.scatter(ds, 100*(r_1s - r_0s)/r_0s, s=0.5, label='Deep Learning')
    plt.scatter(ds, 100*(r_2s - r_0s)/r_0s, s=0.5, label='Guinier Fit')
    plt.xscale('log')
    plt.xlabel('PDI')
    plt.ylabel('Percentage Error (%)')
    plt.legend()

    plt.show()
    
    num = 64
    
    PDIs = np.linspace(np.log10(1e-3), np.log10(5e-1), num)
    
    XD = np.power(10, PDIs + (np.log10(5e-1) - np.log10(1e-3))/128)[:-1]
    PDIs = np.power(10, PDIs)
    eds = np.zeros(shape=(num - 1, ))
    
    for i in range(num - 1):
        truths = np.logical_and(ds >= PDIs[i], ds < PDIs[i + 1])
        eds[i] += np.average((r_1s[truths] - r_0s[truths])/r_0s[truths])
        
    a, b = linear_least_square(X=XD, Y=eds)
    
    temp_0 = a*XD + b

    plt.figure()
    
    plt.scatter(XD, eds, label='Deep Learning')
    plt.plot(XD, temp_0, label='Approximation')
    plt.xlabel('PDI')
    plt.ylabel('Percentage Error (%)')
    plt.legend()
    
    plt.show()
    
    corr_0 = a*Ds + b + 1
    
    print(a, b)
    
    RMSE_DL, STD_RD = root_mean_squared_error(X=r_0s, Y=r_1s)
    RMSE_CR, STD_RC = root_mean_squared_error(X=r_0s, Y=r_1s/corr_0)
    RMSE_GN, STD_RG = root_mean_squared_error(X=r_0s, Y=r_2s)
    
    MAPE_DL, STD_MD = mean_absolute_percentage_error(X=r_0s, Y=r_1s)
    MAPE_CR, STD_MC = mean_absolute_percentage_error(X=r_0s, Y=r_1s/corr_0)
    MAPE_GN, STD_MG = mean_absolute_percentage_error(X=r_0s, Y=r_2s)
    
    print(RMSE_DL, RMSE_CR, RMSE_GN)
    print(MAPE_DL, MAPE_CR, MAPE_GN)
    
    RMSEs = [RMSE_DL, RMSE_CR, RMSE_GN]
    STD_R = [STD_RD, STD_RC, STD_RG]
    MAPEs = [MAPE_DL, MAPE_CR, MAPE_GN]
    STD_M = [STD_MD, STD_MC, STD_MG]

    methods = ['Deep\nLearning', 'Corrected\nDeep\nLearning', 'Guinier\nFit']
    color = ['blue', 'green', 'red']
    
    fig, axes = plt.subplots(1, 2, figsize=(7, 4))
    ax_0, ax_1 = axes[0], axes[1]
    
    ax_0.bar(methods, RMSEs, color=color)
    ax_0.set_xlabel('(a)')
    ax_0.set_ylabel('Root-Mean-Squared Error (Å)')
    
    ax_1.bar(methods, MAPEs, color=color)
    ax_1.set_xlabel('(b)')
    ax_1.set_ylabel('Mean Absolute Percentage Error (%)')

    plt.show()

    return None


def guinier_test(*args, **kwargs) -> None:
    
    shape = 'sphere'
    I, y = get_data(shape=shape)
    
    X = I
    
    X[X <= 0] = np.min(X[X >= 0])
    X = 1 + np.log10(X)/2
    X = np.tanh(X)
    X = X[np.newaxis, :, np.newaxis]
    
    q_log_arr = np.arange(-2, 0, np.true_divide(1, 128))
    q = np.power(10, q_log_arr - 2*np.log10(2))
    
    index = np.random.randint(0, y.shape[0])
    I_true = I[index, :]
    
    R_g, I_0 = guinier_fit(q=q, I=I_true)
    
    I_pred = I_0*np.exp(-np.square(q)*R_g**2/3)
    I_pred = I_pred/np.max(I_pred)
    
    r = y[index, 0]
    e = y[index, 1]
    
    a = b = r
    c = e*r
    r_g = sqrt((a**2 + b**2 + c**2)/5)
    
    print(f'True: {r_g:.3f}')
    print(f'Prediction: {R_g:.3f}')
    
    plt.figure()
    plt.plot(q, I_true)
    plt.plot(q, I_pred)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim((1e-5, 2))
    plt.show()
    
    q, I = cutoff(q, I[index, :])
    
    plt.figure()
    plt.plot(q, I)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    
    return None


def cutoff_test(*args, **kwargs) -> None:
    
    shape = 'sphere'
    I, y = get_data(shape=shape)
    
    X = I
    X[X <= 0] = np.min(X[X >= 0])
    X = 1 + np.log10(X)/2
    X = np.tanh(X)
    X = X[np.newaxis, :, np.newaxis]
    
    q_log_arr = np.arange(-2, 0, np.true_divide(1, 128))
    q_arr = np.power(10, q_log_arr - 2*np.log10(2))
    
    index = np.random.randint(0, y.shape[0])
    q, I = cutoff(q_arr, X[index, :])
    
    plt.figure()
    plt.plot(q, I)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    
    return None


def linear_test(*args, **kwargs) -> None:
    
    X = np.linspace(0, 1, 256)
    Y = 2*X + 1
    Y = np.random.normal(Y, np.square(Y)/10)
    
    m, b = linear_least_square(X=X, Y=Y)
    
    print(m, b)
    
    return None


def main(*args, **kwargs) -> int:
    
    sphere_correction()
    cylinder_correction()
    
    return 0


if __name__ == '__main__':
    main()
