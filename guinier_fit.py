import os # Module for file path operations
import numpy as np # Library for numerical computations
import matplotlib.pyplot as plt # Plotting library
import tensorflow as tf # TensorFlow for machine learning models
from math import sqrt, exp # Mathematical functions

# Function to compute Root Mean Squared Error and its standard deviation
def root_mean_squared_error(X: np.ndarray, Y: np.ndarray) -> float:
    return np.sqrt(np.average(np.square(X - Y))), np.sqrt(np.std(np.square(X - Y), ddof=1))

# Function to compute Mean Absolute Percentage Error (MAPE) and its standard deviation
def mean_absolute_percentage_error(X: np.ndarray, Y: np.ndarray) -> float:
    return 100*np.average(np.absolute((X - Y)/X)), 100*np.std(np.absolute((X - Y)/X), ddof=1)

# Function to load and preprocess data based on the shape type
def get_data(shape: str) -> tuple[np.ndarray]:
    
    cwd = os.getcwd()
    file_dir = os.path.join(cwd, 'Data')

    # Choose file name based on shape
    if shape == 'sphere':
        name = '2024_09_12_disperse_spheroid_core_shell_yj5782_0.npy'
    elif shape == 'cylinder':
        name = '2024_09_11_disperse_cylinder_core_shell_yj5782_0.npy'
    else:
        pass
    
    file_name = os.path.join(file_dir, name)
    data = np.load(file_name)   # Load data from the file
    
    d = []  # List to store indices of rows with NaN values

    # Identify rows containing NaN values
    for i in range(data.shape[0]):
        if True in np.isnan(data[i, :]):
            d.append(i)

    np.delete(data, d, 0)   # Remove rows with NaN values
    
    X = data[:, 4:]     # Feature data
    y = data[:, 0:4]    # Target data
        
    return X, y

# Linear least squares fitting
def linear_least_square(X: np.ndarray, Y: np.ndarray) -> tuple[float]:
    
    a_11 = np.sum(np.square(X))
    a_12 = a_21 = np.sum(X)
    a_22 = X.size
    
    b_1 = np.sum(X*Y)
    b_2 = np.sum(Y)
    
    A = np.array(((a_11, a_12), 
                  (a_21, a_22)))
    b = np.array((b_1, b_2)).reshape(-1, 1)
    
    v = np.matmul(np.linalg.inv(A), b)  # Solve the linear system
    
    return v[0, 0], v[1, 0]

# Computes the quadratic least squares fit for the given data
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
    
    v = np.matmul(np.linalg.inv(A), b)  # Solves the quadratic system
    
    return v[0, 0], v[1, 0], v[2, 0]

# Computes the Lennard-Jones potential fitting
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

# Processes data to compute radius of gyration (Rg) and intensity at zero angle (I0)
def data_processing(q: np.ndarray, I: np.ndarray) -> tuple[float]:
    
    X = np.square(q)
    Y = np.log(I)
    
    m, b = linear_least_square(X=X, Y=Y)
    R_g = sqrt(-3*m)
    I_0 = exp(b)
    
    return R_g, I_0


# Applies machine learning model to filter data based on a cutoff
def cutoff(q: np.ndarray, I: np.ndarray) -> tuple[np.ndarray]:
    
    cwd = os.getcwd()
    model_dir = os.path.join(cwd, 'Models')
    model_name = '2024_11_18_SCNN_qr_0.keras'
    
    model_path = os.path.join(model_dir, model_name)
    model = tf.keras.models.load_model(model_path, compile=False)
    
    X = I
    X[X <= 0] = np.min(X[X >= 0])   # Replace non-positive values
    X = 1 + np.log10(X)/2
    X = np.tanh(X)
    X = X[np.newaxis, :, np.newaxis]
    
    r = 256*model.predict(X)[0]
    qr = q*r
        
    q_new = q[qr <= 3]
    I_new = I[qr <= 3]

    return q_new, I_new

# Performs Guinier fitting to estimate radius of gyration and zero-angle intensity
def guinier_fit(q: np.ndarray, I: np.ndarray) -> tuple[float]:
    
    q, I = cutoff(q=q, I=I)
    R_g, I_0 = data_processing(q=q, I=I)
    
    return R_g, I_0

# Compares the performance of deep learning models with Guinier fitting for
# predicting structural parameters of a sphere from small-angle X-ray scattering (SAXS) data
def compare(*args, **kwargs) -> None:
    # Generate logarithmic q values ranging from -2 to 0 with 128 steps and scale to match sphere
    # model parameters
    q_log_arr = np.arange(-2, 0, np.true_divide(1, 128))
    q = np.power(10, q_log_arr - 2*np.log10(2))

    # Set the data shape to 'sphere' for model-specific data retrieval
    shape = 'sphere'

    # Retrieve experimental data for the given shape
    I, y = get_data(shape=shape)

    # Normalize intensity data (I) using logarithmic scaling and apply a tanh transformation
    X = I
    X[X <= 0] = np.min(X[X >= 0])   # Avoid log of zero or negative values
    X = 1 + np.log10(X)/2
    X = np.tanh(X)
    X = X[:, :, np.newaxis] # Reshape for model compatibility

    # Set up model paths for different properties: Radius, Aspect Ratio, and Polydispersity Index
    cwd = os.getcwd()
    model_dir = os.path.join(cwd, 'Models')
    file_name_sr = '2024_11_18_sphere_CPNN_Radius_0.keras'
    file_name_se = '2024_11_18_sphere_CPNN_AspectRatio_0.keras'
    file_name_sd = '2024_12_10_sphere_CPNN_PDI_0.keras'
    file_path_sr = os.path.join(model_dir, file_name_sr)
    file_path_se = os.path.join(model_dir, file_name_se)
    file_path_sd = os.path.join(model_dir, file_name_sd)

    # Load pre-trained deep learning models for radius, aspect ratio, and PDI predictions
    model_sr = tf.keras.models.load_model(file_path_sr, compile=False)
    model_se = tf.keras.models.load_model(file_path_se, compile=False)
    model_sd = tf.keras.models.load_model(file_path_sd, compile=False)

    # Initialize arrays to store computed and predicted radii
    r_0s = np.zeros(shape=(X.shape[0], ))   # Computed radii using theoretical model
    r_1s = np.zeros(shape=(X.shape[0], ))   # Predicted radii from deep learning model
    r_2s = np.zeros(shape=(X.shape[0], ))   # Radii from Guinier fit

    # Extract ground-truth values for radius, aspect ratio, and PDI
    rs = y[:, 0]
    es = y[:, 1]
    ds = y[:, 2]

    # Compute initial radii based on geometric properties
    As = Bs = rs
    Cs = es*rs
    r_0s += np.sqrt((np.square(As) + np.square(Bs) + np.square(Cs))/5)

    # Perform Guinier fitting for each radius
    for i in range(rs.size):
        r_g = r_0s[i]
        qr = q*r_g
        q_temp = q[qr <= 1.3]
        I_temp = I[i, :]
        I_temp = I_temp[qr <= 1.3]
        r_g, _ = data_processing(q=q_temp, I=I_temp)
        r_2s[i] += r_g

    # Predict values using deep learning models
    r_pred = model_sr.predict(X)
    e_pred = model_se.predict(X)
    d_pred = model_sd.predict(X)

    # Convert predictions into physical quantities
    Rs = 256*r_pred[:, 0]
    Es = 2*e_pred[:, 0]
    Ds = np.power(10, -4*d_pred[:, 0])/2

    # Adjust radius using scaling factor based on PDI
    gamma = 1 + 0.08*Ds/0.5
    As = Bs = Rs
    Cs = Es*Rs
    r_1s += np.sqrt((np.square(As) + np.square(Bs) + np.square(Cs))/5)
    r_1s = r_1s/gamma

    # Calculate error metrics: Root Mean Squared Error (RMSE) and
    RMSE_DL = root_mean_squared_error(X=r_0s, Y=r_1s)
    RMSE_GN = root_mean_squared_error(X=r_0s, Y=r_2s)
    MAPE_DL = mean_absolute_percentage_error(X=r_0s, Y=r_1s)
    MAPE_GN = mean_absolute_percentage_error(X=r_0s, Y=r_2s)

    # Display error results
    print(RMSE_DL, MAPE_DL)
    print(RMSE_GN, MAPE_GN)

    # Prepare error metrics for visualization
    RMSEs = [RMSE_DL, RMSE_GN]
    MAPEs = [MAPE_DL, MAPE_GN]
    methods = ['Deep Learning', 'Guinier Fit']
    color = ['blue', 'red']

    # Create bar plots for error comparison
    fig, axes = plt.subplots(1, 2, figsize=(6, 4))
    ax_0, ax_1 = axes[0], axes[1]
    ax_0.bar(methods, RMSEs, color=color)
    ax_0.set_xlabel('(a)')
    ax_0.set_ylabel('Root-Mean-Squared Error (Å)')
    ax_1.bar(methods, MAPEs, color=color)
    ax_1.set_xlabel('(b)')
    ax_1.set_ylabel('Mean Absolute Percentage Error (%)')

    # Save plot of error comparison
    plot_dir = os.path.join(cwd, 'assets')
    name = os.path.join(plot_dir, 'guinier.png')
    fig.savefig(name, bbox_inches='tight')
    plt.show()

    # Create scatter plot showing percentage errors for different PDIs
    plt.figure()
    plt.scatter(ds, 100*(r_1s - r_0s)/r_0s, s=0.5, label='Deep Learning')
    plt.scatter(ds, 100*(r_2s - r_0s)/r_0s, s=0.5, label='Guinier Fit')
    plt.xscale('log')
    plt.xlabel('PDI')
    plt.ylabel('Percentage Error (%)')
    plt.legend()
    plt.show()

    # Analyze percentage errors for different PDI bins and fit curves to model error trends
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

    # Perform least-square fits for error trends
    a, b, c = quadratic_least_square(X=XD, Y=eds)
    d, e = linear_least_square(X=XD, Y=egs)
    temp_0 = a*np.square(XD) + b*XD + c
    # temp_1 = d*XD + e     # Optional linear fit for Guinier errors

    # Create a new figure for plotting
    plt.figure()

    # Plot the percentage error as a function of PDI using different methods
    plt.plot(XD, eds, label='Deep Learning')    # Plot Deep Learning error
    plt.plot(XD, temp_0, label='Mock 1')    # Plot quadratic fit correction (Mock 1)
    # plt.plot(XD, egs, label='Guinier Fit')    # Uncomment to plot Guinier Fit error
    # plt.plot(XD, temp_1, label='Mock 2')      # Uncomment to plot linear fit correction (Mock 2)
    # plt.xscale('log')     # Uncomment to use a logarithmic x-scale for better visualization
    plt.xlabel('PDI')   # Label for the x-axis
    plt.ylabel('Percentage Error (%)')  # Label for the y-axis
    plt.legend()    # Display the legend for the plot

    # Show the plot
    plt.show()

    # Apply quadratic correction formula for percentage error
    corr_0 = a*np.square(Ds) + b*Ds + c + 1 # Corrected deep learning model error
    # corr_1 = d*Ds + e + 1     # Uncomment for alternative linear correction

    # Create another plot for percentage error before and after correction
    plt.figure()

    # Scatter plot to show percentage error distribution for different methods
    plt.scatter(ds, 100*(r_1s - r_0s)/r_0s, s=0.5, label='Deep Learning')
    plt.scatter(ds, 100*(r_1s/corr_0 - r_0s)/r_0s, s=0.5, label='Corrected Deep Learning')
    plt.scatter(ds, 100*(r_2s - r_0s)/r_0s, s=0.5, label='Guinier Fit')
    plt.xscale('log')
    plt.xlabel('PDI')
    plt.ylabel('Percentage Error (%)')
    plt.legend()

    plt.show()

    # Calculate error metrics (RMSE and MAPE) for all methods
    RMSE_DL = root_mean_squared_error(X=r_0s, Y=r_1s)
    RMSE_CR = root_mean_squared_error(X=r_0s, Y=r_1s/corr_0)
    RMSE_GN = root_mean_squared_error(X=r_0s, Y=r_2s)
    
    MAPE_DL = mean_absolute_percentage_error(X=r_0s, Y=r_1s)
    MAPE_CR = mean_absolute_percentage_error(X=r_0s, Y=r_1s/corr_0)
    MAPE_GN = mean_absolute_percentage_error(X=r_0s, Y=r_2s)

    # Print the calculated RMSE and MAPE values
    print(RMSE_DL, RMSE_CR, RMSE_GN)
    print(MAPE_DL, MAPE_CR, MAPE_GN)

    # Store RMSE and MAPE values for bar plot
    RMSEs = [RMSE_DL, RMSE_CR, RMSE_GN]
    MAPEs = [MAPE_DL, MAPE_CR, MAPE_GN]

    # Define labels and colros for bar plots
    methods = ['Deep\nLearning', 'Corrected\nDeep\nLearning', 'Guinier\nFit']
    color = ['blue', 'green', 'red']

    # Create subplots for RMSE and MAPE comparsion
    fig, axes = plt.subplots(1, 2, figsize=(6, 4))
    ax_0, ax_1 = axes[0], axes[1]

    # Plot bar graph for RMSE values
    ax_0.bar(methods, RMSEs, color=color)
    ax_0.set_xlabel('(a)')
    ax_0.set_ylabel('Root-Mean-Squared Error (Å)')

    # Plot bar graph for MAPE values
    ax_1.bar(methods, MAPEs, color=color)
    ax_1.set_xlabel('(b)')
    ax_1.set_ylabel('Mean Absolute Percentage Error (%)')
    
    plt.show()

    return None


def sphere_correction(*args, **kwargs) -> None:
    # Generate an array of q values, logarithmically spaced between -2 and 0 with 128 points
    q_log_arr = np.arange(-2, 0, np.true_divide(1, 128))
    # Calculate q values from the logarithmic array
    q = np.power(10, q_log_arr - 2*np.log10(2))

    # Define the shape type as 'sphere'
    shape = 'sphere'

    # Get the data for the specified shape
    I, y = get_data(shape=shape)

    # Initialize X as I (Intensity data)
    X = I
    X[X <= 0] = np.min(X[X >= 0])
    # Apply a transformation to X
    X = 1 + np.log10(X)/2
    X = np.tanh(X)
    # Add a new axis to X for compatibility with the model input
    X = X[:, :, np.newaxis]

    # Get the current working directory and set the model directory path
    cwd = os.getcwd()
    model_dir = os.path.join(cwd, 'Models')

    # Define file names for the pre-trained models
    file_name_sr = '2024_11_18_sphere_CPNN_Radius_0.keras'
    file_name_se = '2024_11_18_sphere_CPNN_AspectRatio_0.keras'
    file_name_sd = '2024_12_10_sphere_CPNN_PDI_0.keras'

    # Construct the full file paths for the models
    file_path_sr = os.path.join(model_dir, file_name_sr)
    file_path_se = os.path.join(model_dir, file_name_se)
    file_path_sd = os.path.join(model_dir, file_name_sd)

    # Load the pre-trained models without compiling them
    model_sr = tf.keras.models.load_model(file_path_sr, compile=False)
    model_se = tf.keras.models.load_model(file_path_se, compile=False)
    model_sd = tf.keras.models.load_model(file_path_sd, compile=False)

    # Initialize arrays to store radius predictions from different methods
    r_0s = np.zeros(shape=(X.shape[0], ))
    r_1s = np.zeros(shape=(X.shape[0], ))
    r_2s = np.zeros(shape=(X.shape[0], ))

    # Extract relevant data from the 'y' array
    rs = y[:, 0]
    es = y[:, 1]
    ds = y[:, 2]

    # Calculate the Guinier approximation for r_0
    As = Bs = rs
    Cs = es*rs
    r_0s += np.sqrt((np.square(As) + np.square(Bs) + np.square(Cs))/5)

    # Iterate over the data to apply Guinier fitting for each data point
    for i in range(rs.size):
        r_g = r_0s[i]
        qr = q*r_g
        # Select q and I values wer qr <= 1.3
        q_temp = q[qr <= 1.3]
        I_temp = I[i, :]
        I_temp = I_temp[qr <= 1.3]

        # Apply data processing and update r_2s with the new radius
        r_g, _ = data_processing(q=q_temp, I=I_temp)
        r_2s[i] += r_g

    # Use the pre-trained models to predict the raidus, aspect ratio, and PDI
    r_pred = model_sr.predict(X)
    e_pred = model_se.predict(X)
    d_pred = model_sd.predict(X)

    # Calculate the predicted radii, aspect ratios, and PDIs
    Rs = 256*r_pred[:, 0]
    Es = 2*e_pred[:, 0]
    Ds = np.power(10, -4*d_pred[:, 0])/2

    # Calculate r_1s using the predicted values
    As = Bs = Rs
    Cs = Es*Rs
    r_1s += np.sqrt((np.square(As) + np.square(Bs) + np.square(Cs))/5)

    # Compute RMSE and MAPE for both methods
    RMSE_DL, _ = root_mean_squared_error(X=r_0s, Y=r_1s)
    RMSE_GN, _ = root_mean_squared_error(X=r_0s, Y=r_2s)
    
    MAPE_DL, _ = mean_absolute_percentage_error(X=r_0s, Y=r_1s)
    MAPE_GN, _ = mean_absolute_percentage_error(X=r_0s, Y=r_2s)

    # Print the RMSE and MAPE for both methods
    print(RMSE_DL, MAPE_DL)
    print(RMSE_GN, MAPE_GN)

    # Store the RMSE and MAPE values in lists for plotting
    RMSEs = [RMSE_DL, RMSE_GN]
    MAPEs = [MAPE_DL, MAPE_GN]

    # Define the methods and colors for the plots
    methods = ['Deep Learning', 'Guinier Fit']
    color = ['blue', 'red']

    # Create subplots for RMSE and MAPE comparison
    fig, axes = plt.subplots(1, 2, figsize=(6, 4))
    ax_0, ax_1 = axes[0], axes[1]

    # Plot RMSE values
    ax_0.bar(methods, RMSEs, color=color)
    ax_0.set_xlabel('(a)')
    ax_0.set_ylabel('Root-Mean-Squared Error (Å)')

    # Plot MAPE values
    ax_1.bar(methods, MAPEs, color=color)
    ax_1.set_xlabel('(b)')
    ax_1.set_ylabel('Mean Absolute Percentage Error (%)')

    # Save the plot as an image file
    plot_dir = os.path.join(cwd, 'assets')
    name = os.path.join(plot_dir, 'guinier.png')
    fig.savefig(name, bbox_inches='tight')

    # Display the plot
    plt.show()

    # Create a scatter plot for percentage error vs PDI
    plt.figure()
    plt.scatter(ds, 100*(r_1s - r_0s)/r_0s, s=0.5, label='Deep Learning')
    plt.scatter(ds, 100*(r_2s - r_0s)/r_0s, s=0.5, label='Guinier Fit')
    plt.xscale('log')
    plt.xlabel('PDI')
    plt.ylabel('Percentage Error (%)')
    plt.legend()
    plt.show()

    # Define the number of points for PDI range and calculate corresponding values
    num = 64
    PDIs = np.linspace(np.log10(1e-3), np.log10(5e-1), num)
    XD = np.power(10, PDIs + (np.log10(5e-1) - np.log10(1e-3))/128)[:-1]
    PDIs = np.power(10, PDIs)
    eds = np.zeros(shape=(num - 1, ))
    egs = np.zeros(shape=(num - 1, ))

    # Compute the average percentage error for each PDI range
    for i in range(num - 1):
        truths = np.logical_and(ds >= PDIs[i], ds < PDIs[i + 1])
        eds[i] += np.average((r_1s[truths] - r_0s[truths])/r_0s[truths])
        egs[i] += np.average((r_2s[truths] - r_0s[truths])/r_0s[truths])
        
    # Perform quadratic least squares fitting on the error data
    a, b, c = quadratic_least_square(X=XD, Y=eds)
    print(a, b, c)

    # Calculate the fitted error curve
    temp_0 = a*np.square(XD) + b*XD + c

    # Create a plot for the error vs PDI
    plt.figure()
    plt.scatter(XD, 100*eds, label='Deep Learning Error')
    plt.plot(XD, 100*temp_0, label='Approximation', color='k', linestyle='dotted')
    plt.xlabel('PDI')
    plt.ylabel('Percentage Error (%)')
    plt.legend()
    plt.show()

    # Calculate the corrected radius using a quadratic equation and a constant offset of 1
    corr_0 = a*np.square(Ds) + b*Ds + c + 1

    # Compute the RMSE and STD for Deep Learning predictions
    RMSE_DL, STD_RD = root_mean_squared_error(X=r_0s, Y=r_1s)

    # Compute the RMSE and STD for the corrected Deep Learning predictions
    RMSE_CR, STD_RC = root_mean_squared_error(X=r_0s, Y=r_1s/corr_0)

    # Compute the RMSE and STD for the Guinier Fit predictions
    RMSE_GN, STD_RG = root_mean_squared_error(X=r_0s, Y=r_2s)

    # Compute the MAPE and STD for Deep Learning predictions
    MAPE_DL, STD_MD = mean_absolute_percentage_error(X=r_0s, Y=r_1s)

    # Compute the MAPE and STD for the corrected Deep Learning predictions
    MAPE_CR, STD_MC = mean_absolute_percentage_error(X=r_0s, Y=r_1s/corr_0)

    # Compute the MAPE and STD for Guinier Fit prediction
    MAPE_GN, STD_MG = mean_absolute_percentage_error(X=r_0s, Y=r_2s)

    # Print RMSE and MAPE values for the three methods
    print(RMSE_DL, RMSE_CR, RMSE_GN)
    print(MAPE_DL, MAPE_CR, MAPE_GN)

    # Store RMSE, STD, and MAPE values for all three methods in lists
    RMSEs = [RMSE_DL, RMSE_CR, RMSE_GN]
    STD_R = [STD_RD, STD_RC, STD_RG]
    MAPEs = [MAPE_DL, MAPE_CR, MAPE_GN]
    STD_M = [STD_MD, STD_MC, STD_MG]

    # Define labels for the methods and corresponding colors for plotting
    methods = ['Deep\nLearning', 'Corrected\nDeep\nLearning', 'Guinier\nFit']
    color = ['blue', 'green', 'red']

    # Create subplots for visual comparison of RMSE and MAPE for the three methods
    fig, axes = plt.subplots(1, 2, figsize=(6, 4))
    ax_0, ax_1 = axes[0], axes[1]

    # Plot RMSE values for the three methods
    ax_0.bar(methods, RMSEs, color=color)
    ax_0.set_xlabel('(a)')
    ax_0.set_ylabel('Root-Mean-Squared Error (Å)')

    # Plot MAPE values for the three methods
    ax_1.bar(methods, MAPEs, color=color)
    ax_1.set_xlabel('(b)')
    ax_1.set_ylabel('Mean Absolute Percentage Error (%)')

    # Display the plots
    plt.show()

    return None

# Function to perform calculations and visualizations related to the 'sphere' shape
# This function loads pre-trained models for predicting the raidus, aspect ratio, and PDI of a
# sphere. It computes percentage errors for these predictions, fits polynomial models to the
# errors, and plots the results. It also computes and prints the RMSEand MAPE for original and
# corrected predictions.
def sphere_check(*args, **kwargs) -> None:
    # Define the shape to be 'sphere'
    shape = 'sphere'

    # Retrieve the data for the specified shape
    I, y = get_data(shape=shape)

    # Process the input data
    X = I
    X[X <= 0] = np.min(X[X >= 0])
    X = 1 + np.log10(X)/2
    X = np.tanh(X)
    X = X[:, :, np.newaxis]

    # Define paths for model directories and files
    cwd = os.getcwd()
    model_dir = os.path.join(cwd, 'Models')

    # Define model filenames for different parameters
    file_name_sr = '2024_11_18_sphere_CPNN_Radius_0.keras'
    file_name_se = '2024_11_18_sphere_CPNN_AspectRatio_0.keras'
    file_name_sd = '2024_12_10_sphere_CPNN_PDI_0.keras'

    # Create full file paths
    file_path_sr = os.path.join(model_dir, file_name_sr)
    file_path_se = os.path.join(model_dir, file_name_se)
    file_path_sd = os.path.join(model_dir, file_name_sd)

    # Load pre-trained models for radius, AspectRatio, and PDI
    model_sr = tf.keras.models.load_model(file_path_sr, compile=False)
    model_se = tf.keras.models.load_model(file_path_se, compile=False)
    model_sd = tf.keras.models.load_model(file_path_sd, compile=False)

    # Extract target data for Radius, AspectRatio, and PDI
    rs = y[:, 0]
    es = y[:, 1]
    ds = y[:, 2]

    # Make predictions using the pre-trained models
    r_pred = model_sr.predict(X)
    e_pred = model_se.predict(X)
    d_pred = model_sd.predict(X)

    # Process predictions for Radius, AspectRatio, and PDI
    Rs = 256*r_pred[:, 0]
    Es = 2*e_pred[:, 0]
    Ds = np.power(10, -4*d_pred[:, 0])/2
    
    num = 64    # Number of data points for PDI

    # Create a linear space for PDI values
    PDIs = np.linspace(np.log10(1e-3), np.log10(5e-1), num)

    # Compute the PDI and corresponding error arrays
    XD = np.power(10, PDIs + (np.log10(5e-1) - np.log10(1e-3))/128)[:-1]
    PDIs = np.power(10, PDIs)
    e_rs = np.zeros(shape=(num - 1, ))
    e_es = np.zeros(shape=(num - 1, ))
    e_ds = np.zeros(shape=(num - 1, ))

    # Calculate percentage errors for each range of PDI
    for i in range(num - 1):
        truths = np.logical_and(ds >= PDIs[i], ds < PDIs[i + 1])
        e_rs[i] += np.average((Rs[truths] - rs[truths])/rs[truths])
        e_es[i] += np.average((Es[truths] - es[truths])/es[truths])
        e_ds[i] += np.average((Ds[truths] - ds[truths])/ds[truths])

    # Fit quadratic and cubic polynomial approximations for the errors
    a_0, b_0, c_0 = np.polyfit(x=XD, y=e_rs, deg=2)
    a_1, b_1, c_1 = np.polyfit(x=XD, y=e_es, deg=2)
    a_2, b_2, c_2, d_2 = np.polyfit(x=np.log10(XD), y=e_ds, deg=3)

    # Calculate the polynomial approximation values for the errors
    temp_0 = a_0*np.square(XD) + b_0*XD + c_0
    temp_1 = a_1*np.square(XD) + b_1*XD + c_1
    temp_2 = a_2*np.power(np.log10(XD), 3) + b_2*np.square(np.log10(XD)) + c_2*np.log10(XD) + d_2

    # Plot the percentage errors for Radius
    plt.figure()
    # plt.scatter(ds, 100*(Rs - rs)/rs, s=0.5, label='Radius')
    plt.scatter(XD, e_rs, label='Average', s=1.0)
    plt.plot(XD, temp_0, label='Approximation', linestyle='dotted', color='k')
    # plt.xscale('log')
    plt.xlabel('PDI')
    plt.ylabel('Percentage Error (%)')
    plt.legend()
    plt.show()

    # Plot the percentage errors for AspectRatio
    plt.figure()
    # plt.scatter(ds, 100*(Es - es)/es, s=0.5, label='Aspect Ratio')
    plt.scatter(XD, e_es, label='Average', s=1.0)
    plt.plot(XD, temp_1, label='Approximation', linestyle='dotted', color='k')
    # plt.xscale('log')
    plt.xlabel('PDI')
    plt.ylabel('Percentage Error (%)')
    plt.legend()
    plt.show()

    # Plot the percentage errors for PDI
    plt.figure()
    # plt.scatter(ds, 100*(Ds - ds)/ds, s=0.5, label='PDI')
    plt.scatter(XD, e_ds, label='Average', s=1.0)
    plt.plot(XD, temp_2, label='Approximation', linestyle='dotted', color='k')
    # plt.xscale('log')
    plt.xlabel('PDI')
    plt.ylabel('Percentage Error (%)')
    plt.legend()
    plt.show()

    # Apply corrections to the PDI values based on the fitted models
    corr_2 = a_2*np.power(np.log10(Ds), 3) + b_2*np.square(np.log10(Ds)) + c_2*np.log10(Ds) + d_2 + 1
    Ds_corr = Ds/corr_2     # Corrected PDI
        
    corr_0 = a_0*np.square(Ds_corr) + b_0*Ds_corr + c_0 + 1     # Corrected Radius
    corr_1 = a_1*np.square(Ds_corr) + b_1*Ds_corr + c_1 + 1     # Corrected AspectRatio

    Rs_corr = Rs/corr_0
    Es_corr = Es/corr_1

    # Compute RMSE for original vs. corrected values
    RMSE_RR, _ = root_mean_squared_error(X=rs, Y=Rs)
    RMSE_RC, _ = root_mean_squared_error(X=rs, Y=Rs_corr)
    RMSE_ER, _ = root_mean_squared_error(X=es, Y=Es)
    RMSE_EC, _ = root_mean_squared_error(X=es, Y=Es_corr)
    RMSE_DR, _ = root_mean_squared_error(X=ds, Y=Ds)
    RMSE_DC, _ = root_mean_squared_error(X=ds, Y=Ds_corr)

    # Compute MAPE for original vs. corrected values
    MAPE_RR, _ = mean_absolute_percentage_error(X=rs, Y=Rs)
    MAPE_RC, _ = mean_absolute_percentage_error(X=rs, Y=Rs_corr)
    MAPE_ER, _ = mean_absolute_percentage_error(X=es, Y=Es)
    MAPE_EC, _ = mean_absolute_percentage_error(X=es, Y=Es_corr)
    MAPE_DR, _ = mean_absolute_percentage_error(X=ds, Y=Ds)
    MAPE_DC, _ = mean_absolute_percentage_error(X=ds, Y=Ds_corr)

    # Print RMSE and MAPE values for original and corrected methods
    print(RMSE_RR, RMSE_RC)
    print(RMSE_ER, RMSE_EC)
    print(RMSE_DR, RMSE_DC)

    print(MAPE_RR, MAPE_RC)
    print(MAPE_ER, MAPE_EC)
    print(MAPE_DR, MAPE_DC)

    # Store RMSE and MAPE values for all methods in lists
    RMSEs = [RMSE_RR, RMSE_RC, RMSE_ER, RMSE_EC, RMSE_DR, RMSE_DC]
    MAPEs = [MAPE_RR, MAPE_RC, MAPE_ER, MAPE_EC, MAPE_DR, MAPE_DC]

    # Define labels for the methods
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

# Function to perform cylinder shape correction using pre-trained models and the Guinier Fit.
# This function loads pre-trained models for predicting the radius, aspect ratio, and PDI of a
# cylinder. It computes the radius using deep learning and Guinier Fit methods, compares the
# results, and visualizes the percentage errors. It also computes and prints the RMSE and MAPE
# for both methods.
def cylinder_correction(*args, **kwargs) -> None:
    # Generate a range of q values for Guinier Fit
    q_log_arr = np.arange(-2, 0, np.true_divide(1, 128))
    q = np.power(10, q_log_arr - 2*np.log10(2))

    # Set the shape to 'cylinder' and get associated data
    shape = 'cylinder'
    I, y = get_data(shape=shape)

    # Process the input data
    X = I
    X[X <= 0] = np.min(X[X >= 0])   # Replace non-positive values with minimum positive value
    X = 1 + np.log10(X)/2   # Apply a transformation to X
    X = np.tanh(X)      # Apply a tanh transformation to X
    X = X[:, :, np.newaxis]     # Reshape X to 3D array

    # Get the current working directory and define model file paths
    cwd = os.getcwd()
    model_dir = os.path.join(cwd, 'Models')
    file_name_sr = '2024_11_18_cylinder_CPNN_Radius_0.keras'
    file_name_se = '2024_11_18_cylinder_CPNN_AspectRatio_0.keras'
    file_name_sd = '2024_12_10_cylinder_CPNN_PDI_0.keras'

    file_path_sr = os.path.join(model_dir, file_name_sr)
    file_path_se = os.path.join(model_dir, file_name_se)
    file_path_sd = os.path.join(model_dir, file_name_sd)

    # Load pre-trained models for cylinder shape
    model_sr = tf.keras.models.load_model(file_path_sr, compile=False)
    model_se = tf.keras.models.load_model(file_path_se, compile=False)
    model_sd = tf.keras.models.load_model(file_path_sd, compile=False)

    # Extract actual radius, aspect ratio, and PDI values from the dataset
    rs = y[:, 0]
    ls = y[:, 1]
    ds = y[:, 2]

    # Make predictions using the models
    r_pred = model_sr.predict(X)
    l_pred = model_se.predict(X)
    d_pred = model_sd.predict(X)

    # Calculate derived physical quantities from predictions
    Rs = 256*r_pred[:, 0]
    Ls = 16*l_pred[:, 0]*Rs
    Ds = np.power(10, -4*d_pred[:, 0])/2

    # Compite radii based on ground truth and model predictions
    r_0s = np.sqrt(np.square(rs)/2 + np.square(ls)/12)
    r_1s = np.sqrt(np.square(Rs)/2 + np.square(Ls)/12)
    r_2s = np.zeros(shape=(X.shape[0], ))

    # Perform Guinier Fit for each sample
    for i in range(rs.size):
        r_g = r_0s[i]
        qr = q*r_g
        q_temp = q[qr <= 1.0]
        I_temp = I[i, :]
        I_temp = I_temp[qr <= 1.0]

        # If there are too few data points, use the first 8
        if q_temp.size < 8:
            q_temp = q[:8]
            I_temp = I[i, :]
            I_temp = I_temp[:8]

        # Perform data processing on the Guinier Fit data
        r_g, _ = data_processing(q=q_temp, I=I_temp)
        r_2s[i] += r_g

    # Compute RMSE and MAPE for both methods
    RMSE_DL, _ = root_mean_squared_error(X=r_0s, Y=r_1s)
    RMSE_GN, _ = root_mean_squared_error(X=r_0s, Y=r_2s)
    MAPE_DL, _ = mean_absolute_percentage_error(X=r_0s, Y=r_1s)
    MAPE_GN, _ = mean_absolute_percentage_error(X=r_0s, Y=r_2s)
    
    print(RMSE_DL, RMSE_GN)
    print(MAPE_DL, MAPE_GN)

    # Plot RMSE and MAPE comparison between the two methods
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

    # Plot percentage error comparison for the two methods
    plt.figure()
    plt.scatter(ds, 100*(r_1s - r_0s)/r_0s, s=0.5, label='Deep Learning')
    plt.scatter(ds, 100*(r_2s - r_0s)/r_0s, s=0.5, label='Guinier Fit')
    plt.xscale('log')
    plt.xlabel('PDI')
    plt.ylabel('Percentage Error (%)')
    plt.legend()
    plt.show()

    # Prepare for additional error analysis
    num = 64
    PDIs = np.linspace(np.log10(1e-3), np.log10(5e-1), num)
    XD = np.power(10, PDIs + (np.log10(5e-1) - np.log10(1e-3))/128)[:-1]
    PDIs = np.power(10, PDIs)
    eds = np.zeros(shape=(num - 1, ))

    # Calculate percentage error in radius for each PDI bin
    for i in range(num - 1):
        truths = np.logical_and(ds >= PDIs[i], ds < PDIs[i + 1])
        eds[i] += np.average((r_1s[truths] - r_0s[truths])/r_0s[truths])

    # Fit a linear model to the errors
    a, b = linear_least_square(X=XD, Y=eds)
    temp_0 = a*XD + b

    # Plot the approximation of errors
    plt.figure()
    plt.scatter(XD, eds, label='Deep Learning')
    plt.plot(XD, temp_0, label='Approximation')
    plt.xlabel('PDI')
    plt.ylabel('Percentage Error (%)')
    plt.legend()
    plt.show()

    # Perform correction based on the fitted model
    corr_0 = a*Ds + b + 1
    print(a, b)

    # Compute RMSE and MAPE for the corrected predictions
    RMSE_DL, STD_RD = root_mean_squared_error(X=r_0s, Y=r_1s)
    RMSE_CR, STD_RC = root_mean_squared_error(X=r_0s, Y=r_1s/corr_0)
    RMSE_GN, STD_RG = root_mean_squared_error(X=r_0s, Y=r_2s)
    
    MAPE_DL, STD_MD = mean_absolute_percentage_error(X=r_0s, Y=r_1s)
    MAPE_CR, STD_MC = mean_absolute_percentage_error(X=r_0s, Y=r_1s/corr_0)
    MAPE_GN, STD_MG = mean_absolute_percentage_error(X=r_0s, Y=r_2s)

    # Print the final RMSE and MAPE values for comparison
    print(RMSE_DL, RMSE_CR, RMSE_GN)
    print(MAPE_DL, MAPE_CR, MAPE_GN)

    # Plot the final RMSE and MAPE comparison for all methods
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


# Performs a Guinier fit test on randomly selected SAXS data for spheres. The function compares
# the predicted radius of gyration from the Guinier fit with the true radius of gyration and
# visualizes the results.
def guinier_test(*args, **kwargs) -> None:
    # Define the shape of the object being tested
    shape = 'sphere'

    # Retrieve SAXS data
    I, y = get_data(shape=shape)

    # Assign intensity data to X for processing
    X = I

    # Ensure no zero or negative values in X to prevent computational errors
    X[X <= 0] = np.min(X[X >= 0])

    # Apply logarithmic scaling and tanh normalization to the intensity data
    X = 1 + np.log10(X)/2
    X = np.tanh(X)

    # Add an extra axis to match the input shape for models or functions that require 3D arrays
    X = X[np.newaxis, :, np.newaxis]

    # Generate an array of logarithmic q values for the Guinier fit
    q_log_arr = np.arange(-2, 0, np.true_divide(1, 128))
    q = np.power(10, q_log_arr - 2*np.log10(2))

    # Select a random index from the dataset
    index = np.random.randint(0, y.shape[0])

    # Extract the intensity data for the selected index
    I_true = I[index, :]

    # Perform a Guinier fit on the selected data to estimate R_g and I_0
    R_g, I_0 = guinier_fit(q=q, I=I_true)

    # Compute the predicted intensity using the Guinier approximation
    I_pred = I_0*np.exp(-np.square(q)*R_g**2/3)

    # Normalize the predicted intensity for comparison
    I_pred = I_pred/np.max(I_pred)

    # Retrieve the true radius and elongation from the dataset
    r = y[index, 0]
    e = y[index, 1]

    # Compute the true radius of gyration using geometric properties
    a = b = r
    c = e*r
    r_g = sqrt((a**2 + b**2 + c**2)/5)

    # Print the true and predicted radius of gyration
    print(f'True: {r_g:.3f}')
    print(f'Prediction: {R_g:.3f}')

    # Plot the true and predicted intensity as a function of q
    plt.figure()
    plt.plot(q, I_true)
    plt.plot(q, I_pred)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim((1e-5, 2))
    plt.show()

    # Apply a cutoff function to filter the intensity data
    q, I = cutoff(q, I[index, :])

    # Plot the filtered intensity data
    plt.figure()
    plt.plot(q, I)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    
    return None

# Function to test the cutoff function with SAXS data
def cutoff_test(*args, **kwargs) -> None:
    # Define the shape of the object being tested
    shape = 'sphere'

    # Retrieve SAXS data for the specified shape
    I, y = get_data(shape=shape)

    # Assign intensity data to X for further processing
    X = I

    # Ensure no zero or negative values in X to avoid computational issues
    X[X <= 0] = np.min(X[X >= 0])

    # Apply logarithmic scaling and tanh normalization to the intensity data
    X = 1 + np.log10(X)/2
    X = np.tanh(X)

    # Add an additional axis to match the expected input shape for certain functions
    X = X[np.newaxis, :, np.newaxis]

    # Generate an array of logarithmic q values for SAXS data
    q_log_arr = np.arange(-2, 0, np.true_divide(1, 128))
    q_arr = np.power(10, q_log_arr - 2*np.log10(2))

    # Select a random index from the dataset
    index = np.random.randint(0, y.shape[0])

    # Apply the cutoff function to the q values and intensity data of the selected index
    q, I = cutoff(q_arr, X[index, :])

    # Plot the flitered intensity data as a function of q
    plt.figure()
    plt.plot(q, I)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    
    return None

# Function to test linear least square fitting
def linear_test(*args, **kwargs) -> None:
    # Generate evenly spaced data points between 0 and 1
    X = np.linspace(0, 1, 256)

    # Create a linear relationship with a slope of 2 and intercept of 1
    Y = 2*X + 1

    # Add random noise to the linear data, scaled by the square of Y
    Y = np.random.normal(Y, np.square(Y)/10)

    # Perform linear least squares fitting to determine the slope and intercept
    m, b = linear_least_square(X=X, Y=Y)

    # Print the calculated slope and intercept
    print(m, b)
    
    return None


# Main function to test sphere and cylinder corrections
def main(*args, **kwargs) -> int:
    
    sphere_correction()
    cylinder_correction()
    
    return 0


if __name__ == '__main__':
    main()
