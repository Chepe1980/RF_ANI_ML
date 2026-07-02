import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
from io import BytesIO
import base64
import scipy.stats as stats
from scipy.optimize import curve_fit
from scipy import signal
import pywt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import lasio
from numpy.fft import fft, ifft, fftfreq, fftshift, ifftshift
import warnings
warnings.filterwarnings('ignore')

# Fix for Bokeh/NumPy compatibility
try:
    if not hasattr(np, 'bool8'):
        np.bool8 = np.bool_
except:
    pass

# Check matplotlib version for compatibility
import matplotlib
matplotlib_version = matplotlib.__version__.split('.')
matplotlib_major = int(matplotlib_version[0])
matplotlib_minor = int(matplotlib_version[1]) if len(matplotlib_version) > 1 else 0

# Bokeh imports - using direct imports without streamlit_bokeh_events
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, CustomJS
from bokeh.palettes import Category10
from bokeh.transform import factor_cmap
from bokeh.embed import components
from bokeh.io import output_notebook, show, save

# Try importing TensorFlow/Keras for CNN
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Try importing rockphypy
try:
    from rockphypy import QI, GM, Fluid
    ROCKPHYPY_AVAILABLE = True
except ImportError:
    ROCKPHYPY_AVAILABLE = False

# Set page config
st.set_page_config(layout="wide", page_title="Enhanced Rock Physics & AVO Modeling with Anisotropy")

# ==============================================
# PNN REGRESSOR CLASS
# ==============================================

class PNNRegressor:
    """
    Probabilistic Neural Network for Regression
    Uses radial basis activation and linear output
    """
    def __init__(self, sigma=1.0):
        self.sigma = sigma
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X.copy()
        self.y_train = y.copy()
        return self

    def predict(self, X):
        # Calculate Euclidean distance between each test and training point
        distances = np.zeros((X.shape[0], self.X_train.shape[0]))
        for i in range(X.shape[0]):
            for j in range(self.X_train.shape[0]):
                distances[i, j] = np.linalg.norm(X[i] - self.X_train[j])

        # Apply Gaussian kernel
        weights = np.exp(-(distances**2) / (2 * self.sigma**2))

        # Weighted prediction
        predictions = np.sum(weights * self.y_train, axis=1) / (np.sum(weights, axis=1) + 1e-10)
        return predictions

# ==============================================
# BACKUS AVERAGING FUNCTIONS
# ==============================================

def backus_average(vp, vs, rho, thickness, window_size=None):
    """
    Apply Backus averaging to anisotropic velocities
    
    Parameters:
    -----------
    vp : array-like
        Compressional wave velocity
    vs : array-like
        Shear wave velocity
    rho : array-like
        Density
    thickness : array-like or float
        Layer thickness (if constant) or array of thicknesses
    window_size : int, optional
        Number of samples in averaging window
    
    Returns:
    --------
    dict : Averaged elastic properties
    """
    if window_size is None:
        window_size = max(5, len(vp) // 20)  # Default to 5% of data length
    
    # Convert to numpy arrays
    vp = np.array(vp)
    vs = np.array(vs)
    rho = np.array(rho)
    
    # Calculate elastic moduli
    mu = rho * vs**2  # Shear modulus
    k = rho * vp**2 - (4/3) * mu  # Bulk modulus
    
    # Backus averaging
    n = len(vp)
    vp_backus = np.zeros(n)
    vs_backus = np.zeros(n)
    rho_backus = np.zeros(n)
    
    half_window = window_size // 2
    
    for i in range(n):
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        
        # Weighted average (by thickness)
        weights = np.ones(end - start)
        if isinstance(thickness, (int, float)):
            weights = np.ones(end - start) * thickness
        else:
            weights = thickness[start:end]
        
        weights = weights / np.sum(weights)
        
        # Backus averaging formulas
        mu_avg = np.sum(weights * mu[start:end])
        k_avg = np.sum(weights * k[start:end])
        rho_avg = np.sum(weights * rho[start:end])
        
        # Voigt-Reuss-Hill average for effective moduli
        mu_eff = mu_avg
        k_eff = k_avg
        
        # Reconstruct velocities
        vp_backus[i] = np.sqrt((k_eff + (4/3) * mu_eff) / rho_avg)
        vs_backus[i] = np.sqrt(mu_eff / rho_avg)
        rho_backus[i] = rho_avg
    
    return {
        'VP_backus': vp_backus,
        'VS_backus': vs_backus,
        'RHO_backus': rho_backus,
        'window_size': window_size
    }

def backus_average_anisotropic(df, window_size=None):
    """
    Apply Backus averaging to anisotropic velocities VP(0), VP(45), VP(90)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing anisotropic velocities
    window_size : int, optional
        Number of samples in averaging window
    
    Returns:
    --------
    pandas.DataFrame : DataFrame with Backus averaged velocities
    """
    df_result = df.copy()
    
    # Check if anisotropic velocities exist
    if 'VP_0' not in df.columns or 'VP_45' not in df.columns or 'VP_90' not in df.columns:
        st.warning("Anisotropic velocities not found. Run anisotropy calculation first.")
        return df_result
    
    # Get parameters
    vp0 = df['VP_0'].values
    vp45 = df['VP_45'].values
    vp90 = df['VP_90'].values
    
    # Use density if available, otherwise estimate
    if 'RHO' in df.columns:
        rho = df['RHO'].values
    elif 'rho' in df.columns:
        rho = df['rho'].values
    else:
        rho = np.ones_like(vp0) * 2.5  # Default density
    
    # Use VS if available, otherwise estimate
    if 'VS' in df.columns:
        vs = df['VS'].values
    elif 'vs' in df.columns:
        vs = df['vs'].values
    else:
        vs = vp0 / 2.0  # Default Vp/Vs ratio
    
    # Get thickness (use depth if available)
    if 'DEPTH' in df.columns:
        depth = df['DEPTH'].values
        thickness = np.diff(depth)
        thickness = np.append(thickness, thickness[-1])  # Add last value
    else:
        thickness = 1.0  # Constant thickness
    
    # Apply Backus averaging to each anisotropic velocity
    window_size = window_size or max(5, len(vp0) // 20)
    
    # For VP(0) - using Vp and Vs
    backus_0 = backus_average(vp0, vs, rho, thickness, window_size)
    
    # For VP(45) - use Vp(45) as the velocity
    # For Backus averaging, we use the same Vs and Rho
    backus_45 = backus_average(vp45, vs, rho, thickness, window_size)
    
    # For VP(90) - use Vp(90) as the velocity
    backus_90 = backus_average(vp90, vs, rho, thickness, window_size)
    
    # Add Backus averaged velocities to DataFrame
    df_result['VP_0_Backus'] = backus_0['VP_backus']
    df_result['VP_45_Backus'] = backus_45['VP_backus']
    df_result['VP_90_Backus'] = backus_90['VP_backus']
    
    # Calculate anisotropy from Backus averaged velocities
    df_result['VP_0_Backus_Aniso'] = ((df_result['VP_90_Backus'] - df_result['VP_0_Backus']) / 
                                       (df_result['VP_0_Backus'] + 1e-10)) * 100
    
    # Add ML predictions for Backus averaged velocities
    df_result['RF_pred_VP_0_Backus'] = 0
    df_result['RF_pred_VP_45_Backus'] = 0
    df_result['RF_pred_VP_90_Backus'] = 0
    df_result['PNN_pred_VP_0_Backus'] = 0
    df_result['PNN_pred_VP_45_Backus'] = 0
    df_result['PNN_pred_VP_90_Backus'] = 0
    
    return df_result

# ==============================================
# ROCK PHYSICS MODEL FUNCTIONS
# ==============================================

def frm(vp1, vs1, rho1, rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi):
    """Gassmann's Fluid Substitution"""
    vp1 = vp1/1000.  # Convert m/s to km/s
    vs1 = vs1/1000.
    mu1 = rho1 * vs1**2
    k_s1 = rho1 * vp1**2 - (4./3.) * mu1

    # Dry rock bulk modulus (Gassmann's equation)
    kdry = (k_s1*((phi*k0)/k_f1 + 1 - phi) - k0) / \
           ((phi*k0)/k_f1 + (k_s1/k0) - 1 - phi + 1e-10)

    # Apply Gassmann to get new fluid properties
    k_s2 = kdry + (1 - (kdry/k0))**2 / \
           ((phi/k_f2) + ((1-phi)/k0) - (kdry/k0**2) + 1e-10)
    rho2 = rho1 - phi*rho_f1 + phi*rho_f2
    mu2 = mu1  # Shear modulus unaffected by fluid change
    vp2 = np.sqrt((k_s2 + (4./3)*mu2) / rho2)
    vs2 = np.sqrt(mu2 / rho2)

    return vp2*1000, vs2*1000, rho2, k_s2

def critical_porosity_model(vp1, vs1, rho1, rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi, phi_c):
    """Critical Porosity Model (Nur et al.)"""
    vp1 = vp1/1000.
    vs1 = vs1/1000.
    mu1 = rho1*vs1**2.
    k_s1 = rho1*vp1**2 - (4./3.)*mu1
    
    # Modified dry rock modulus for critical porosity
    kdry = k0 * (1 - phi/phi_c)
    mudry = mu0 * (1 - phi/phi_c)
    
    # Gassmann substitution
    k_s2 = kdry + (1-(kdry/k0))**2/((phi/k_f2)+((1-phi)/k0)-(kdry/k0**2)+1e-10)
    rho2 = rho1-phi*rho_f1+phi*rho_f2
    mu2 = mudry
    vp2 = np.sqrt((k_s2+(4./3)*mu2)/rho2)
    vs2 = np.sqrt(mu2/rho2)
    
    return vp2*1000, vs2*1000, rho2, k_s2

def hertz_mindlin_model(vp1, vs1, rho1, rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi, Cn, P):
    """Hertz-Mindlin contact theory model"""
    vp1 = vp1/1000.
    vs1 = vs1/1000.
    mu1 = rho1*vs1**2.
    k_s1 = rho1*vp1**2 - (4./3.)*mu1
    
    # Hertz-Mindlin dry rock moduli
    PR0 = (3*k0 - 2*mu0)/(6*k0 + 2*mu0 + 1e-10)  # Poisson's ratio
    kdry = (Cn**2 * (1 - phi)**2 * P * mu0**2 / (18 * np.pi**2 * (1 - PR0)**2 + 1e-10))**(1/3)
    mudry = ((2 + 3*PR0 - PR0**2)/(5*(2 - PR0) + 1e-10)) * (
        (3*Cn**2 * (1 - phi)**2 * P * mu0**2)/(2 * np.pi**2 * (1 - PR0)**2 + 1e-10)
    )**(1/3)
    
    # Gassmann substitution
    k_s2 = kdry + (1-(kdry/k0))**2/((phi/k_f2)+((1-phi)/k0)-(kdry/k0**2)+1e-10)
    rho2 = rho1-phi*rho_f1+phi*rho_f2
    mu2 = mudry
    vp2 = np.sqrt((k_s2+(4./3)*mu2)/rho2)
    vs2 = np.sqrt(mu2/rho2)
    
    return vp2*1000, vs2*1000, rho2, k_s2

def dvorkin_nur_model(vp1, vs1, rho1, rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi, Cn=9, P=10, phi_c=0.4):
    """Dvorkin-Nur Soft Sand model for unconsolidated sands"""
    vp1 = vp1/1000.  # Convert to km/s
    vs1 = vs1/1000.
    
    # Hertz-Mindlin for dry rock moduli at critical porosity
    PR0 = (3*k0 - 2*mu0)/(6*k0 + 2*mu0 + 1e-10)  # Poisson's ratio
    
    # Dry rock moduli at critical porosity
    k_hm = (Cn**2 * (1-phi_c)**2 * P * mu0**2 / (18 * np.pi**2 * (1-PR0)**2 + 1e-10))**(1/3)
    mu_hm = ((2 + 3*PR0 - PR0**2)/(5*(2-PR0) + 1e-10)) * (
        (3*Cn**2 * (1-phi_c)**2 * P * mu0**2)/(2*np.pi**2*(1-PR0)**2 + 1e-10)
    )**(1/3)
    
    # Modified Hashin-Shtrikman lower bound for dry rock
    k_dry = (phi/phi_c)/(k_hm + (4/3)*mu_hm + 1e-10) + (1 - phi/phi_c)/(k0 + (4/3)*mu_hm + 1e-10)
    k_dry = 1/(k_dry + 1e-10) - (4/3)*mu_hm
    k_dry = np.maximum(k_dry, 0)  # Ensure positive values
    
    mu_dry = (phi/phi_c)/(mu_hm + (mu_hm/6)*((9*k_hm + 8*mu_hm)/(k_hm + 2*mu_hm + 1e-10)) + 1e-10) + \
             (1 - phi/phi_c)/(mu0 + (mu_hm/6)*((9*k_hm + 8*mu_hm)/(k_hm + 2*mu_hm + 1e-10)) + 1e-10)
    mu_dry = 1/(mu_dry + 1e-10) - (mu_hm/6)*((9*k_hm + 8*mu_hm)/(k_hm + 2*mu_hm + 1e-10))
    mu_dry = np.maximum(mu_dry, 0)
    
    # Gassmann fluid substitution
    k_sat = k_dry + (1 - (k_dry/k0))**2 / ((phi/k_f2) + ((1-phi)/k0) - (k_dry/k0**2) + 1e-10)
    rho2 = rho1 - phi*rho_f1 + phi*rho_f2
    vp2 = np.sqrt((k_sat + (4/3)*mu_dry)/rho2) * 1000  # Convert back to m/s
    vs2 = np.sqrt(mu_dry/rho2) * 1000
    
    return vp2, vs2, rho2, k_sat

def raymer_hunt_model(vp1, vs1, rho1, rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi):
    """Raymer-Hunt-Gardner empirical model"""
    # Empirical relationships for dry rock
    vp_dry = (1 - phi)**2 * np.sqrt(k0/rho1) + phi * np.sqrt(k_f1/rho_f1)
    vp_dry = vp_dry * 1000  # Convert to m/s
    
    # For saturated rock
    vp_sat = (1 - phi)**2 * np.sqrt(k0/rho1) + phi * np.sqrt(k_f2/rho_f2)
    vp_sat = vp_sat * 1000
    
    # VS is less affected by fluids (use empirical relationship)
    vs_sat = vs1 * (1 - 1.5*phi)  # Simple porosity correction
    
    # Density calculation
    rho2 = rho1 - phi*rho_f1 + phi*rho_f2
    
    return vp_sat, vs_sat, rho2, None

# ==============================================
# AVO AND SEISMIC MODELING FUNCTIONS
# ==============================================

def ricker_wavelet(frequency, length=0.128, dt=0.001):
    """Generate a Ricker wavelet"""
    t = np.linspace(-length/2, length/2, int(length/dt))
    y = (1 - 2*(np.pi**2)*(frequency**2)*(t**2)) * np.exp(-(np.pi**2)*(frequency**2)*(t**2))
    return t, y

def smith_gidlow(vp1, vp2, vs1, vs2, rho1, rho2):
    """Calculate Smith-Gidlow AVO attributes (intercept, gradient)"""
    # Calculate reflectivities
    rp = 0.5 * (vp2 - vp1) / (vp2 + vp1 + 1e-10) + 0.5 * (rho2 - rho1) / (rho2 + rho1 + 1e-10)
    rs = 0.5 * (vs2 - vs1) / (vs2 + vs1 + 1e-10) + 0.5 * (rho2 - rho1) / (rho2 + rho1 + 1e-10)
    
    # Smith-Gidlow coefficients
    intercept = rp
    gradient = rp - 2 * rs
    fluid_factor = rp + 1.16 * (vp1/vs1 + 1e-10) * rs
    
    return intercept, gradient, fluid_factor

def calculate_reflection_coefficients(vp1, vp2, vs1, vs2, rho1, rho2, angle):
    """Calculate PP reflection coefficients using Aki-Richards approximation"""
    theta = np.radians(angle)
    vp_avg = (vp1 + vp2)/2
    vs_avg = (vs1 + vs2)/2
    rho_avg = (rho1 + rho2)/2
    
    dvp = vp2 - vp1
    dvs = vs2 - vs1
    drho = rho2 - rho1
    
    a = 0.5 * (1 + np.tan(theta)**2)
    b = -4 * (vs_avg**2/vp_avg**2 + 1e-10) * np.sin(theta)**2
    c = 0.5 * (1 - 4 * (vs_avg**2/vp_avg**2 + 1e-10) * np.sin(theta)**2)
    
    rc = a*(dvp/vp_avg + 1e-10) + b*(dvs/vs_avg + 1e-10) + c*(drho/rho_avg + 1e-10)
    return rc

def fit_avo_curve(angles, rc_values):
    """Fit a line to AVO response to get intercept and gradient"""
    def linear_func(x, intercept, gradient):
        return intercept + gradient * np.sin(np.radians(x))**2
    
    try:
        popt, pcov = curve_fit(linear_func, angles, rc_values)
        intercept, gradient = popt
        return intercept, gradient, np.sqrt(np.diag(pcov))
    except:
        return np.nan, np.nan, (np.nan, np.nan)

# ==============================================
# WEDGE MODELING FUNCTIONS
# ==============================================

def calc_rc(vp_mod, rho_mod):
    """Calculate reflection coefficients"""
    nlayers = len(vp_mod)
    nint = nlayers - 1
    rc_int = []
    for i in range(0, nint):
        buf1 = vp_mod[i+1]*rho_mod[i+1]-vp_mod[i]*rho_mod[i]
        buf2 = vp_mod[i+1]*rho_mod[i+1]+vp_mod[i]*rho_mod[i]
        buf3 = buf1/(buf2 + 1e-10)
        rc_int.append(buf3)
    return rc_int

def calc_times(z_int, vp_mod):
    """Calculate two-way times to interfaces"""
    nlayers = len(vp_mod)
    nint = nlayers - 1
    t_int = []
    for i in range(0, nint):
        if i == 0:
            tbuf = z_int[i]/vp_mod[i]
            t_int.append(tbuf)
        else:
            zdiff = z_int[i]-z_int[i-1]
            tbuf = 2*zdiff/vp_mod[i] + t_int[i-1]
            t_int.append(tbuf)
    return t_int

def digitize_model(rc_int, t_int, t):
    """Digitize model for convolution"""
    nlayers = len(rc_int)
    nint = nlayers - 1
    nsamp = len(t)
    rc = list(np.zeros(nsamp,dtype='float'))
    lyr = 0
    
    for i in range(0, nsamp):
        if t[i] >= t_int[lyr]:
            rc[i] = rc_int[lyr]
            lyr = lyr + 1    
        if lyr > nint:
            break
    return rc

def plot_vawig(axhdl, data, t, excursion, highlight=None):
    """Plot variable area wiggle traces"""
    [ntrc, nsamp] = data.shape
    t = np.hstack([0, t, t.max()])
    
    for i in range(0, ntrc):
        tbuf = excursion * data[i] / (np.max(np.abs(data)) + 1e-10) + i
        tbuf = np.hstack([i, tbuf, i])
            
        if i==highlight:
            lw = 2
        else:
            lw = 0.5

        axhdl.plot(tbuf, t, color='black', linewidth=lw)
        plt.fill_betweenx(t, tbuf, i, where=tbuf>i, facecolor=[0.6,0.6,1.0], linewidth=0)
        plt.fill_betweenx(t, tbuf, i, where=tbuf<i, facecolor=[1.0,0.7,0.7], linewidth=0)
    
    axhdl.set_xlim((-excursion, ntrc+excursion))
    axhdl.xaxis.tick_top()
    axhdl.xaxis.set_label_position('top')
    axhdl.invert_yaxis()

# ==============================================
# ANISOTROPY AND ML FUNCTIONS
# ==============================================

def calculate_anisotropic_velocities(df):
    """
    Calculate VP(0), VP(45), and VP(90) from Thomsen parameters
    """
    df_result = df.copy()
    
    # Ensure column names are lowercase
    df_result.columns = df_result.columns.str.lower()
    
    # Check if we have the required columns
    if 'vp' not in df_result.columns:
        st.error("Required column 'VP' not found in data")
        return df_result
    
    # Calculate epsilon if Cij are available
    if all(col in df_result.columns for col in ['c11', 'c33']):
        df_result['epsilon'] = (df_result['c11'] - df_result['c33']) / (2 * df_result['c33'] + 1e-10)
    
    # Ensure epsilon exists
    if 'epsilon' not in df_result.columns:
        st.warning("Epsilon not found in data. Using default value of 0.")
        df_result['epsilon'] = 0
    
    # Ensure delta exists
    if 'delta' not in df_result.columns:
        df_result['delta'] = 0
    
    # Calculate velocities using Thomsen's equations
    # VP(0) = VP vertical
    df_result['VP_0'] = df_result['vp']
    
    # VP(45) = VP at 45 degrees
    df_result['VP_45'] = df_result['VP_0'] * (1 + 0.25 * df_result['delta'] + 0.25 * df_result['epsilon'])
    
    # VP(90) = VP horizontal
    df_result['VP_90'] = df_result['VP_0'] * (1 + df_result['epsilon'])
    
    # Calculate anisotropy percentages
    df_result['anisotropy_epsilon'] = df_result['epsilon'] * 100
    df_result['anisotropy_delta'] = df_result['delta'] * 100
    
    # Calculate VP variation percentage
    df_result['VP_variation'] = ((df_result['VP_90'] - df_result['VP_0']) / (df_result['VP_0'] + 1e-10)) * 100
    
    return df_result

def prepare_ml_data(df, target_cols=['VP_0', 'VP_45', 'VP_90']):
    """
    Prepare data for machine learning
    """
    # Select features
    feature_cols = ['vp', 'epsilon']
    
    # Add additional features if available
    additional_features = ['delta', 'rhob', 'vs', 'gama', 'c11', 'c13', 'c33', 'c44', 'c66']
    for col in additional_features:
        if col in df.columns:
            feature_cols.append(col)
    
    # Also add rock physics attributes if available
    rock_physics_features = ['phi', 'vsh', 'sw']
    for col in rock_physics_features:
        if col in df.columns:
            feature_cols.append(col)
    
    X = df[feature_cols].values
    y_dict = {}
    for target in target_cols:
        if target in df.columns:
            y_dict[target] = df[target].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_dict, scaler, feature_cols

def train_ml_models(X, y_dict, test_size=0.2, random_state=42):
    """
    Train Random Forest, PNN, and CNN/MLP models
    """
    results = {}
    models = {}
    
    # Split data
    X_train_dict = {}
    X_test_dict = {}
    y_train_dict = {}
    y_test_dict = {}
    
    for target, y in y_dict.items():
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        X_train_dict[target] = X_train
        X_test_dict[target] = X_test
        y_train_dict[target] = y_train
        y_test_dict[target] = y_test
    
    # 1. Random Forest
    rf_models = {}
    for target in y_dict.keys():
        rf = RandomForestRegressor(
            n_estimators=100, max_depth=20,
            min_samples_split=5, random_state=random_state, n_jobs=-1
        )
        rf.fit(X_train_dict[target], y_train_dict[target])
        rf_models[target] = rf
        y_pred = rf.predict(X_test_dict[target])
        results[f'RF_{target}'] = {
            'RMSE': np.sqrt(mean_squared_error(y_test_dict[target], y_pred)),
            'MAE': mean_absolute_error(y_test_dict[target], y_pred),
            'R2': r2_score(y_test_dict[target], y_pred)
        }
    models['RF'] = rf_models
    
    # 2. PNN
    pnn_models = {}
    sigma_optimo = 0.5
    for target in y_dict.keys():
        # Take a smaller sample for PNN (computational complexity)
        n_samples = min(1000, len(X_train_dict[target]))
        idx = np.random.choice(len(X_train_dict[target]), n_samples, replace=False)
        X_train_pnn = X_train_dict[target][idx]
        y_train_pnn = y_train_dict[target][idx]
        
        pnn = PNNRegressor(sigma=sigma_optimo)
        pnn.fit(X_train_pnn, y_train_pnn)
        pnn_models[target] = pnn
        y_pred = pnn.predict(X_test_dict[target])
        results[f'PNN_{target}'] = {
            'RMSE': np.sqrt(mean_squared_error(y_test_dict[target], y_pred)),
            'MAE': mean_absolute_error(y_test_dict[target], y_pred),
            'R2': r2_score(y_test_dict[target], y_pred)
        }
    models['PNN'] = pnn_models
    
    # 3. CNN or MLP
    if TENSORFLOW_AVAILABLE:
        cnn_models = {}
        for target in y_dict.keys():
            n_features = X_train_dict[target].shape[1]
            X_train_cnn = X_train_dict[target].reshape(-1, n_features, 1)
            X_test_cnn = X_test_dict[target].reshape(-1, n_features, 1)
            
            # Adjust kernel sizes based on feature count
            kernel_size1 = min(3, max(1, n_features))
            kernel_size2 = min(2, max(1, n_features // 2))
            pool_size1 = min(2, max(1, n_features // 2))
            pool_size2 = min(2, max(1, n_features // 4))
            
            model = Sequential([
                Conv1D(filters=32, kernel_size=kernel_size1, activation='relu', 
                       input_shape=(n_features, 1)),
                MaxPooling1D(pool_size=pool_size1),
                Conv1D(filters=16, kernel_size=kernel_size2, activation='relu'),
                MaxPooling1D(pool_size=pool_size2),
                Flatten(),
                Dense(50, activation='relu'),
                Dropout(0.2),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            
            history = model.fit(
                X_train_cnn, y_train_dict[target],
                epochs=50, batch_size=32,
                validation_split=0.1, verbose=0
            )
            
            cnn_models[target] = model
            y_pred = model.predict(X_test_cnn, verbose=0).flatten()
            results[f'CNN_{target}'] = {
                'RMSE': np.sqrt(mean_squared_error(y_test_dict[target], y_pred)),
                'MAE': mean_absolute_error(y_test_dict[target], y_pred),
                'R2': r2_score(y_test_dict[target], y_pred)
            }
        models['CNN'] = cnn_models
    else:
        # Use MLP as alternative
        mlp_models = {}
        for target in y_dict.keys():
            mlp = MLPRegressor(
                hidden_layer_sizes=(100, 50, 25), max_iter=500,
                random_state=random_state, early_stopping=True,
                validation_fraction=0.1, n_iter_no_change=20
            )
            mlp.fit(X_train_dict[target], y_train_dict[target])
            mlp_models[target] = mlp
            y_pred = mlp.predict(X_test_dict[target])
            results[f'CNN_{target}'] = {
                'RMSE': np.sqrt(mean_squared_error(y_test_dict[target], y_pred)),
                'MAE': mean_absolute_error(y_test_dict[target], y_pred),
                'R2': r2_score(y_test_dict[target], y_pred)
            }
        models['CNN'] = mlp_models
    
    return models, results, X_train_dict, X_test_dict, y_train_dict, y_test_dict

def plot_ml_results_with_backus(models, results, X_test_dict, y_test_dict, targets, target_names, df_aniso=None):
    """
    Create ML comparison plots with Backus averaged results
    """
    plots = {}
    
    # Figure 1: Cross-plots
    fig1, axes = plt.subplots(3, 3, figsize=(18, 18))
    
    metodos = ['RF', 'PNN', 'CNN']
    colores = ['blue', 'green', 'red']
    
    for i, metodo in enumerate(metodos):
        for j, target in enumerate(targets):
            ax = axes[i, j]
            
            # Get predictions
            if metodo == 'RF':
                y_pred = models['RF'][target].predict(X_test_dict[target])
            elif metodo == 'PNN':
                y_pred = models['PNN'][target].predict(X_test_dict[target])
            else:
                if TENSORFLOW_AVAILABLE:
                    X_test_cnn = X_test_dict[target].reshape(-1, X_test_dict[target].shape[1], 1)
                    y_pred = models['CNN'][target].predict(X_test_cnn, verbose=0).flatten()
                else:
                    y_pred = models['CNN'][target].predict(X_test_dict[target])
            
            y_true = y_test_dict[target]
            
            ax.scatter(y_true, y_pred, alpha=0.5, s=20, c=colores[i], 
                      edgecolors='k', linewidth=0.2)
            
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, alpha=0.7)
            
            rmse = results[f'{metodo}_{target}']['RMSE']
            r2 = results[f'{metodo}_{target}']['R2']
            ax.text(0.05, 0.95, f'RMSE = {rmse:.1f}\nR² = {r2:.3f}',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel('Real Value (m/s)')
            ax.set_ylabel('Predicted Value (m/s)')
            if i == 0:
                ax.set_title(f'{target_names[j]}', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
    
    for i, metodo in enumerate(metodos):
        axes[i, 0].set_ylabel(f'{metodo}\nPredicted (m/s)', fontweight='bold', fontsize=12)
    
    plt.suptitle('ML Predictions Cross-plots', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plots['crossplots'] = fig1
    
    # Figure 2: RMSE comparison
    fig2, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, target in enumerate(targets):
        ax = axes[i]
        
        metodos_plot = ['RF', 'PNN', 'CNN']
        rmse_values = [results[f'RF_{target}']['RMSE'],
                       results[f'PNN_{target}']['RMSE'],
                       results[f'CNN_{target}']['RMSE']]
        r2_values = [results[f'RF_{target}']['R2'],
                     results[f'PNN_{target}']['R2'],
                     results[f'CNN_{target}']['R2']]
        
        bars = ax.bar(metodos_plot, rmse_values, color=['blue', 'green', 'red'],
                      alpha=0.7, edgecolor='black')
        
        for bar, r2 in zip(bars, r2_values):
            height = bar.get_height()
            ax.annotate(f'R²={r2:.3f}', xy=(bar.get_x() + bar.get_width()/2, height + 5),
                        ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel('RMSE (m/s)')
        ax.set_title(f'{target_names[i]}', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(bottom=0)
    
    plt.suptitle('RMSE Comparison: RF vs PNN vs CNN', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plots['rmse_comparison'] = fig2
    
    # Figure 3: Heatmap of R²
    fig3, ax = plt.subplots(figsize=(10, 7))
    
    r2_matrix = np.zeros((len(metodos), len(targets)))
    for i, metodo in enumerate(metodos):
        for j, target in enumerate(targets):
            r2_matrix[i, j] = results[f'{metodo}_{target}']['R2']
    
    sns.heatmap(r2_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                xticklabels=target_names, yticklabels=metodos,
                cbar_kws={'label': 'R²'}, ax=ax, annot_kws={'size': 14})
    
    ax.set_title('R² Comparison of ML Models', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plots['r2_heatmap'] = fig3
    
    # Figure 4: Backus averaged velocities plot
    if df_aniso is not None:
        fig_backus, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Check if Backus averaged velocities exist
        has_backus = ('VP_0_Backus' in df_aniso.columns and 
                     'VP_45_Backus' in df_aniso.columns and 
                     'VP_90_Backus' in df_aniso.columns)
        
        if has_backus:
            # Plot 1: Original vs Backus averaged VP(0)
            ax1 = axes[0, 0]
            depth_col = 'DEPTH' if 'DEPTH' in df_aniso.columns else None
            x_vals = df_aniso[depth_col] if depth_col else range(len(df_aniso))
            
            ax1.plot(x_vals, df_aniso['VP_0'], label='Original VP(0)', linewidth=1.5, alpha=0.7, color='blue')
            ax1.plot(x_vals, df_aniso['VP_0_Backus'], label='Backus VP(0)', linewidth=2.5, color='darkblue', linestyle='--')
            ax1.set_xlabel('Depth (m)' if depth_col else 'Sample')
            ax1.set_ylabel('Velocity (m/s)')
            ax1.set_title('VP(0): Original vs Backus Average')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Original vs Backus averaged VP(45)
            ax2 = axes[0, 1]
            ax2.plot(x_vals, df_aniso['VP_45'], label='Original VP(45)', linewidth=1.5, alpha=0.7, color='green')
            ax2.plot(x_vals, df_aniso['VP_45_Backus'], label='Backus VP(45)', linewidth=2.5, color='darkgreen', linestyle='--')
            ax2.set_xlabel('Depth (m)' if depth_col else 'Sample')
            ax2.set_ylabel('Velocity (m/s)')
            ax2.set_title('VP(45): Original vs Backus Average')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Original vs Backus averaged VP(90)
            ax3 = axes[1, 0]
            ax3.plot(x_vals, df_aniso['VP_90'], label='Original VP(90)', linewidth=1.5, alpha=0.7, color='red')
            ax3.plot(x_vals, df_aniso['VP_90_Backus'], label='Backus VP(90)', linewidth=2.5, color='darkred', linestyle='--')
            ax3.set_xlabel('Depth (m)' if depth_col else 'Sample')
            ax3.set_ylabel('Velocity (m/s)')
            ax3.set_title('VP(90): Original vs Backus Average')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Anisotropy comparison
            ax4 = axes[1, 1]
            ax4.plot(x_vals, df_aniso['VP_variation'], label='Original Anisotropy', linewidth=1.5, alpha=0.7, color='orange')
            if 'VP_0_Backus_Aniso' in df_aniso.columns:
                ax4.plot(x_vals, df_aniso['VP_0_Backus_Aniso'], label='Backus Anisotropy', linewidth=2.5, color='darkorange', linestyle='--')
            ax4.set_xlabel('Depth (m)' if depth_col else 'Sample')
            ax4.set_ylabel('Anisotropy (%)')
            ax4.set_title('Anisotropy Comparison')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.suptitle('Backus Averaging of Anisotropic Velocities', fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            plots['backus_averaging'] = fig_backus
    
    # Figure 5: Boxplot of errors - FIXED for matplotlib compatibility
    fig5, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, target in enumerate(targets):
        ax = axes[i]
        
        errores = []
        method_labels = ['RF', 'PNN', 'CNN']
        
        for metodo in method_labels:
            if metodo == 'RF':
                y_pred = models['RF'][target].predict(X_test_dict[target])
            elif metodo == 'PNN':
                y_pred = models['PNN'][target].predict(X_test_dict[target])
            else:
                if TENSORFLOW_AVAILABLE:
                    X_test_cnn = X_test_dict[target].reshape(-1, X_test_dict[target].shape[1], 1)
                    y_pred = models['CNN'][target].predict(X_test_cnn, verbose=0).flatten()
                else:
                    y_pred = models['CNN'][target].predict(X_test_dict[target])
            
            error = y_pred - y_test_dict[target]
            errores.append(error)
        
        # Version-compatible boxplot
        if matplotlib_major >= 3 and matplotlib_minor >= 9:
            bp = ax.boxplot(errores, patch_artist=True,
                            tick_labels=method_labels,
                            boxprops=dict(facecolor='lightblue', alpha=0.7))
        else:
            bp = ax.boxplot(errores, patch_artist=True,
                            positions=range(1, len(method_labels) + 1),
                            boxprops=dict(facecolor='lightblue', alpha=0.7))
            ax.set_xticklabels(method_labels)
        
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1)
        ax.set_ylabel('Error (m/s)')
        ax.set_title(f'{target_names[i]}', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Error Distribution: RF vs PNN vs CNN', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plots['error_distribution'] = fig5
    
    # Figure 6: Summary comparison
    fig6, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(targets))
    width = 0.25
    metodos_plot = ['RF', 'PNN', 'CNN']
    colores_plot = ['blue', 'green', 'red']
    
    for i, (metodo, color) in enumerate(zip(metodos_plot, colores_plot)):
        rmse_vals = [results[f'{metodo}_{target}']['RMSE'] for target in targets]
        bars = ax.bar(x + i*width - width, rmse_vals, width,
                      label=metodo, color=color, alpha=0.7, edgecolor='black')
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width()/2, height + 5),
                        ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Target Variable')
    ax.set_ylabel('RMSE (m/s)')
    ax.set_title('Final Comparison: RMSE of RF vs PNN vs CNN', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(target_names)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plots['summary_comparison'] = fig6
    
    return plots

# ==============================================
# DATA PROCESSING FUNCTION
# ==============================================

def process_data(uploaded_file, model_choice, **kwargs):
    """
    Process data with rock physics models
    """
    # Read data
    if isinstance(uploaded_file, str):
        logs = pd.read_csv(uploaded_file)
    else:
        uploaded_file.seek(0)
        if uploaded_file.name.endswith('.las'):
            las = lasio.read(uploaded_file)
            logs = las.df()
            logs['DEPTH'] = logs.index
        else:
            logs = pd.read_csv(uploaded_file)
    
    # Ensure required columns exist
    required_columns = {'DEPTH', 'VP', 'VS', 'RHO', 'VSH', 'SW', 'PHI'}
    missing = required_columns - set(logs.columns)
    for col in missing:
        if col == 'SW':
            logs['SW'] = 0.8
        elif col == 'PHI':
            logs['PHI'] = 0.15
        elif col == 'VSH':
            logs['VSH'] = 0.2
    
    # Extract parameters
    rho_qz = kwargs.get('rho_qz', 2.65)
    k_qz = kwargs.get('k_qz', 37.0)
    mu_qz = kwargs.get('mu_qz', 44.0)
    rho_sh = kwargs.get('rho_sh', 2.81)
    k_sh = kwargs.get('k_sh', 15.0)
    mu_sh = kwargs.get('mu_sh', 5.0)
    rho_b = kwargs.get('rho_b', 1.09)
    k_b = kwargs.get('k_b', 2.8)
    rho_o = kwargs.get('rho_o', 0.78)
    k_o = kwargs.get('k_o', 0.94)
    rho_g = kwargs.get('rho_g', 0.25)
    k_g = kwargs.get('k_g', 0.06)
    sand_cutoff = kwargs.get('sand_cutoff', 0.12)
    sw = kwargs.get('sw', 0.8)
    so = kwargs.get('so', 0.15)
    sg = kwargs.get('sg', 0.05)
    
    # VRH function
    def vrh(volumes, k, mu):
        f = np.array(volumes).T
        k = np.resize(np.array(k), np.shape(f))
        mu = np.resize(np.array(mu), np.shape(f))
        
        k_u = np.sum(f*k, axis=1)
        k_l = 1. / (np.sum(f/k, axis=1) + 1e-10)
        mu_u = np.sum(f*mu, axis=1)
        mu_l = 1. / (np.sum(f/mu, axis=1) + 1e-10)
        k0 = (k_u+k_l)/2.
        mu0 = (mu_u+mu_l)/2.
        return k_u, k_l, mu_u, mu_l, k0, mu0
    
    # Process data
    shale = logs.VSH.values
    sand = 1 - shale - logs.PHI.values
    sand = np.maximum(sand, 0)
    shaleN = shale/(shale+sand+1e-10)
    sandN = sand/(shale+sand+1e-10)
    k_u, k_l, mu_u, mu_l, k0, mu0 = vrh([shaleN, sandN], [k_sh, k_qz], [mu_sh, mu_qz])
    
    # Fluid mixtures
    water = sw
    oil = so
    gas = sg
    rho_fl = water*rho_b + oil*rho_o + gas*rho_g
    k_fl = 1.0 / (water/k_b + oil/k_o + gas/k_g + 1e-10)
    
    # Select model function
    phi = logs.PHI.values
    vp_array = logs.VP.values
    vs_array = logs.VS.values
    rho_array = logs.RHO.values
    
    if model_choice == "Gassmann's Fluid Substitution":
        def model_func(rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi):
            return frm(logs.VP, logs.VS, logs.RHO, rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi)
    elif model_choice == "Critical Porosity Model (Nur)":
        phi_c = kwargs.get('critical_porosity', 0.4)
        def model_func(rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi, phi_c):
            return critical_porosity_model(logs.VP, logs.VS, logs.RHO, rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi, phi_c)
    elif model_choice == "Contact Theory (Hertz-Mindlin)":
        Cn = kwargs.get('coordination_number', 9)
        P = kwargs.get('effective_pressure', 10)
        def model_func(rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi, Cn, P):
            return hertz_mindlin_model(logs.VP, logs.VS, logs.RHO, rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi, Cn, P)
    elif model_choice == "Dvorkin-Nur Soft Sand Model":
        Cn = kwargs.get('coordination_number', 9)
        P = kwargs.get('effective_pressure', 10)
        phi_c = kwargs.get('critical_porosity', 0.4)
        def model_func(rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi, Cn, P, phi_c):
            return dvorkin_nur_model(logs.VP, logs.VS, logs.RHO, rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi, Cn, P, phi_c)
    elif model_choice == "Raymer-Hunt-Gardner Model":
        def model_func(rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi):
            return raymer_hunt_model(logs.VP, logs.VS, logs.RHO, rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi)
    
    # Apply model for each fluid case
    fluid_cases = {
        'B': (rho_b, k_b, rho_b, k_b),
        'O': (rho_b, k_b, rho_o, k_o),
        'G': (rho_b, k_b, rho_g, k_g),
        'MIX': (rho_b, k_b, rho_fl, k_fl)
    }
    
    for case, (rho_f1, k_f1, rho_f2, k_f2) in fluid_cases.items():
        if model_choice == "Gassmann's Fluid Substitution":
            vp, vs, rho, k = model_func(rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi)
        elif model_choice == "Critical Porosity Model (Nur)":
            vp, vs, rho, k = model_func(rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi, phi_c)
        elif model_choice in ["Contact Theory (Hertz-Mindlin)", "Dvorkin-Nur Soft Sand Model"]:
            vp, vs, rho, k = model_func(rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi, Cn, P, phi_c)
        elif model_choice == "Raymer-Hunt-Gardner Model":
            vp, vs, rho, k = model_func(rho_f1, k_f1, rho_f2, k_f2, k0, mu0, phi)
        
        logs[f'VP_FRM{case}'] = vp
        logs[f'VS_FRM{case}'] = vs
        logs[f'RHO_FRM{case}'] = rho
        logs[f'IP_FRM{case}'] = vp * rho
        logs[f'VPVS_FRM{case}'] = vp / (vs + 1e-10)
    
    # Store in session state
    st.session_state.logs = logs
    st.session_state.model_choice = model_choice
    
    return logs

# ==============================================
# BOKEH CROSSPLOT FUNCTION
# ==============================================

def create_bokeh_crossplot(logs):
    """Create interactive Bokeh crossplot using components"""
    try:
        # Prepare data
        plot_data = logs[['IP_FRMMIX', 'VPVS_FRMMIX', 'DEPTH']].copy()
        plot_data = plot_data.dropna()
        
        # Filter unrealistic values
        plot_data = plot_data[
            (plot_data['IP_FRMMIX'] > 0) & 
            (plot_data['IP_FRMMIX'] < 30000) & 
            (plot_data['VPVS_FRMMIX'] > 1.0) & 
            (plot_data['VPVS_FRMMIX'] < 4.0)
        ]
        
        if len(plot_data) == 0:
            st.warning("No valid data for crossplot")
            return None
        
        # Create Bokeh figure
        source = ColumnDataSource(plot_data)
        p = figure(width=800, height=500, 
                   tools="pan,wheel_zoom,box_zoom,reset,box_select,lasso_select",
                   title="IP vs Vp/Vs Crossplot")
        
        p.scatter('IP_FRMMIX', 'VPVS_FRMMIX', source=source, size=5, 
                  alpha=0.6, color='navy')
        
        p.xaxis.axis_label = 'IP (m/s*g/cc)'
        p.yaxis.axis_label = 'Vp/Vs'
        
        hover = HoverTool(tooltips=[
            ("Depth", "@DEPTH{0.2f}"),
            ("IP", "@IP_FRMMIX{0.2f}"),
            ("Vp/Vs", "@VPVS_FRMMIX{0.2f}")
        ])
        p.add_tools(hover)
        
        # Generate components
        script, div = components(p)
        
        # Display with HTML
        st.components.v1.html(
            f"""
            <!DOCTYPE html>
            <html>
            <head>
                <link rel="stylesheet" href="https://cdn.bokeh.org/bokeh/release/bokeh-2.4.3.min.css" 
                      type="text/css" />
                <script type="text/javascript" src="https://cdn.bokeh.org/bokeh/release/bokeh-2.4.3.min.js"></script>
                {script}
            </head>
            <body>
                {div}
            </body>
            </html>
            """,
            height=550,
            scrolling=True
        )
        
        return p
        
    except Exception as e:
        st.warning(f"Bokeh crossplot error: {str(e)}")
        # Fallback to matplotlib
        fig, ax = plt.subplots(figsize=(8, 5))
        if 'IP_FRMMIX' in logs.columns and 'VPVS_FRMMIX' in logs.columns:
            ax.scatter(logs['IP_FRMMIX'], logs['VPVS_FRMMIX'], alpha=0.5, s=5)
            ax.set_xlabel('IP (m/s*g/cc)')
            ax.set_ylabel('Vp/Vs')
            ax.set_title('IP vs Vp/Vs Crossplot')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        return None

# ==============================================
# MAIN APPLICATION
# ==============================================

def main():
    st.title("🌍 Enhanced Rock Physics & AVO Modeling with Anisotropy Analysis")
    st.markdown("""
    This comprehensive tool combines:
    - **Rock Physics Modeling**: Gassmann, Critical Porosity, Hertz-Mindlin, Dvorkin-Nur, Raymer-Hunt
    - **AVO Analysis**: Reflection coefficients, Smith-Gidlow attributes, synthetic gathers
    - **Wedge Modeling**: Seismic wedge models, tuning thickness analysis
    - **Anisotropy**: VP(0), VP(45), VP(90) calculations from Thomsen parameters
    - **Backus Averaging**: Scale up anisotropic velocities to seismic scale
    - **Machine Learning**: Random Forest, PNN, CNN/MLP for velocity prediction
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Mode selection
        mode = st.radio(
            "Select Analysis Mode",
            ["Rock Physics & AVO", "Anisotropy & ML Prediction", "Combined Analysis"],
            index=0,
            help="Choose the analysis mode for your workflow"
        )
        
        # File upload
        st.subheader("📂 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload CSV or LAS file",
            type=["csv", "las"],
            help="Upload well log data for analysis"
        )
        
        if mode in ["Rock Physics & AVO", "Combined Analysis"]:
            st.subheader("🔬 Rock Physics Model")
            model_options = [
                "Gassmann's Fluid Substitution",
                "Critical Porosity Model (Nur)",
                "Contact Theory (Hertz-Mindlin)",
                "Dvorkin-Nur Soft Sand Model",
                "Raymer-Hunt-Gardner Model"
            ]
            
            if ROCKPHYPY_AVAILABLE:
                model_options.extend(["Soft Sand RPT", "Stiff Sand RPT"])
            
            model_choice = st.selectbox("Select Model", model_options, index=0)
            
            # Mineral properties
            st.subheader("Mineral Properties")
            col1, col2 = st.columns(2)
            with col1:
                rho_qz = st.number_input("Quartz Density (g/cc)", value=2.65, step=0.01)
                k_qz = st.number_input("Quartz Bulk Modulus (GPa)", value=37.0, step=0.1)
                mu_qz = st.number_input("Quartz Shear Modulus (GPa)", value=44.0, step=0.1)
            with col2:
                rho_sh = st.number_input("Shale Density (g/cc)", value=2.81, step=0.01)
                k_sh = st.number_input("Shale Bulk Modulus (GPa)", value=15.0, step=0.1)
                mu_sh = st.number_input("Shale Shear Modulus (GPa)", value=5.0, step=0.1)
            
            # Fluid properties
            st.subheader("Fluid Properties")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Brine**")
                rho_b = st.number_input("Brine Density (g/cc)", value=1.09, step=0.01)
                k_b = st.number_input("Brine Bulk Modulus (GPa)", value=2.8, step=0.1)
            with col2:
                st.markdown("**Oil**")
                rho_o = st.number_input("Oil Density (g/cc)", value=0.78, step=0.01)
                k_o = st.number_input("Oil Bulk Modulus (GPa)", value=0.94, step=0.1)
            with col3:
                st.markdown("**Gas**")
                rho_g = st.number_input("Gas Density (g/cc)", value=0.25, step=0.01)
                k_g = st.number_input("Gas Bulk Modulus (GPa)", value=0.06, step=0.01)
            
            # Saturation
            st.subheader("Saturation Settings")
            sw = st.slider("Water Saturation (Sw)", 0.0, 1.0, 0.8, 0.01)
            remaining = max(0.0, 1.0 - sw)
            so = st.slider("Oil Saturation (So)", 0.0, remaining, 0.15, 0.01)
            sg = remaining - so
            st.write(f"**Gas Saturation (Sg):** {sg:.2f}")
            
            # Additional parameters
            if "Critical Porosity" in model_choice:
                critical_porosity = st.slider("Critical Porosity (φc)", 0.3, 0.5, 0.4, 0.01)
            if "Hertz-Mindlin" in model_choice or "Dvorkin-Nur" in model_choice:
                coordination_number = st.slider("Coordination Number", 6, 12, 9)
                effective_pressure = st.slider("Effective Pressure (MPa)", 1, 50, 10)
            
            # AVO parameters
            st.subheader("AVO Parameters")
            min_angle = st.slider("Minimum Angle (deg)", 0, 10, 0)
            max_angle = st.slider("Maximum Angle (deg)", 30, 50, 45)
            angle_step = st.slider("Angle Step (deg)", 1, 5, 1)
            wavelet_freq = st.slider("Wavelet Frequency (Hz)", 20, 80, 50)
            sand_cutoff = st.slider("Sand Cutoff (VSH)", 0.0, 0.3, 0.12, 0.01)
        
        if mode in ["Anisotropy & ML Prediction", "Combined Analysis"]:
            st.subheader("📐 Anisotropy Parameters")
            
            # Thomsen parameters
            st.markdown("**Thomsen Parameters**")
            epsilon_default = st.number_input("Epsilon (ε)", value=0.1, step=0.01, 
                                             help="Thomsen's epsilon parameter")
            delta_default = st.number_input("Delta (δ)", value=0.05, step=0.01,
                                           help="Thomsen's delta parameter")
            
            # Backus averaging parameters
            st.subheader("📊 Backus Averaging")
            use_backus = st.checkbox("Apply Backus Averaging", value=True)
            backus_window = st.slider("Averaging Window Size", 5, 50, 10, 
                                     help="Number of samples for Backus averaging")
            
            # ML settings
            st.markdown("**🤖 Machine Learning Settings**")
            ml_models_selected = st.multiselect(
                "Select ML Models",
                ["Random Forest", "PNN", "CNN"],
                default=["Random Forest", "PNN"],
                help="Select which ML models to train and compare"
            )
            
            test_size = st.slider("Test Split Size", 0.1, 0.4, 0.2, 0.05)
            
            # ML training button
            run_ml = st.button("🚀 Run Machine Learning", use_container_width=True)
        
        # Wedge modeling
        if mode in ["Rock Physics & AVO", "Combined Analysis"]:
            show_wedge = st.checkbox("Show Wedge Modeling", value=False)
    
    # Main content area
    if uploaded_file is not None:
        try:
            # Load data
            if uploaded_file.name.endswith('.las'):
                las = lasio.read(uploaded_file)
                df = las.df()
                df['DEPTH'] = df.index
            else:
                df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Data loaded successfully! {len(df)} samples loaded.")
            
            # Show data preview
            with st.expander("📊 Data Preview"):
                st.dataframe(df.head(10))
                st.caption(f"Total samples: {len(df)}, Columns: {list(df.columns)}")
            
            # Process based on mode
            if mode in ["Rock Physics & AVO", "Combined Analysis"]:
                st.header("🔬 Rock Physics Analysis")
                
                # Process data with rock physics
                kwargs = {
                    'rho_qz': rho_qz, 'k_qz': k_qz, 'mu_qz': mu_qz,
                    'rho_sh': rho_sh, 'k_sh': k_sh, 'mu_sh': mu_sh,
                    'rho_b': rho_b, 'k_b': k_b,
                    'rho_o': rho_o, 'k_o': k_o,
                    'rho_g': rho_g, 'k_g': k_g,
                    'sand_cutoff': sand_cutoff,
                    'sw': sw, 'so': so, 'sg': sg
                }
                
                if "Critical Porosity" in model_choice:
                    kwargs['critical_porosity'] = critical_porosity
                if "Hertz-Mindlin" in model_choice or "Dvorkin-Nur" in model_choice:
                    kwargs['coordination_number'] = coordination_number
                    kwargs['effective_pressure'] = effective_pressure
                
                logs = process_data(uploaded_file, model_choice, **kwargs)
                
                if logs is not None:
                    st.success("✅ Rock physics modeling completed!")
                    
                    # Depth range selection
                    ztop, zbot = st.slider(
                        "Select Depth Range",
                        float(logs.DEPTH.min()),
                        float(logs.DEPTH.max()),
                        (float(logs.DEPTH.min()), float(logs.DEPTH.max()))
                    )
                    
                    # Well log visualization
                    st.subheader("Well Log Visualization")
                    
                    # Create the well log figure
                    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(12, 8))
                    
                    ll = logs.loc[(logs.DEPTH>=ztop) & (logs.DEPTH<=zbot)]
                    
                    # VSH, SW, PHI
                    axes[0].plot(ll.VSH, ll.DEPTH, '-g', label='Vsh')
                    if 'SW' in ll.columns:
                        axes[0].plot(ll.SW, ll.DEPTH, '-b', label='Sw')
                    axes[0].plot(ll.PHI, ll.DEPTH, '-k', label='phi')
                    
                    # IP curves
                    axes[1].plot(ll.IP_FRMG, ll.DEPTH, '-r', label='Gas')
                    axes[1].plot(ll.IP_FRMB, ll.DEPTH, '-b', label='Brine')
                    if 'IP_FRMO' in ll.columns:
                        axes[1].plot(ll.IP_FRMO, ll.DEPTH, '-g', label='Oil')
                    axes[1].plot(ll.IP_FRMMIX, ll.DEPTH, '-m', label='Mixed')
                    axes[1].plot(ll.VP*ll.RHO, ll.DEPTH, '-', color='0.5', label='Original')
                    
                    # VPVS curves
                    axes[2].plot(ll.VPVS_FRMG, ll.DEPTH, '-r', label='Gas')
                    axes[2].plot(ll.VPVS_FRMB, ll.DEPTH, '-b', label='Brine')
                    if 'VPVS_FRMO' in ll.columns:
                        axes[2].plot(ll.VPVS_FRMO, ll.DEPTH, '-g', label='Oil')
                    axes[2].plot(ll.VPVS_FRMMIX, ll.DEPTH, '-m', label='Mixed')
                    axes[2].plot(ll.VP/ll.VS, ll.DEPTH, '-', color='0.5', label='Original')
                    
                    # LFC (Litho-Fluid Classification)
                    ccc = ['#B3B3B3','blue','green','red','magenta','#996633']
                    cmap_facies = colors.ListedColormap(ccc[0:len(ccc)], 'indexed')
                    cluster_col = 'LFC_MIX' if 'LFC_MIX' in logs.columns else 'LFC_B'
                    if cluster_col in ll.columns:
                        cluster = np.repeat(np.expand_dims(ll[cluster_col].values,1), 100, 1)
                        im = axes[3].imshow(cluster, interpolation='none', aspect='auto', cmap=cmap_facies, vmin=0, vmax=5)
                        cbar = plt.colorbar(im, ax=axes[3])
                        cbar.set_label((12*' ').join(['undef', 'brine', 'oil', 'gas', 'mixed', 'shale']))
                        cbar.set_ticks(range(0,6))
                        cbar.set_ticklabels(['']*6)
                    
                    # Formatting
                    for ax in axes[:-1]:
                        ax.set_ylim(ztop,zbot)
                        ax.invert_yaxis()
                        ax.grid()
                        ax.locator_params(axis='x', nbins=4)
                    
                    axes[0].legend(fontsize='small', loc='lower right')
                    axes[1].legend(fontsize='small', loc='lower right')
                    axes[2].legend(fontsize='small', loc='lower right')
                    axes[0].set_xlabel("Vcl/phi/Sw")
                    axes[0].set_xlim(-.1,1.1)
                    axes[1].set_xlabel("Ip [m/s*g/cc]")
                    axes[1].set_xlim(6000,15000)
                    axes[2].set_xlabel("Vp/Vs")
                    axes[2].set_xlim(1.5,2)
                    axes[3].set_xlabel('LFC')
                    axes[1].set_yticklabels([])
                    axes[2].set_yticklabels([])
                    axes[3].set_yticklabels([])
                    axes[3].set_xticklabels([])
                    
                    st.pyplot(fig)
                    plt.close()
                    
                    # AVO Modeling
                    st.header("AVO Modeling")
                    
                    # Select interface for AVO analysis
                    middle_top = ztop + (zbot - ztop) * 0.4
                    middle_bot = ztop + (zbot - ztop) * 0.6
                    
                    cases = ['Brine', 'Oil', 'Gas', 'Mixed']
                    case_data = {
                        'Brine': {'vp': 'VP_FRMB', 'vs': 'VS_FRMB', 'rho': 'RHO_FRMB', 'color': 'b'},
                        'Oil': {'vp': 'VP_FRMO', 'vs': 'VS_FRMO', 'rho': 'RHO_FRMO', 'color': 'g'},
                        'Gas': {'vp': 'VP_FRMG', 'vs': 'VS_FRMG', 'rho': 'RHO_FRMG', 'color': 'r'},
                        'Mixed': {'vp': 'VP_FRMMIX', 'vs': 'VS_FRMMIX', 'rho': 'RHO_FRMMIX', 'color': 'm'}
                    }
                    
                    wlt_time, wlt_amp = ricker_wavelet(wavelet_freq)
                    angles = np.arange(min_angle, max_angle + 1, angle_step)
                    
                    fig3, (ax_wavelet, ax_avo) = plt.subplots(1, 2, figsize=(12, 5), 
                                                             gridspec_kw={'width_ratios': [1, 2]})
                    
                    ax_wavelet.plot(wlt_time, wlt_amp, color='purple', linewidth=2)
                    ax_wavelet.fill_between(wlt_time, wlt_amp, color='purple', alpha=0.3)
                    ax_wavelet.set_title(f"Wavelet ({wavelet_freq} Hz)")
                    ax_wavelet.set_xlabel("Time (s)")
                    ax_wavelet.set_ylabel("Amplitude")
                    ax_wavelet.grid(True)
                    
                    avo_attributes = {'Case': [], 'Intercept': [], 'Gradient': [], 'Fluid_Factor': []}
                    
                    for case in cases:
                        if case == 'Oil' and 'VP_FRMO' not in logs.columns:
                            continue
                            
                        vp_upper = logs.loc[(logs.DEPTH >= middle_top - (middle_bot-middle_top)), 'VP'].values.mean()
                        vs_upper = logs.loc[(logs.DEPTH >= middle_top - (middle_bot-middle_top)), 'VS'].values.mean()
                        rho_upper = logs.loc[(logs.DEPTH >= middle_top - (middle_bot-middle_top)), 'RHO'].values.mean()
                        
                        vp_middle = logs.loc[(logs.DEPTH >= middle_top) & (logs.DEPTH <= middle_bot), case_data[case]['vp']].values.mean()
                        vs_middle = logs.loc[(logs.DEPTH >= middle_top) & (logs.DEPTH <= middle_bot), case_data[case]['vs']].values.mean()
                        rho_middle = logs.loc[(logs.DEPTH >= middle_top) & (logs.DEPTH <= middle_bot), case_data[case]['rho']].values.mean()
                        
                        rc = []
                        for angle in angles:
                            rc_val = calculate_reflection_coefficients(
                                vp_upper, vp_middle, vs_upper, vs_middle, rho_upper, rho_middle, angle
                            )
                            rc.append(rc_val)
                        
                        intercept, gradient, _ = fit_avo_curve(angles, rc)
                        fluid_factor = intercept + 1.16 * (vp_upper/vs_upper) * (intercept - gradient)
                        
                        avo_attributes['Case'].append(case)
                        avo_attributes['Intercept'].append(intercept)
                        avo_attributes['Gradient'].append(gradient)
                        avo_attributes['Fluid_Factor'].append(fluid_factor)
                        
                        ax_avo.plot(angles, rc, f"{case_data[case]['color']}-", label=f"{case}")
                    
                    ax_avo.set_title("AVO Reflection Coefficients (Middle Interface)")
                    ax_avo.set_xlabel("Angle (degrees)")
                    ax_avo.set_ylabel("Reflection Coefficient")
                    ax_avo.grid(True)
                    ax_avo.legend()
                    
                    st.pyplot(fig3)
                    plt.close()
                    
                    # Smith-Gidlow AVO Attributes
                    st.subheader("Smith-Gidlow AVO Attributes")
                    avo_df = pd.DataFrame(avo_attributes)
                    st.dataframe(avo_df)
                    
                    # Wedge Modeling
                    if show_wedge:
                        st.header("Seismic Wedge Modeling")
                        st.info("Wedge modeling would be displayed here")
                        
                        # Simplified wedge model
                        st.subheader("Wedge Model Parameters")
                        col1, col2 = st.columns(2)
                        with col1:
                            wedge_vp1 = st.number_input("Layer 1 VP (m/s)", value=float(logs.VP.mean()), step=100)
                            wedge_vs1 = st.number_input("Layer 1 VS (m/s)", value=float(logs.VS.mean()), step=50)
                            wedge_rho1 = st.number_input("Layer 1 Rho (g/cc)", value=float(logs.RHO.mean()), step=0.1)
                        with col2:
                            wedge_vp2 = st.number_input("Layer 2 VP (m/s)", value=float(logs.VP.mean()*1.2), step=100)
                            wedge_vs2 = st.number_input("Layer 2 VS (m/s)", value=float(logs.VS.mean()*1.1), step=50)
                            wedge_rho2 = st.number_input("Layer 2 Rho (g/cc)", value=float(logs.RHO.mean()*1.05), step=0.1)
                        
                        if st.button("Generate Wedge Model"):
                            st.info("Wedge model generation would run here")
            
            # Anisotropy & ML Analysis
            if mode in ["Anisotropy & ML Prediction", "Combined Analysis"]:
                st.header("📐 Anisotropy & Machine Learning Analysis")
                
                # Calculate anisotropic velocities
                df_aniso = calculate_anisotropic_velocities(df)
                
                # Apply Backus averaging if selected
                if 'use_backus' in locals() and use_backus:
                    st.subheader("📊 Backus Averaging Applied")
                    df_aniso = backus_average_anisotropic(df_aniso, window_size=backus_window)
                    st.info(f"Backus averaging applied with window size: {backus_window} samples")
                
                # Show anisotropic velocity statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    vp0_mean = df_aniso['VP_0'].mean()
                    st.metric("VP(0)", f"{vp0_mean:.0f} m/s")
                with col2:
                    vp45_mean = df_aniso['VP_45'].mean()
                    st.metric("VP(45)", f"{vp45_mean:.0f} m/s")
                with col3:
                    vp90_mean = df_aniso['VP_90'].mean()
                    st.metric("VP(90)", f"{vp90_mean:.0f} m/s")
                with col4:
                    variation = ((df_aniso['VP_90'].mean() - df_aniso['VP_0'].mean()) / (df_aniso['VP_0'].mean() + 1e-10)) * 100
                    st.metric("Anisotropy", f"{variation:.1f}%")
                
                # Anisotropy visualization - UPDATED with Depth as y-axis
                st.subheader("Anisotropic Velocity Profiles")
                
                # Determine y-axis variable
                depth_col = 'DEPTH' if 'DEPTH' in df_aniso.columns else None
                if depth_col:
                    y_vals = df_aniso[depth_col].values
                    y_label = 'Depth (m)'
                else:
                    y_vals = np.arange(len(df_aniso))
                    y_label = 'Sample Number'
                
                fig_aniso, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # Plot 1: Anisotropic velocities vs Depth
                ax1 = axes[0, 0]
                ax1.plot(df_aniso['VP_0'], y_vals, label='VP(0)', linewidth=2, color='blue')
                ax1.plot(df_aniso['VP_45'], y_vals, label='VP(45)', linewidth=2, color='green')
                ax1.plot(df_aniso['VP_90'], y_vals, label='VP(90)', linewidth=2, color='red')
                
                # Add Backus averaged if available
                if 'VP_0_Backus' in df_aniso.columns:
                    ax1.plot(df_aniso['VP_0_Backus'], y_vals, label='Backus VP(0)', linewidth=3, 
                            color='darkblue', linestyle='--', alpha=0.8)
                if 'VP_45_Backus' in df_aniso.columns:
                    ax1.plot(df_aniso['VP_45_Backus'], y_vals, label='Backus VP(45)', linewidth=3, 
                            color='darkgreen', linestyle='--', alpha=0.8)
                if 'VP_90_Backus' in df_aniso.columns:
                    ax1.plot(df_aniso['VP_90_Backus'], y_vals, label='Backus VP(90)', linewidth=3, 
                            color='darkred', linestyle='--', alpha=0.8)
                
                ax1.set_xlabel('Velocity (m/s)')
                ax1.set_ylabel(y_label)
                ax1.set_title('Anisotropic Velocities with Backus Average')
                ax1.legend(loc='lower right')
                ax1.grid(True, alpha=0.3)
                if depth_col:
                    ax1.invert_yaxis()  # Depth increases downward
                
                # Plot 2: Thomsen parameters vs Depth
                ax2 = axes[0, 1]
                if 'anisotropy_epsilon' in df_aniso.columns:
                    ax2.plot(df_aniso['anisotropy_epsilon'], y_vals, label='ε (%)', linewidth=2, color='orange')
                if 'anisotropy_delta' in df_aniso.columns:
                    ax2.plot(df_aniso['anisotropy_delta'], y_vals, label='δ (%)', linewidth=2, color='purple')
                ax2.set_xlabel('Anisotropy (%)')
                ax2.set_ylabel(y_label)
                ax2.set_title('Thomsen Parameters')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                if depth_col:
                    ax2.invert_yaxis()
                
                # Plot 3: VP variation vs Depth
                ax3 = axes[1, 0]
                if 'VP_variation' in df_aniso.columns:
                    ax3.plot(df_aniso['VP_variation'], y_vals, linewidth=2, color='darkred')
                    ax3.axvline(x=0, color='k', linestyle='--', alpha=0.3)
                if 'VP_0_Backus_Aniso' in df_aniso.columns:
                    ax3.plot(df_aniso['VP_0_Backus_Aniso'], y_vals, linewidth=3, 
                            color='darkorange', linestyle='--', alpha=0.8, label='Backus Anisotropy')
                ax3.set_xlabel('VP Variation (%)')
                ax3.set_ylabel(y_label)
                ax3.set_title('VP Variation: (VP90 - VP0) / VP0 * 100')
                ax3.grid(True, alpha=0.3)
                if depth_col:
                    ax3.invert_yaxis()
                
                # Plot 4: VP(45) vs VP(0) with Backus
                ax4 = axes[1, 1]
                ax4.scatter(df_aniso['VP_0'], df_aniso['VP_45'], alpha=0.5, s=10, c='green', label='Original')
                if 'VP_0_Backus' in df_aniso.columns and 'VP_45_Backus' in df_aniso.columns:
                    ax4.scatter(df_aniso['VP_0_Backus'], df_aniso['VP_45_Backus'], 
                              alpha=0.8, s=30, c='darkgreen', marker='s', label='Backus Average')
                ax4.plot([df_aniso['VP_0'].min(), df_aniso['VP_0'].max()],
                        [df_aniso['VP_0'].min(), df_aniso['VP_0'].max()], 'k--', alpha=0.5)
                ax4.set_xlabel('VP(0) (m/s)')
                ax4.set_ylabel('VP(45) (m/s)')
                ax4.set_title('VP(45) vs VP(0)')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig_aniso)
                plt.close()
                
                # Machine Learning Analysis
                if 'run_ml' in locals() and run_ml:
                    with st.spinner("🚀 Training Machine Learning models..."):
                        st.subheader("🤖 Machine Learning Results")
                        
                        # Prepare data
                        X_scaled, y_dict, scaler, features = prepare_ml_data(df_aniso)
                        targets = list(y_dict.keys())
                        target_names = {
                            'VP_0': 'VP(0)',
                            'VP_45': 'VP(45)',
                            'VP_90': 'VP(90)'
                        }
                        
                        if len(targets) == 0:
                            st.error("No target variables available for ML training")
                        else:
                            # Train models
                            models, results, X_train, X_test, y_train, y_test = train_ml_models(
                                X_scaled, y_dict, test_size=test_size
                            )
                            
                            # Results table
                            df_results = pd.DataFrame(results).T.round(3)
                            st.dataframe(df_results.style.background_gradient(cmap='RdYlGn', subset=['R2']))
                            
                            # Best model summary
                            st.subheader("📊 Best Model Summary")
                            best_models = {}
                            for target in targets:
                                best = min(['RF', 'PNN', 'CNN'], 
                                          key=lambda x: results[f'{x}_{target}']['RMSE'])
                                best_models[target] = best
                            
                            cols = st.columns(len(targets))
                            for idx, target in enumerate(targets):
                                with cols[idx]:
                                    model = best_models[target]
                                    st.metric(
                                        f"{target_names[target]}",
                                        model,
                                        f"RMSE: {results[f'{model}_{target}']['RMSE']:.1f} m/s"
                                    )
                            
                            # Generate and display plots with Backus results
                            st.subheader("📈 ML Model Comparison Plots with Backus Averaging")
                            plots = plot_ml_results_with_backus(
                                models, results, X_test, y_test,
                                targets, list(target_names.values()),
                                df_aniso if 'use_backus' in locals() and use_backus else None
                            )
                            
                            for name, fig in plots.items():
                                with st.expander(f"Figure: {name.replace('_', ' ').title()}"):
                                    st.pyplot(fig)
                                    plt.close()
                            
                            # Make predictions on full dataset
                            st.subheader("📥 Predictions on Full Dataset")
                            
                            pred_df = df_aniso.copy()
                            for target in targets:
                                for model_name in ['RF', 'PNN', 'CNN']:
                                    if model_name in models:
                                        if model_name == 'RF':
                                            pred = models['RF'][target].predict(X_scaled)
                                        elif model_name == 'PNN':
                                            pred = models['PNN'][target].predict(X_scaled)
                                        else:
                                            if TENSORFLOW_AVAILABLE:
                                                X_cnn = X_scaled.reshape(-1, X_scaled.shape[1], 1)
                                                pred = models['CNN'][target].predict(X_cnn, verbose=0).flatten()
                                            else:
                                                pred = models['CNN'][target].predict(X_scaled)
                                        pred_df[f'{model_name}_pred_{target}'] = pred
                            
                            # Show prediction summary
                            st.dataframe(pred_df[['VP_0', 'VP_45', 'VP_90'] + 
                                                [col for col in pred_df.columns if '_pred_' in col]].head(20))
                            
                            # Download buttons
                            col1, col2 = st.columns(2)
                            with col1:
                                csv_results = df_results.to_csv()
                                st.download_button(
                                    label="📊 Download Results CSV",
                                    data=csv_results,
                                    file_name="ml_results.csv",
                                    mime="text/csv"
                                )
                            with col2:
                                csv_pred = pred_df.to_csv()
                                st.download_button(
                                    label="📥 Download Predictions CSV",
                                    data=csv_pred,
                                    file_name="predictions_with_backus.csv",
                                    mime="text/csv"
                                )
            
            if mode == "Combined Analysis":
                st.header("🔗 Integrated Analysis")
                
                st.info("""
                **Integrated Analysis**: This mode combines rock physics modeling with anisotropic velocity prediction.
                
                The rock physics models provide the basis for understanding the rock properties, while the 
                machine learning models predict the anisotropic velocities (VP(0), VP(45), VP(90)) using 
                Thomsen parameters. Backus averaging is applied to scale up velocities to seismic scale.
                """)
                
                # Show summary of both analyses
                if 'logs' in locals() and 'df_aniso' in locals():
                    st.subheader("Integration Summary")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Rock Physics Model", model_choice)
                        st.metric("Samples Processed", len(logs))
                        if 'use_backus' in locals() and use_backus:
                            st.metric("Backus Window", f"{backus_window} samples")
                    with col2:
                        st.metric("Anisotropy", f"{df_aniso['anisotropy_epsilon'].mean():.2f}%")
                        st.metric("VP Variation", f"{df_aniso['VP_variation'].mean():.2f}%")
                        if 'VP_0_Backus' in df_aniso.columns:
                            backus_aniso = ((df_aniso['VP_90_Backus'].mean() - df_aniso['VP_0_Backus'].mean()) / 
                                          (df_aniso['VP_0_Backus'].mean() + 1e-10)) * 100
                            st.metric("Backus Anisotropy", f"{backus_aniso:.2f}%")
                    
                    # Cross-plot of rock physics vs anisotropy
                    fig_integrated, ax = plt.subplots(figsize=(10, 6))
                    
                    if 'IP_FRMMIX' in logs.columns and 'VP_0' in df_aniso.columns:
                        # Use the same index for both dataframes
                        common_idx = logs.index.intersection(df_aniso.index)
                        ip_mix = logs.loc[common_idx, 'IP_FRMMIX'].values
                        vp0 = df_aniso.loc[common_idx, 'VP_0'].values
                        vp45 = df_aniso.loc[common_idx, 'VP_45'].values
                        vp90 = df_aniso.loc[common_idx, 'VP_90'].values
                        
                        ax.scatter(ip_mix, vp0, label='VP(0)', alpha=0.5, s=10, c='blue')
                        ax.scatter(ip_mix, vp45, label='VP(45)', alpha=0.5, s=10, c='green')
                        ax.scatter(ip_mix, vp90, label='VP(90)', alpha=0.5, s=10, c='red')
                        
                        # Add Backus averaged points if available
                        if 'VP_0_Backus' in df_aniso.columns:
                            vp0_backus = df_aniso.loc[common_idx, 'VP_0_Backus'].values
                            vp45_backus = df_aniso.loc[common_idx, 'VP_45_Backus'].values
                            vp90_backus = df_aniso.loc[common_idx, 'VP_90_Backus'].values
                            ax.scatter(ip_mix, vp0_backus, label='Backus VP(0)', alpha=0.8, s=30, 
                                      c='darkblue', marker='s')
                            ax.scatter(ip_mix, vp45_backus, label='Backus VP(45)', alpha=0.8, s=30, 
                                      c='darkgreen', marker='s')
                            ax.scatter(ip_mix, vp90_backus, label='Backus VP(90)', alpha=0.8, s=30, 
                                      c='darkred', marker='s')
                        
                        ax.set_xlabel('IP (Mixed Fluid) (m/s*g/cc)')
                        ax.set_ylabel('Velocity (m/s)')
                        ax.set_title('Rock Physics IP vs Anisotropic Velocities with Backus Averaging')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        st.pyplot(fig_integrated)
                        plt.close()
        
        except Exception as e:
            st.error(f"❌ Error processing data: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    else:
        st.info("📤 Please upload a CSV or LAS file to begin analysis.")
        st.markdown("""
        ### Supported File Formats:
        - **CSV**: Comma-separated values with well log data
        - **LAS**: Log ASCII Standard format
        
        ### Required Columns:
        - DEPTH, VP, VS, RHO, VSH, SW, PHI (for rock physics)
        - VP, epsilon, delta (for anisotropy)
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Enhanced Rock Physics & AVO Tool with Anisotropy** | 
    Built with Streamlit
    
    **Features**: 
    - Rock Physics Models (Gassmann, Critical Porosity, Hertz-Mindlin, Dvorkin-Nur, Raymer-Hunt)
    - AVO Analysis
    - Wedge Modeling
    - Anisotropy (VP(0), VP(45), VP(90))
    - Backus Averaging for Seismic Scale-up
    - Machine Learning (RF, PNN, CNN)
    """)

if __name__ == "__main__":
    main()
