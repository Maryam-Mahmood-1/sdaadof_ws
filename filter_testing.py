import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import math

# ---------- One Euro Filter Classes ----------

class LowPassFilter:
    def __init__(self, alpha):
        self.set_alpha(alpha)
        self.y = self.s = None

    def set_alpha(self, alpha):
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha should be in (0.0, 1.0]")
        self.alpha = float(alpha)

    def __call__(self, value, alpha=None):
        if alpha is not None:
            self.set_alpha(alpha)
        s = value if self.y is None else self.alpha * value + (1.0 - self.alpha) * self.s
        self.y = value
        self.s = s
        return s

    def reset(self):
        self.y = self.s = None

class OneEuroFilter:
    def __init__(self, freq, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.freq = float(freq)
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self.x = LowPassFilter(self.alpha(self.mincutoff))
        self.dx = LowPassFilter(self.alpha(self.dcutoff))

    def alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x):
        dx_val = 0.0 if self.x.s is None else (x - self.x.s) * self.freq
        edx = self.dx(dx_val, alpha=self.alpha(self.dcutoff))
        cutoff = self.mincutoff + self.beta * abs(edx)
        cutoff = max(1e-3, cutoff)
        alpha = self.alpha(cutoff)
        alpha = min(1.0, max(1e-3, alpha))
        return self.x(x, alpha=alpha)

# ---------- Step Interpolator with Smoothstep ----------

class SmoothStepInterpolator:
    def __init__(self, freq):
        self.prev_value = None
        self.curr_value = None
        self.start_time = None
        self.duration = 1.0 / freq

    def smoothstep(self, x):
        return x * x * (3 - 2 * x)

    def __call__(self, value, t):
        if self.curr_value is None:
            self.prev_value = value
            self.curr_value = value
            self.start_time = t
            return value

        if value != self.curr_value:
            self.prev_value = self.curr_value
            self.curr_value = value
            self.start_time = t

        alpha = min((t - self.start_time) / self.duration, 1.0)
        s = self.smoothstep(alpha)
        return self.prev_value + s * (self.curr_value - self.prev_value)

# ---------- Kalman Filter Class (1D constant velocity model) ----------

class KalmanFilter1D:
    def __init__(self, dt, process_variance=1e-4, measurement_variance=1e-2):
        self.dt = dt
        self.x = np.array([[0.0], [0.0]])  # state = [position, velocity]
        self.P = np.eye(2) * 1.0
        self.F = np.array([[1.0, dt], [0, 1.0]])
        self.H = np.array([[1.0, 0]])
        self.Q = process_variance * np.array([[dt**4/4, dt**3/2], [dt**3/2, dt**2]])
        self.R = np.array([[measurement_variance]])

    def update(self, z):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return self.x[0, 0]

# ---------- Main Code ----------

# Load data
df = pd.read_csv("/home/maryam-mahmood/filtering_experimentation_file.csv")

# Define columns
pos_col = "/joint_states/joint_7/position"
ref_col = "/velocity_arm_controller/controller_state/reference/positions[6]"

df[pos_col] = df[pos_col].fillna(method='ffill').fillna(0)
df[ref_col] = df[ref_col].fillna(method='ffill').fillna(0)

# Extract arrays
t = df["__time"].values if "__time" in df.columns else np.arange(len(df)) / 100.0
pos = df[pos_col].values
ref = df[ref_col].values

# Parameters
freq = 1000
dt = 1.0 / freq

# Interpolation
step_interp = SmoothStepInterpolator(freq=freq)
step_interp_filtered = [step_interp(pos[0], t[0])]
for i in range(1, len(pos)):
    step_interp_filtered.append(step_interp(pos[i], t[i]))

# Filters
euro = OneEuroFilter(freq=freq, mincutoff=0.0081, beta=3.6, dcutoff=0.0081)
euro_from_interp = [euro(p) for p in step_interp_filtered]

kf = KalmanFilter1D(dt=dt, process_variance=1e-2, measurement_variance=1e-6)
kalman_filtered = [kf.update(z) for z in pos]

savgol_filtered = savgol_filter(pos, window_length=9, polyorder=2)
gaussian_filtered = gaussian_filter1d(pos, sigma=2)

# Hybrids
hybrid_kalman_euro = 0.5 * np.array(kalman_filtered) + 0.5 * np.array(euro_from_interp)
hybrid_keg = (np.array(kalman_filtered) + np.array(euro_from_interp) + gaussian_filtered) / 3.0

# Store
for name, data in [
    ("step_interp", step_interp_filtered),
    ("euro_filtered", euro_from_interp),
    ("kalman_filtered", kalman_filtered),
    ("savgol_filtered", savgol_filtered),
    ("gaussian_filtered", gaussian_filtered),
    ("hybrid_kalman_euro", hybrid_kalman_euro),
    ("hybrid_keg", hybrid_keg)
]:
    df[name] = data

# Differences
df["diff_kalman"] = ref - df["kalman_filtered"]
df["diff_euro"] = ref - df["euro_filtered"]
df["diff_unfiltered"] = ref - pos
df["diff_hybrid_keg"] = ref - df["hybrid_keg"]

# Save
df.to_csv("filtered_motor_readings_kalman_hybrid.csv", index=False)
print("✅ Saved: filtered_motor_readings_kalman_hybrid.csv")

# Plotting
plt.figure(figsize=(14, 12))

plt.subplot(2, 1, 1)
plt.plot(t, pos, label="Original Position", alpha=0.4)
plt.plot(t, ref, label="Reference Position", linestyle='--', alpha=0.6)
plt.plot(t, df["euro_filtered"], label="1-Euro", linewidth=1.5)
plt.plot(t, df["kalman_filtered"], label="Kalman", linewidth=1.5)
plt.plot(t, df["hybrid_keg"], label="Hybrid Kalman+Euro+Gaussian", linestyle=':')
plt.title("Filtered Position Comparison")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, df["diff_unfiltered"], label="Unfiltered - Ref", alpha=0.5)
plt.plot(t, df["diff_euro"], label="1-Euro - Ref")
plt.plot(t, df["diff_kalman"], label="Kalman - Ref")
plt.plot(t, df["diff_hybrid_keg"], label="Hybrid KEG - Ref")
plt.title("Signed Differences")
plt.legend()

plt.tight_layout()
plt.savefig("filtering_comparison_plot_kalman_hybrid.png")
plt.show()
print("📈 Saved: filtering_comparison_plot_kalman_hybrid.png")