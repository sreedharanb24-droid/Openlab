import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Posture Haptic Feedback Simulator", layout="wide")

st.title("Posture Haptic Feedback — Streamlit App")
st.caption("IMU → orientation (pitch) → dead-zone + hysteresis → vibrotactile cue → metrics")

with st.expander("📄 Input format help", expanded=False):
    st.write("""
    **Option A — Raw IMU CSV (7 columns):**
    - `t_s, ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps`  
    This app computes pitch from accel+gyro (complementary filter).

    **Option B — Angle-only CSV (2 columns):**
    - `t_s, angle_deg`  
    This app uses the angle directly (skips IMU fusion).
    """)

uploaded = st.file_uploader("Upload CSV", type=["csv"])

col_params1, col_params2, col_params3 = st.columns(3)

with col_params1:
    mode = st.radio("Input mode", ["Auto-detect", "Raw IMU (7 cols)", "Angle-only (2 cols)"], index=0)
    alpha = st.slider("Complementary filter α (gyro weight)", 0.85, 0.999, 0.98, 0.001)
    smooth_win_s = st.slider("Smoothing window (seconds)", 0.1, 2.0, 0.5, 0.1)

with col_params2:
    safe_limit = st.slider("Safe posture limit (°)", 5, 30, 15, 1)
    hyst = st.slider("Hysteresis width (°)", 0, 5, 2, 1)
    min_bad_s = st.slider("Debounce: min bad duration (s)", 0.0, 5.0, 2.0, 0.1)

with col_params3:
    min_cue_s = st.slider("Minimum cue ON time (s)", 0.0, 3.0, 1.0, 0.1)
    show_complementary = st.checkbox("Show complementary (unfiltered) trace", True)
    autoscale_vibe = st.checkbox("Autoscale cue line height", True, help="Scale the black cue line for visibility")

def complementary_pitch_from_imu(df):
    """
    df: columns = t_s, ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps
    returns: t, pitch_deg (complementary), fs
    """
    t = df["t_s"].to_numpy()
    ax = df["ax_g"].to_numpy()
    ay = df["ay_g"].to_numpy()
    az = df["az_g"].to_numpy()
    gy = df["gy_dps"].to_numpy()  # pitch rate deg/s

    fs = int(np.round(1 / np.mean(np.diff(t))))
    # accel tilt (deg)
    pitch_acc = np.degrees(np.arctan2(-ax, np.sqrt(ay**2 + az**2)))
    # complementary filter
    pitch = np.zeros_like(t)
    pitch[0] = pitch_acc[0]
    for i in range(1, t.size):
        dt = t[i] - t[i-1]
        pred = pitch[i-1] + gy[i] * dt
        pitch[i] = alpha * pred + (1 - alpha) * pitch_acc[i]
    return t, pitch, fs

def smooth_moving_average(x, fs, win_s):
    win = max(1, int(round(win_s * fs)))
    kernel = np.ones(win) / win
    # 'same' style conv
    y = np.convolve(x, kernel, mode="same")
    return y

def vibrotactile_controller(angle_deg, fs, safe_limit, hyst, min_bad_s, min_cue_s):
    """
    angle_deg: 1D array
    returns: vibrate (bool array), and some counters are internal
    """
    upper = safe_limit + hyst
    lower = safe_limit - hyst
    min_bad_n = int(round(min_bad_s * fs))
    min_cue_n = int(round(min_cue_s * fs))

    N = angle_deg.size
    vibrate = np.zeros(N, dtype=bool)
    bad_cnt = 0
    cue_cnt = 0
    in_cue = False

    for i in range(N):
        a = abs(angle_deg[i])
        if not in_cue:
            if a > upper:
                bad_cnt += 1
            elif a < lower:
                bad_cnt = 0
            else:
                bad_cnt = max(bad_cnt - 1, 0)
            if bad_cnt >= min_bad_n:
                in_cue = True
                cue_cnt = min_cue_n
                vibrate[i] = True
                bad_cnt = 0
        else:
            if cue_cnt > 0:
                cue_cnt -= 1
                vibrate[i] = True
            else:
                if a < lower:
                    in_cue = False
                    vibrate[i] = False
                else:
                    vibrate[i] = True
    return vibrate, upper, lower

def compute_metrics(angle_filt, vibrate, safe_limit):
    pct_time_safe = 100.0 * np.mean(np.abs(angle_filt) <= safe_limit)
    rms_deg = float(np.sqrt(np.mean(angle_filt**2)))
    n_cues = int(np.sum(np.diff(np.concatenate([[0], vibrate.astype(int)])) == 1))
    return pct_time_safe, rms_deg, n_cues

# Load data
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    st.info("Upload a CSV to get started (or use the dataset you generated earlier).")
    st.stop()

# Detect mode
if mode == "Auto-detect":
    if set(["t_s","angle_deg"]).issubset(df.columns):
        mode_effective = "Angle-only (2 cols)"
    elif set(["t_s","ax_g","ay_g","az_g","gx_dps","gy_dps","gz_dps"]).issubset(df.columns):
        mode_effective = "Raw IMU (7 cols)"
    else:
        st.error("Could not detect CSV type. Please ensure columns match one of the supported formats.")
        st.stop()
else:
    mode_effective = mode

st.write(f"**Detected / chosen input:** {mode_effective}")

# Build angle series
if mode_effective == "Angle-only (2 cols)":
    t = df["t_s"].to_numpy()
    angle_input = df["angle_deg"].to_numpy()
    fs = int(np.round(1 / np.mean(np.diff(t))))
    comp_angle = angle_input.copy()
else:
    t, comp_angle, fs = complementary_pitch_from_imu(df)

# Filter
angle_filt = smooth_moving_average(comp_angle, fs, smooth_win_s)

# Controller
vibrate, upper, lower = vibrotactile_controller(angle_filt, fs, safe_limit, hyst, min_bad_s, min_cue_s)

# Metrics
pct_time_safe, rms_deg, n_cues = compute_metrics(angle_filt, vibrate, safe_limit)

# --- Layout for outputs ---
g1, g2 = st.columns([2.2, 1.0])

with g1:
    st.subheader("Angle & Vibrotactile Cue")
    fig, ax = plt.subplots(figsize=(12, 4))
    if show_complementary:
        ax.plot(t, comp_angle, label="Pitch (complementary)", color=(0.9, 0.6, 0))
    ax.plot(t, angle_filt, "b", linewidth=1.2, label="Pitch (filtered)")
    ax.axhline(upper, linestyle="--", color="r"); ax.text(t[0], upper, "Upper", va="bottom", color="r")
    ax.axhline(lower, linestyle="--", color="r"); ax.text(t[0], lower, "Lower", va="top", color="r")
    scale = (max(upper, float(np.max(np.abs(angle_filt)))) * 0.6) if autoscale_vibe else safe_limit
    ax.step(t, vibrate.astype(float) * scale, where="post", linewidth=1.2, label="Vibrate (scaled)", color="k")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Angle (deg)")
    ax.set_title("Estimated Pitch Angle & Vibrotactile Cue")
    ax.grid(True); ax.legend(loc="best")
    st.pyplot(fig)

with g2:
    st.subheader("Summary Metrics")
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.bar([0,1,2], [pct_time_safe, rms_deg, n_cues])
    ax2.set_xticks([0,1,2])
    ax2.set_xticklabels(["% Time Safe","RMS Angle (deg)","Cues"])
    ax2.set_ylim(bottom=0)
    ax2.set_title("Summary")
    ax2.grid(True, axis="y")
    st.pyplot(fig2)

st.markdown("### Numbers")
c1, c2, c3, c4 = st.columns(4)
c1.metric("% Time Safe", f"{pct_time_safe:.1f}%")
c2.metric("RMS Angle", f"{rms_deg:.2f}°")
c3.metric("# Cues", f"{n_cues}")
c4.metric("fs", f"{fs} Hz")

# Download processed timeline
proc = pd.DataFrame({
    "t": t,
    "pitch_deg": comp_angle,
    "pitch_filt": angle_filt,
    "vibrate": vibrate.astype(int)
})
csv_bytes = proc.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download processed timeline (CSV)", data=csv_bytes, file_name="processed_posture_timeline.csv", mime="text/csv")

st.caption("Tip: use Angle-only mode for quick testing; use Raw IMU mode when you have 7-column sensor data.")
