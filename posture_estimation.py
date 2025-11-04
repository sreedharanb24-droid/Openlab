# app.py — Posture Haptic Feedback (polished UX)
# Run: streamlit run app.py

import io, base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---------- Page & Theme ----------
st.set_page_config(page_title="Posture Haptic Feedback", page_icon="🧭", layout="wide")
st.markdown("""
<style>
/* improve spacing */
.block-container {padding-top: 1rem; padding-bottom: 2rem;}
/* tighter metric cards */
.css-12w0qpk {padding: 0.5rem 0.75rem;}
</style>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
@st.cache_data
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

@st.cache_data
def generate_demo_imu(fs=100, T=300, seed=123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(0, T, 1/fs)
    N = t.size
    theta = 5 + 0.5*np.sin(2*np.pi*0.02*t) + 0.3*np.sin(2*np.pi*0.1*t)
    episodes = [(20,55,25), (90,130,30), (170,200,22), (230,260,28), (275,290,18)]
    def apply(theta, s, e, target):
        fs_local = fs
        i0, i1 = int(s*fs_local), int(e*fs_local)
        L = i1 - i0 + 1
        ramp = int(3*fs_local)
        pre = np.mean(theta[max(0,i0-2*fs_local):i0]) if i0>0 else theta[0]
        up = np.linspace(pre, target, ramp, endpoint=False)
        hold = np.full(max(1, L-2*ramp), target)
        down = np.linspace(target, pre, ramp)
        seg = np.concatenate([up,hold,down])[:L]
        theta[i0:i0+L] = seg + 0.6*np.sin(2*np.pi*0.05*np.arange(L)/fs_local)
    for s,e,tg in episodes: apply(theta, s,e,tg)
    theta += 0.1*np.cumsum(rng.normal(scale=0.001, size=N))
    theta += rng.normal(scale=0.15, size=N)
    theta_dot = np.gradient(theta, 1/fs)
    rad = np.deg2rad(theta)
    ax = -np.sin(rad) + 0.02*rng.normal(size=N)
    ay = 0.02*rng.normal(size=N)
    az = np.cos(rad) + 0.02*rng.normal(size=N)
    gx = 0.2*rng.normal(size=N)
    gy = theta_dot + 0.5*rng.normal(size=N)
    gz = 0.2*rng.normal(size=N)
    return pd.DataFrame({
        "t_s": t, "ax_g": ax, "ay_g": ay, "az_g": az,
        "gx_dps": gx, "gy_dps": gy, "gz_dps": gz
    })

def complementary_pitch(df, alpha: float):
    t = df["t_s"].to_numpy()
    ax = df["ax_g"].to_numpy(); ay = df["ay_g"].to_numpy(); az = df["az_g"].to_numpy()
    gy = df["gy_dps"].to_numpy()
    fs = int(np.round(1/np.mean(np.diff(t))))
    pitch_acc = np.degrees(np.arctan2(-ax, np.sqrt(ay**2 + az**2)))
    pitch = np.zeros_like(t); pitch[0] = pitch_acc[0]
    for i in range(1, t.size):
        dt = t[i]-t[i-1]
        pred = pitch[i-1] + gy[i]*dt
        pitch[i] = alpha*pred + (1-alpha)*pitch_acc[i]
    return t, pitch, fs

def moving_average(x, fs, win_s):
    win = max(1, int(round(win_s*fs)))
    return np.convolve(x, np.ones(win)/win, mode="same")

def controller(angle_filt, fs, safe, hyst, min_bad_s, min_cue_s):
    upper, lower = safe + hyst, safe - hyst
    min_bad_n, min_cue_n = int(round(min_bad_s*fs)), int(round(min_cue_s*fs))
    N = angle_filt.size
    vibrate = np.zeros(N, dtype=bool)
    bad = cue = 0; in_cue = False
    for i in range(N):
        a = abs(angle_filt[i])
        if not in_cue:
            bad = bad + 1 if a > upper else 0 if a < lower else max(bad-1,0)
            if bad >= min_bad_n:
                in_cue = True; cue = min_cue_n; vibrate[i]=True; bad=0
        else:
            if cue>0: cue-=1; vibrate[i]=True
            else:
                if a < lower: in_cue=False; vibrate[i]=False
                else: vibrate[i]=True
    return vibrate, upper, lower

def metrics(angle_filt, vibrate, safe):
    pct_safe = 100*np.mean(np.abs(angle_filt) <= safe)
    rms = float(np.sqrt(np.mean(angle_filt**2)))
    n_cues = int(np.sum(np.diff(np.concatenate([[0], vibrate.astype(int)]))==1))
    return pct_safe, rms, n_cues

def fig_to_png_bytes(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight", dpi=200); buf.seek(0)
    return buf.getvalue()

# ---------- Header ----------
st.title("⛑ Posture Haptic Feedback Simulator")
st.subheader("IMU → Orientation (Pitch) → Dead-Zone + Hysteresis → Vibrotactile Cue")

with st.sidebar:
    st.header("Configure")
    st.caption("Upload data or use a built-in demo. Then tune parameters below.")

    source = st.radio("Data source", ["Upload CSV", "Use built-in demo"], index=1)
    uploaded = st.file_uploader("Upload CSV", type=["csv"], help="Either 7-col IMU or 2-col angle CSV")
    if st.button("Download demo CSV"):
        demo_csv = generate_demo_imu().to_csv(index=False).encode("utf-8")
        st.download_button("Save demo IMU CSV", demo_csv, file_name="demo_imu.csv", mime="text/csv")

    st.divider()
    preset = st.selectbox("Presets", [
        "Maternal gentle (default)",
        "Office strict",
        "Rehab tolerant",
        "Custom"
    ])

    # defaults per preset
    if preset == "Maternal gentle (default)":
        _safe,_hyst,_bad,_cue,_alpha,_smooth = 18, 2, 2.0, 1.0, 0.98, 0.6
    elif preset == "Office strict":
        _safe,_hyst,_bad,_cue,_alpha,_smooth = 12, 2, 1.0, 0.8, 0.985, 0.4
    elif preset == "Rehab tolerant":
        _safe,_hyst,_bad,_cue,_alpha,_smooth = 20, 3, 2.5, 1.2, 0.97, 0.8
    else:
        _safe,_hyst,_bad,_cue,_alpha,_smooth = 15, 2, 2.0, 1.0, 0.98, 0.5

    safe = st.slider("Safe limit (°)", 5, 30, _safe)
    hyst = st.slider("Hysteresis (°)", 0, 5, _hyst)
    bad_s = st.slider("Debounce: min bad duration (s)", 0.0, 5.0, _bad, 0.1)
    cue_s = st.slider("Min cue on-time (s)", 0.0, 3.0, _cue, 0.1)
    alpha = st.slider("Complementary filter α", 0.90, 0.999, _alpha, 0.001, help="Higher = trust gyro more")
    smooth_s = st.slider("Smoothing window (s)", 0.1, 2.0, _smooth, 0.1)
    show_comp = st.checkbox("Show complementary (unfiltered) trace", True)
    autoscale_vibe = st.checkbox("Auto-scale cue line height", True)

st.info("**Tip:** You can upload either a **7-column raw IMU CSV** (`t_s, ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps`) "
        "or a **2-column angle CSV** (`t_s, angle_deg`). The app auto-detects the format.")

# ---------- Load data ----------
if source == "Use built-in demo":
    df = generate_demo_imu()
else:
    if uploaded is None:
        st.warning("Upload a CSV or select **Use built-in demo**.", icon="📄")
        st.stop()
    try:
        df = load_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

# Detect type
if {"t_s","angle_deg"}.issubset(df.columns):
    mode = "angle"
    t = df["t_s"].to_numpy()
    angle_comp = df["angle_deg"].to_numpy()
    fs = int(np.round(1/np.mean(np.diff(t))))
elif {"t_s","ax_g","ay_g","az_g","gx_dps","gy_dps","gz_dps"}.issubset(df.columns):
    mode = "imu"
    t, angle_comp, fs = complementary_pitch(df, alpha)
else:
    st.error("CSV columns do not match supported formats. See help above.")
    st.stop()

# ---------- Process ----------
angle_filt = moving_average(angle_comp, fs, smooth_s)
vibrate, upper, lower = controller(angle_filt, fs, safe, hyst, bad_s, cue_s)
pct_safe, rms_deg, n_cues = metrics(angle_filt, vibrate, safe)

# ---------- UI: Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(["📈 Plots", "📊 Metrics", "📁 Data", "🛠 Logs"])

with tab1:
    colA, colB = st.columns([2.2, 1.0])
    with colA:
        fig, ax = plt.subplots(figsize=(12,4))
        if show_comp: ax.plot(t, angle_comp, label="Pitch (complementary)", color=(0.9,0.6,0))
        ax.plot(t, angle_filt, "b", lw=1.2, label="Pitch (filtered)")
        ax.axhline(upper, ls="--", c="r"); ax.text(t[0], upper, "Upper", va="bottom", color="r")
        ax.axhline(lower, ls="--", c="r"); ax.text(t[0], lower, "Lower", va="top", color="r")
        scale = (max(upper, float(np.max(np.abs(angle_filt))))*0.6) if autoscale_vibe else safe
        ax.step(t, vibrate.astype(float)*scale, where="post", lw=1.2, label="Vibrate (scaled)", c="k")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Angle (deg)")
        ax.set_title("Estimated Pitch Angle & Vibrotactile Cue"); ax.grid(True); ax.legend(loc="best")
        st.pyplot(fig)
        png1 = fig_to_png_bytes(fig)
    with colB:
        fig2, ax2 = plt.subplots(figsize=(5,4))
        ax2.bar([0,1,2], [pct_safe, rms_deg, n_cues])
        ax2.set_xticks([0,1,2]); ax2.set_xticklabels(["% Time Safe","RMS Angle (deg)","Cues"])
        ax2.set_ylim(bottom=0); ax2.set_title("Summary"); ax2.grid(True, axis="y")
        st.pyplot(fig2)
        png2 = fig_to_png_bytes(fig2)

    st.download_button("⬇️ Download plots (timeline).png", data=png1, file_name="timeline.png", mime="image/png")
    st.download_button("⬇️ Download plots (summary).png", data=png2, file_name="summary.png", mime="image/png")

with tab2:
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("% Time Safe", f"{pct_safe:.1f}%")
    c2.metric("RMS Angle", f"{rms_deg:.2f}°")
    c3.metric("# Cues", f"{n_cues}")
    c4.metric("fs", f"{fs} Hz")
    st.caption("Higher % Time Safe and lower RMS Angle indicate better posture. Cues = how many corrections were needed.")

with tab3:
    st.write("**Processed timeline preview**")
    proc = pd.DataFrame({"t": t, "pitch_deg": angle_comp, "pitch_filt": angle_filt, "vibrate": vibrate.astype(int)})
    st.dataframe(proc.head(500), use_container_width=True)
    st.download_button("⬇️ Download processed timeline CSV", proc.to_csv(index=False).encode("utf-8"),
                       file_name="processed_posture_timeline.csv", mime="text/csv")

with tab4:
    st.json({
        "mode_detected": "Angle-only" if mode=="angle" else "Raw IMU",
        "fs_hz": fs,
        "params": {
            "alpha": alpha, "smooth_window_s": smooth_s,
            "safe_limit_deg": safe, "hysteresis_deg": hyst,
            "min_bad_s": bad_s, "min_cue_s": cue_s
        }
    })
    st.caption("Use this block for your report’s Method section.")

st.success("Done. Adjust parameters in the sidebar to explore behavior. Use **Presets** for quick scenarios.")
