# app.py — Posture Haptic Feedback (Elite UI Edition)
# Run: streamlit run app.py

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# =========================
# Page Setup + Global Style
# =========================
st.set_page_config(page_title="Posture Haptic Feedback — Elite UI", page_icon="🧭", layout="wide")

# Premium palette (accessible on light & dark)
PALETTE = {
    "bg": "#0B1220",           # deep navy (used in accents/shadows)
    "ink": "#101418",
    "ink_muted": "#5B6270",
    "brand": "#6C9EF8",        # bright blue
    "brand_alt": "#8E7BF6",    # blue-violet
    "accent": "#42D392",       # jade
    "warn": "#F26D6D",         # coral
    "gold": "#F0C674",         # warm gold
    "surface": "#F9FAFB",
    "surface_alt": "#EEF2F7",
}

# Inject CSS for refined visuals
st.markdown(f"""
<style>
:root {{
  --ink: {PALETTE["ink"]};
  --ink-muted: {PALETTE["ink_muted"]};
  --brand: {PALETTE["brand"]};
  --brand-alt: {PALETTE["brand_alt"]};
  --accent: {PALETTE["accent"]};
  --warn: {PALETTE["warn"]};
  --gold: {PALETTE["gold"]};
}}
/* card-like feel */
.block-container {{
  padding-top: 1.2rem;
  padding-bottom: 2rem;
}}
/* headings */
h1,h2,h3,h4 {{
  letter-spacing: .2px;
}}
/* subtle panel look for sidebar */
section[data-testid="stSidebar"] > div {{
  background: linear-gradient(180deg, #ffffff 0%, #f7f9fc 100%);
  border-right: 1px solid #e8eef6;
}}
/* metric cards */
[data-testid="stMetric"] {{
  background: #ffffff;
  border: 1px solid #ecf0f6;
  box-shadow: 0 6px 18px rgba(16,20,24,0.06);
  border-radius: 12px;
  padding: .75rem 1rem;
}}
/* download buttons */
.stDownloadButton button {{
  border-radius: 10px !important;
  border: 1px solid #e2e8f0 !important;
  background: linear-gradient(180deg, #ffffff 0%, #f6f8fc 100%) !important;
  color: var(--ink) !important;
}}
/* emphasis chips */
.badge {{
  display:inline-block; padding:.2rem .5rem; font-size:.82rem;
  border-radius:999px; background:rgba(108,158,248,.12); color:var(--brand);
  border:1px solid rgba(108,158,248,.25);
}}
.small-muted {{ color: var(--ink-muted); font-size:.9rem; }}
</style>
""", unsafe_allow_html=True)

# =========================
# Helper functions
# =========================
@st.cache_data
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

@st.cache_data
def generate_demo_imu(fs=100, T=300, seed=123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(0, T, 1/fs)
    N = t.size
    # baseline pitch
    theta = 5 + 0.5*np.sin(2*np.pi*0.02*t) + 0.3*np.sin(2*np.pi*0.1*t)
    episodes = [(20,55,25), (90,130,30), (170,200,22), (230,260,28), (275,290,18)]
    def apply(theta, s, e, target):
        i0, i1 = int(s*fs), int(e*fs)
        L = i1 - i0 + 1
        ramp = int(3*fs)
        pre = np.mean(theta[max(0,i0-2*fs):i0]) if i0>0 else theta[0]
        up = np.linspace(pre, target, ramp, endpoint=False)
        hold = np.full(max(1, L-2*ramp), target)
        down = np.linspace(target, pre, ramp)
        seg = np.concatenate([up,hold,down])[:L]
        theta[i0:i0+L] = seg + 0.6*np.sin(2*np.pi*0.05*np.arange(L)/fs)
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

def compute_metrics(angle_filt, vibrate, safe):
    pct_safe = 100*np.mean(np.abs(angle_filt) <= safe)
    rms = float(np.sqrt(np.mean(angle_filt**2)))
    n_cues = int(np.sum(np.diff(np.concatenate([[0], vibrate.astype(int)]))==1))
    return pct_safe, rms, n_cues

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

# =========================
# Header
# =========================
st.title("🧭 Posture Haptic Feedback")
st.markdown(
    f"""
    <span class="badge">Elite UI</span> <span class="small-muted">IMU → Orientation (Pitch) → Dead-Zone + Hysteresis → Vibrotactile Cue → Metrics</span>
    """,
    unsafe_allow_html=True
)

# =========================
# Sidebar Controls
# =========================
with st.sidebar:
    st.header("Configure")
    source = st.radio("Data source", ["Use built-in demo", "Upload CSV"], index=0)
    uploaded = st.file_uploader("Upload CSV", type=["csv"], help="7-col IMU or 2-col angle CSV")
    st.caption("Supported: IMU (t_s, ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps) or Angle (t_s, angle_deg)")

    st.divider()
    preset = st.selectbox("Presets", ["Maternal gentle (default)","Office strict","Rehab tolerant","Custom"], index=0)

    # Preset defaults
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
    min_bad_s = st.slider("Debounce: min bad duration (s)", 0.0, 5.0, _bad, 0.1)
    min_cue_s = st.slider("Min cue on-time (s)", 0.0, 3.0, _cue, 0.1)
    alpha = st.slider("Complementary filter α", 0.90, 0.999, _alpha, 0.001)
    smooth_s = st.slider("Smoothing window (s)", 0.1, 2.0, _smooth, 0.1)

    show_comp = st.checkbox("Show complementary trace", True)
    autoscale_vibe = st.checkbox("Auto-scale cue height", True)

    st.divider()
    if st.button("Download demo IMU CSV"):
        demo = generate_demo_imu().to_csv(index=False).encode("utf-8")
        st.download_button("Save demo_imu.csv", demo, file_name="demo_imu.csv", mime="text/csv")

# =========================
# Load Data
# =========================
if source == "Use built-in demo":
    df = generate_demo_imu()
else:
    if uploaded is None:
        st.warning("Upload a CSV or switch to **Use built-in demo**.", icon="📄")
        st.stop()
    df = load_csv(uploaded)

# detect input mode
if {"t_s","angle_deg"}.issubset(df.columns):
    mode = "angle"
    t = df["t_s"].to_numpy()
    angle_comp = df["angle_deg"].to_numpy()
    fs = int(np.round(1/np.mean(np.diff(t))))
elif {"t_s","ax_g","ay_g","az_g","gx_dps","gy_dps","gz_dps"}.issubset(df.columns):
    mode = "imu"
    t, angle_comp, fs = complementary_pitch(df, alpha)
else:
    st.error("CSV columns don’t match supported formats.")
    st.stop()

# =========================
# Process
# =========================
angle_filt = moving_average(angle_comp, fs, smooth_s)
vibrate, upper, lower = controller(angle_filt, fs, safe, hyst, min_bad_s, min_cue_s)
pct_safe, rms_deg, n_cues = compute_metrics(angle_filt, vibrate, safe)

# =========================
# Matplotlib THEME for elite palette
# =========================
plt.rcParams.update({
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.edgecolor": "#D7DFEA",
    "axes.labelcolor": PALETTE["ink"],
    "xtick.color": PALETTE["ink_muted"],
    "ytick.color": PALETTE["ink_muted"],
    "grid.color": "#E8EEF6",
    "grid.linestyle": "-",
    "grid.alpha": 0.8,
    "text.color": PALETTE["ink"],
})

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["📈 Plots", "📊 Metrics", "📁 Data", "🛠 Logs"])

with tab1:
    cA, cB = st.columns([2.2, 1])
    with cA:
        fig, ax = plt.subplots(figsize=(12,4))
        if show_comp:
            ax.plot(t, angle_comp, label="Pitch (complementary)", color=PALETTE["gold"], lw=1.2)
        ax.plot(t, angle_filt, label="Pitch (filtered)", color=PALETTE["brand"], lw=2)
        ax.axhline(upper, ls="--", c=PALETTE["warn"]); ax.text(t[0], upper, "Upper", va="bottom", color=PALETTE["warn"])
        ax.axhline(lower, ls="--", c=PALETTE["warn"]); ax.text(t[0], lower, "Lower", va="top", color=PALETTE["warn"])
        height = (max(upper, float(np.max(np.abs(angle_filt))))*0.6) if autoscale_vibe else safe
        ax.step(t, vibrate.astype(float)*height, where="post", color=PALETTE["ink"], lw=2, label="Vibrate (scaled)")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Angle (deg)")
        ax.set_title("Estimated Pitch Angle & Vibrotactile Cue")
        ax.grid(True); ax.legend(loc="best", frameon=False)
        st.pyplot(fig)
        png1 = fig_to_bytes(fig)

    with cB:
        fig2, ax2 = plt.subplots(figsize=(5,4))
        bars = ax2.bar([0,1,2], [pct_safe, rms_deg, n_cues], color=[PALETTE["accent"], PALETTE["brand_alt"], PALETTE["gold"]])
        ax2.set_xticks([0,1,2])
        ax2.set_xticklabels(["% Time Safe","RMS Angle (deg)","Cues"])
        ax2.set_ylim(bottom=0)
        ax2.set_title("Summary")
        ax2.grid(True, axis="y")
        for b in bars:
            ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f"{b.get_height():.1f}" if b!=bars[2] else f"{int(b.get_height())}",
                     ha="center", va="bottom", color=PALETTE["ink_muted"], fontsize=9)
        st.pyplot(fig2)
        png2 = fig_to_bytes(fig2)

    st.download_button("⬇️ timeline.png", data=png1, file_name="timeline.png", mime="image/png")
    st.download_button("⬇️ summary.png",  data=png2, file_name="summary.png",  mime="image/png")

with tab2:
    a,b,c,d = st.columns(4)
    a.metric("% Time Safe", f"{pct_safe:.1f}%")
    b.metric("RMS Angle", f"{rms_deg:.2f}°")
    c.metric("# Cues", f"{n_cues}")
    d.metric("fs", f"{fs} Hz")
    st.caption("Higher % Time Safe and lower RMS Angle indicate better posture. Cues = number of feedback activations.")

with tab3:
    st.write("**Processed timeline (preview)**")
    proc = pd.DataFrame({"t": t, "pitch_deg": angle_comp, "pitch_filt": angle_filt, "vibrate": vibrate.astype(int)})
    st.dataframe(proc.head(1000), use_container_width=True)
    st.download_button("⬇️ processed_posture_timeline.csv",
                       proc.to_csv(index=False).encode("utf-8"),
                       file_name="processed_posture_timeline.csv", mime="text/csv")

with tab4:
    st.json({
        "input_mode": "Angle" if mode=="angle" else "Raw IMU",
        "fs_hz": fs,
        "params": {
            "alpha": float(alpha),
            "smoothing_window_s": float(smooth_s),
            "safe_limit_deg": int(safe),
            "hysteresis_deg": int(hyst),
            "min_bad_s": float(min_bad_s),
            "min_cue_s": float(min_cue_s),
        }
    })
    st.caption("Use this for your Method/Settings section.")

st.success("Ready. Tweak sidebar parameters or switch presets for different ergonomics.")
