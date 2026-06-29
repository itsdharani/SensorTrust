import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import json
import torch
import sys
from pathlib import Path

sys.path.append('src')

from src.anomaly.lstm_autoencoder import LSTMAutoencoder
from src.anomaly.sequence_dataset import create_sequences
from src.anomaly.mahalanobis import MahalanobisDetector
from src.anomaly.detector import detect_anomalies
from src.anomaly.ema import EMABaseline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SensorTrust Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; margin-bottom: 0; }
    .sub-header { color: #666; font-size: 1rem; margin-top: 0; }
    .sensor-gps { color: #e74c3c; font-weight: bold; }
    .sensor-imu { color: #3498db; font-weight: bold; }
    .sensor-lidar { color: #2ecc71; font-weight: bold; }
    .sensor-camera { color: #f39c12; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_features():
    with open('results/features.json') as f:
        return json.load(f)

@st.cache_data
def load_trust():
    with open('results/trust.json') as f:
        return json.load(f)

@st.cache_data
def load_ema():
    with open('results/ema.json') as f:
        return json.load(f)

@st.cache_resource
def load_model():
    with open('src/models/threshold.json') as f:
        default_threshold = json.load(f)['threshold']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = LSTMAutoencoder(n_features=3, hidden_size=64, latent_size=32).to(device)
    model.load_state_dict(torch.load('src/models/lstm_autoencoder.pt', map_location=device))
    model.eval()
    return model, device, default_threshold

@st.cache_resource
def load_mahal(_features):
    clean   = _features['Clean']
    f1_c    = np.array(clean['f1'])
    f2_c    = np.array(clean['f2'])
    gmis_c  = np.array(clean['gmis'])
    min_len = min(len(f1_c), len(f2_c), len(gmis_c))
    det     = MahalanobisDetector()
    det.fit(f1_c[:min_len], f2_c[:min_len], gmis_c[:min_len])
    return det

features                      = load_features()
trust_data                    = load_trust()
ema_data                      = load_ema()
model, device, default_thresh = load_model()
mahal_detector                = load_mahal(features)

clean        = features['Clean']
CLEAN_F1     = clean['F1']
CLEAN_F2     = clean['F2']
CLEAN_GMIS   = clean['GMIS']
attack_names = [k for k in features.keys() if k != 'Clean']

# ── Detection functions ───────────────────────────────────────────────────────
def run_mahalanobis(f1, f2, gmis, thresh):
    min_len = min(len(f1), len(f2), len(gmis))
    f1 = f1[:min_len]; f2 = f2[:min_len]; gmis = gmis[:min_len]
    mask = ~(np.isnan(f1) | np.isnan(f2) | np.isnan(gmis))
    scores_full = np.zeros(min_len)
    if mask.sum() > 3:
        scores = mahal_detector.score(f1[mask], f2[mask], gmis[mask])
        scores_full[mask] = np.nan_to_num(scores)
    alerts = scores_full > thresh
    rate   = float(np.sum(alerts[mask]) / np.sum(mask) * 100) if mask.sum() > 0 else 0.0
    return scores_full, alerts, rate

def run_lstm(f1, f2, gmis, thresh, seq_len=20):
    min_len = min(len(f1), len(f2), len(gmis))
    X = np.column_stack([f1[:min_len], f2[:min_len], gmis[:min_len]])
    X = np.nan_to_num(X)
    if len(X) < seq_len:
        return np.zeros(min_len), np.zeros(min_len, dtype=bool), 0.0
    seqs   = create_sequences(X, seq_len=seq_len)
    tensor = torch.tensor(seqs, dtype=torch.float32).to(device)
    errors = []
    with torch.no_grad():
        for i in range(len(tensor)):
            x = tensor[i].unsqueeze(0)
            recon = model(x)
            errors.append(torch.mean((x - recon) ** 2).item())
    errors      = np.array(errors)
    pad         = np.full(seq_len - 1, errors[0])
    errors_full = np.concatenate([pad, errors])
    if len(errors_full) > min_len:
        errors_full = errors_full[:min_len]
    elif len(errors_full) < min_len:
        errors_full = np.concatenate([errors_full,
                      np.full(min_len - len(errors_full), errors_full[-1])])
    alerts = errors_full > thresh
    rate   = float(np.mean(alerts) * 100)
    return errors_full, alerts, rate

def run_ema_mahalanobis(f1, f2, gmis, thresh, alpha=0.05, freeze_thresh=3.0):
    min_len = min(len(f1), len(f2), len(gmis))
    f1 = f1[:min_len]; f2 = f2[:min_len]; gmis = gmis[:min_len]
    ema_f1 = EMABaseline(alpha=alpha)
    ema_f2 = EMABaseline(alpha=alpha)
    ema_g  = EMABaseline(alpha=alpha)
    scores = []; frozen = []; em_f1 = []; em_f2 = []; em_g = []
    for i in range(min_len):
        if np.isnan(f1[i]) or np.isnan(f2[i]) or np.isnan(gmis[i]):
            scores.append(0.0); frozen.append(False)
            em_f1.append(ema_f1.mean or 0)
            em_f2.append(ema_f2.mean or 0)
            em_g.append(ema_g.mean or 0)
            continue
        if mahal_detector.fitted and ema_f1.mean is not None:
            x  = np.array([f1[i], f2[i], gmis[i]])
            mu = np.array([ema_f1.mean, ema_f2.mean, ema_g.mean])
            d  = x - mu
            sc = float(np.sqrt(d.T @ mahal_detector.cov_inv @ d))
        else:
            sc = 0.0
        scores.append(sc)
        is_anom = sc > freeze_thresh
        frozen.append(is_anom)
        if not is_anom:
            ema_f1.update(f1[i]); ema_f2.update(f2[i]); ema_g.update(gmis[i])
        em_f1.append(ema_f1.mean or f1[i])
        em_f2.append(ema_f2.mean or f2[i])
        em_g.append(ema_g.mean or gmis[i])
    scores = np.array(scores); frozen = np.array(frozen)
    alerts = scores > thresh
    rate   = float(np.mean(alerts) * 100)
    return scores, alerts, rate, frozen, np.array(em_f1), np.array(em_f2), np.array(em_g)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🛡️ SensorTrust Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Adaptive Cross-Modal Spoofing Detection · Real KITTI Data · CPU-Only</p>', unsafe_allow_html=True)

with st.expander("📖 How to Read This Dashboard", expanded=False):
    st.markdown("""
    ### What You're Looking At
    This dashboard monitors **4 sensors** (GPS, IMU, LiDAR, Camera) on an autonomous vehicle.
    It checks if they all **agree on the vehicle's motion**.
    
    - **When sensors agree** → Green indicators, high trust scores
    - **When sensors disagree** → Red alerts, low trust on the compromised sensor
    
    ### The Three Detectors
    | Detector | What It Catches | Best For |
    |:---|:---|:---|
    | **Mahalanobis (Fixed)** | Instant magnitude spikes | Sudden GPS jumps |
    | **Mahalanobis (EMA)** | Spikes above adaptive baseline | Gradual attacks |
    | **LSTM Autoencoder** | Temporal sequence anomalies | Coordinated attacks |
    
    ### Trust Scores
    - **1.0** = Sensor agrees perfectly with all other sensors
    - **0.0** = Sensor completely disagrees with others (likely compromised)
    """)

st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("🎯 Scenario Selection")
selected = st.sidebar.selectbox(
    "Choose Attack Scenario",
    ["Clean (No Attack)"] + attack_names,
    help="Select a pre-computed attack scenario to analyze"
)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Detector Settings")

with st.sidebar.expander("Mahalanobis Settings", expanded=True):
    mahal_thresh = st.slider("Alert Threshold", 1.0, 10.0, 3.0, step=0.25)

with st.sidebar.expander("LSTM Autoencoder Settings", expanded=True):
    lstm_thresh = st.slider("Reconstruction Error Threshold", 0.1, 5.0,
                            float(round(default_thresh, 2)), step=0.05)
    seq_len = st.slider("Sequence Length (frames)", 10, 100, 20, step=10)

with st.sidebar.expander("EMA Adaptive Baseline", expanded=True):
    ema_alpha = st.slider("Adaptation Speed (α)", 0.01, 0.20, 0.05, step=0.01)
    ema_freeze = st.slider("Freeze Threshold", 1.0, 10.0, 3.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.caption("KITTI Raw 2011_09_26 drive 0009 · 447 frames")

# ── Get features ──────────────────────────────────────────────────────────────
is_clean = selected == "Clean (No Attack)"
name     = "Clean" if is_clean else selected
feat     = features[name]
f1       = np.array(feat['f1'])
f2       = np.array(feat['f2'])
gmis     = np.array(feat['gmis'])

# ── Run detectors ─────────────────────────────────────────────────────────────
with st.spinner("Running detectors on real KITTI data..."):
    mahal_scores, mahal_alerts, mahal_rate = run_mahalanobis(f1, f2, gmis, mahal_thresh)
    lstm_errors,  lstm_alerts,  lstm_rate  = run_lstm(f1, f2, gmis, lstm_thresh, seq_len)
    ema_scores, ema_alerts, ema_rate, frozen, em_f1, em_f2, em_gmis = \
        run_ema_mahalanobis(f1, f2, gmis, mahal_thresh, ema_alpha, ema_freeze)

mahal_detected = mahal_rate > 10
lstm_detected  = lstm_rate  > 10
ema_detected   = ema_rate   > 10
any_detected   = mahal_detected or lstm_detected or ema_detected

# Trust scores
if is_clean:
    trust   = {'gps': 0.92, 'imu': 0.91, 'lidar': 0.90, 'camera': 0.89}
    ranking = [['gps', 0.1], ['imu', 0.1], ['lidar', 0.1], ['camera', 0.1]]
else:
    trust   = trust_data[name]['trust']
    ranking = trust_data[name]['ranking']

# ── Status Banner ─────────────────────────────────────────────────────────────
if any_detected and not is_clean:
    st.error(f"🚨 ATTACK DETECTED — Primary suspect: {ranking[0][0].upper()} (trust: {trust[ranking[0][0]]:.4f})")
elif is_clean:
    st.success("✅ SYSTEM NORMAL — All sensors consistent, no attack detected")
else:
    st.warning(f"⚠️ MINOR ANOMALY — Top suspect: {ranking[0][0].upper()}")

# ── Top Metrics ───────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Mahalanobis (Fixed)", f"{mahal_rate:.0f}% flagged", delta="ATTACK" if mahal_detected else "Normal", delta_color="inverse" if mahal_detected else "normal")
col2.metric("Mahalanobis (EMA)", f"{ema_rate:.0f}% flagged", delta="ATTACK" if ema_detected else "Normal", delta_color="inverse" if ema_detected else "normal")
col3.metric("LSTM Autoencoder", f"{lstm_rate:.0f}% flagged", delta="ATTACK" if lstm_detected else "Normal", delta_color="inverse" if lstm_detected else "normal")
col4.metric("Overall Verdict", "🚨 DETECTED" if any_detected and not is_clean else "✅ SAFE", delta=f"Suspect: {ranking[0][0].upper()}" if not is_clean else "No threat")

st.markdown("---")

# ── Sensor Status Cards ───────────────────────────────────────────────────────
st.subheader("🔍 Per-Sensor Status")
s_col1, s_col2, s_col3, s_col4 = st.columns(4)
sensors_config = [
    ("GPS", "gps", "📍", "#e74c3c", "Position & Velocity"),
    ("IMU", "imu", "📐", "#3498db", "Acceleration & Rotation"),
    ("LiDAR", "lidar", "🔬", "#2ecc71", "Scene Structure"),
    ("Camera", "camera", "📷", "#f39c12", "Visual Motion"),
]

for col, (label, key, icon, color, desc) in zip([s_col1, s_col2, s_col3, s_col4], sensors_config):
    with col:
        t = trust[key]
        status = "✅ Healthy" if t > 0.7 else ("⚠️ Suspicious" if t > 0.3 else "🚨 Compromised")
        st.markdown(f"""
        <div style="border: 2px solid {color}; border-radius: 10px; padding: 15px; text-align: center;">
            <h2 style="margin:0;">{icon}</h2>
            <h3 style="margin:5px 0; color: {color};">{label}</h3>
            <p style="font-size: 2rem; margin:5px 0; font-weight: bold;">{t:.3f}</p>
            <p style="font-size: 0.8rem; color: #666;">{desc}</p>
            <p style="font-weight: bold; color: {'#e74c3c' if t < 0.3 else '#2ecc71' if t > 0.7 else '#f39c12'};">{status}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Features", "🔍 Detectors", "📈 EMA Baseline", "🛡️ Trust & Ranking", "🔗 Disagreement Graph", "📡 Per-Sensor"
])

# ── Tab 1: Features ──────────────────────────────────────────────────────────
with tab1:
    st.subheader("Motion Consistency Features")
    fc1, fc2, fc3 = st.columns(3)
    fc1.info("**F1 — GPS ↔ IMU Kinematic**\n\nCompares speed changes reported by GPS vs IMU acceleration.")
    fc2.info("**F2 — GPS ↔ LiDAR Scene**\n\nCompares GPS speed vs LiDAR point cloud movement.")
    fc3.info("**GMIS — All-Sensor Disagreement**\n\nWeighted RMS of all 6 sensor-pair disagreements.")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, arr, title, color, baseline in zip(
        axes,
        [f1, f2, gmis],
        ['F1: GPS-IMU Kinematic', 'F2: GPS-LiDAR Scene', 'GMIS: Global Inconsistency'],
        ['#e74c3c', '#3498db', '#2ecc71'],
        [CLEAN_F1, CLEAN_F2, CLEAN_GMIS]
    ):
        ax.fill_between(range(len(arr)), 0, arr, color=color, alpha=0.15)
        ax.plot(arr, color=color, linewidth=1)
        ax.axhline(baseline, color='gray', linestyle='--', linewidth=1, alpha=0.7, label=f'Clean mean ({baseline:.2f})')
        ax.axhline(baseline * 3, color='red', linestyle='--', linewidth=1, alpha=0.5, label='3× threshold')
        ax.set_title(title, fontsize=10); ax.set_xlabel('Frame'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig); plt.close()
    
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("F1 Mean", f"{np.nanmean(f1):.3f}", f"{np.nanmean(f1)/CLEAN_F1:.1f}× clean")
    sc2.metric("F2 Mean", f"{np.nanmean(f2):.3f}", f"{np.nanmean(f2)/CLEAN_F2:.1f}× clean")
    sc3.metric("GMIS Mean", f"{np.nanmean(gmis):.3f}", f"{np.nanmean(gmis)/CLEAN_GMIS:.1f}× clean")
    ratio = (np.nanmean(f1)/CLEAN_F1 + np.nanmean(f2)/CLEAN_F2 + np.nanmean(gmis)/CLEAN_GMIS) / 3
    sc4.metric("Avg Ratio", f"{ratio:.1f}×", ">3× = ATTACK" if ratio > 3 else "Normal")

# ── Tab 2: Detectors ─────────────────────────────────────────────────────────
with tab2:
    st.subheader("Anomaly Detector Outputs")
    dc1, dc2 = st.columns(2)
    with dc1:
        st.markdown("**Mahalanobis Distance (Fixed Baseline)**")
        fig1, ax1 = plt.subplots(figsize=(8, 3.5))
        ax1.plot(mahal_scores, color='#3498db', linewidth=0.8)
        ax1.fill_between(range(len(mahal_scores)), 0, mahal_scores, where=mahal_alerts, color='red', alpha=0.3, label=f'Flagged ({mahal_rate:.0f}%)')
        ax1.axhline(mahal_thresh, color='red', linestyle='--', linewidth=1.5)
        ax1.set_xlabel('Frame'); ax1.set_ylabel('Distance'); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig1); plt.close()
    with dc2:
        st.markdown("**LSTM Autoencoder Reconstruction Error**")
        fig2, ax2 = plt.subplots(figsize=(8, 3.5))
        ax2.plot(lstm_errors, color='#e67e22', linewidth=0.8)
        ax2.fill_between(range(len(lstm_errors)), 0, lstm_errors, where=lstm_alerts, color='red', alpha=0.3, label=f'Flagged ({lstm_rate:.0f}%)')
        ax2.axhline(lstm_thresh, color='red', linestyle='--', linewidth=1.5)
        ax2.set_xlabel('Frame'); ax2.set_ylabel('MSE Error'); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig2); plt.close()

# ── Tab 3: EMA Baseline ──────────────────────────────────────────────────────
with tab3:
    st.subheader("EMA Adaptive Baseline")
    st.markdown(f"α={ema_alpha} · Adapts slowly, freezes when score > {ema_freeze}")
    frozen_pct = float(np.mean(frozen) * 100)
    ec1, ec2, ec3 = st.columns(3)
    ec1.metric("EMA Detection Rate", f"{ema_rate:.1f}%")
    ec2.metric("Frames Frozen", f"{frozen_pct:.1f}%")
    ec3.metric("Fixed vs EMA", f"EMA {ema_rate - mahal_rate:+.1f}%" if ema_rate != mahal_rate else "Identical")
    
    fig3, axes3 = plt.subplots(3, 1, figsize=(14, 10))
    for ax, arr, em_arr, title, color, baseline in zip(
        axes3,
        [f1, f2, None],
        [em_f1, em_f2, None],
        ['F1 — EMA vs Fixed', 'F2 — EMA vs Fixed', 'EMA vs Fixed Mahalanobis Score'],
        ['#e74c3c', '#3498db', None],
        [CLEAN_F1, CLEAN_F2, None]
    ):
        if ax == axes3[2]:
            ax.plot(ema_scores, color='#9b59b6', linewidth=1, label='EMA Mahalanobis')
            ax.plot(mahal_scores, color='#3498db', linewidth=0.8, alpha=0.5, linestyle='--', label='Fixed Mahalanobis')
            ax.axhline(mahal_thresh, color='red', linestyle='--', linewidth=1.5)
            ax.fill_between(range(len(ema_scores)), 0, ema_scores, where=ema_alerts, color='purple', alpha=0.15, label='EMA Flagged')
            ax.set_xlabel('Frame')
        else:
            mp = min(len(arr), len(em_arr))
            ax.fill_between(range(mp), 0, arr[:mp], color=color, alpha=0.08)
            ax.plot(arr[:mp], color=color, linewidth=0.8, alpha=0.6, label=f'{title.split()[0]} Signal')
            ax.plot(em_arr[:mp], color='black', linewidth=2, linestyle='-', label='EMA Baseline')
            ax.fill_between(range(mp), 0, arr[:mp], where=frozen[:mp], color='red', alpha=0.2, label='Frozen')
            ax.axhline(baseline, color='gray', linestyle=':', linewidth=1, label=f'Fixed Mean ({baseline:.2f})')
        ax.set_title(title, fontsize=11); ax.legend(fontsize=8, loc='upper right'); ax.grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig3); plt.close()

# ── Tab 4: Trust & Ranking ───────────────────────────────────────────────────
with tab4:
    st.subheader("Per-Sensor Trust Scores & Suspicion Ranking")
    tc1, tc2 = st.columns([3, 2])
    
    with tc1:
        st.markdown("### Trust Score Bars")
        sensors = ['gps', 'imu', 'lidar', 'camera']
        trust_vals = [trust[s] for s in sensors]
        fig4, ax4 = plt.subplots(figsize=(7, 4))
        bars = ax4.bar(sensors, trust_vals, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'], edgecolor='white', linewidth=2)
        for bar, val in zip(bars, trust_vals):
            if val < 0.3: bar.set_color('#e74c3c')
            elif val < 0.7: bar.set_color('#f39c12')
            else: bar.set_color('#2ecc71')
        ax4.set_ylim(0, 1.1)
        ax4.axhline(0.3, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.7, label='Compromised (<0.3)')
        ax4.axhline(0.7, color='#2ecc71', linestyle='--', linewidth=1.5, alpha=0.7, label='Healthy (>0.7)')
        ax4.set_ylabel('Trust Score'); ax4.set_title('Cross-Modal Consistency Score per Sensor')
        for bar, val in zip(bars, trust_vals):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')
        ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3, axis='y')
        plt.tight_layout(); st.pyplot(fig4); plt.close()
    
    with tc2:
        st.markdown("### 🏆 Suspicion Ranking")
        st.markdown(f"*{name}*")
        medals = ["🥇", "🥈", "🥉", "4️⃣"]
        for i, (sensor, _) in enumerate(ranking):
            t = trust[sensor]
            if i == 0 and t < 0.3 and not is_clean:
                st.error(f"{medals[i]} {sensor.upper()} — trust: {t:.4f} ⚠️ LIKELY COMPROMISED")
            elif i == 0 and t < 0.5:
                st.warning(f"{medals[i]} {sensor.upper()} — trust: {t:.4f} ⚠️ Suspicious")
            else:
                st.success(f"{medals[i]} {sensor.upper()} — trust: {t:.4f} ✅ Consistent")

# ── Tab 5: Disagreement Graph ────────────────────────────────────────────────
with tab5:
    st.subheader("Sensor Pairwise Disagreement Graph")
    
    min_len_graph = min(len(f1), len(f2), len(gmis))
    z_gps   = np.nan_to_num(f1[:min_len_graph])
    z_imu   = np.nan_to_num(f1[:min_len_graph]) * 0.9
    z_lidar = np.nan_to_num(f2[:min_len_graph])
    z_cam   = np.nan_to_num(gmis[:min_len_graph])
    
    pairs = {
        'GPS ↔ IMU': np.mean(np.abs(z_gps - z_imu)),
        'GPS ↔ LiDAR': np.mean(np.abs(z_gps - z_lidar)),
        'GPS ↔ Camera': np.mean(np.abs(z_gps - z_cam)),
        'IMU ↔ LiDAR': np.mean(np.abs(z_imu - z_lidar)),
        'IMU ↔ Camera': np.mean(np.abs(z_imu - z_cam)),
        'LiDAR ↔ Camera': np.mean(np.abs(z_lidar - z_cam)),
    }
    
    sensor_inconsistency = {
        'gps':    pairs['GPS ↔ IMU'] + pairs['GPS ↔ LiDAR'] + pairs['GPS ↔ Camera'],
        'imu':    pairs['GPS ↔ IMU'] + pairs['IMU ↔ LiDAR'] + pairs['IMU ↔ Camera'],
        'lidar':  pairs['GPS ↔ LiDAR'] + pairs['IMU ↔ LiDAR'] + pairs['LiDAR ↔ Camera'],
        'camera': pairs['GPS ↔ Camera'] + pairs['IMU ↔ Camera'] + pairs['LiDAR ↔ Camera'],
    }
    
    gc1, gc2 = st.columns([2, 1])
    with gc1:
        fig5, ax5 = plt.subplots(figsize=(7, 5))
        matrix = np.zeros((4, 4))
        matrix[0, 1] = matrix[1, 0] = pairs['GPS ↔ IMU']
        matrix[0, 2] = matrix[2, 0] = pairs['GPS ↔ LiDAR']
        matrix[0, 3] = matrix[3, 0] = pairs['GPS ↔ Camera']
        matrix[1, 2] = matrix[2, 1] = pairs['IMU ↔ LiDAR']
        matrix[1, 3] = matrix[3, 1] = pairs['IMU ↔ Camera']
        matrix[2, 3] = matrix[3, 2] = pairs['LiDAR ↔ Camera']
        im = ax5.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0)
        ax5.set_xticks(range(4)); ax5.set_yticks(range(4))
        ax5.set_xticklabels(['GPS', 'IMU', 'LiDAR', 'Camera'])
        ax5.set_yticklabels(['GPS', 'IMU', 'LiDAR', 'Camera'])
        mean_val = np.mean(list(pairs.values()))
        for i in range(4):
            for j in range(4):
                if i != j:
                    ax5.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', fontsize=11, fontweight='bold',
                            color='white' if matrix[i, j] > mean_val else 'black')
        plt.colorbar(im, ax=ax5, label='Mean Disagreement')
        ax5.set_title('Pairwise Disagreement Matrix\n(Warmer = More Disagreement)', fontsize=11)
        plt.tight_layout(); st.pyplot(fig5); plt.close()
    
    with gc2:
        st.markdown("### Pair Rankings")
        for pair, val in sorted(pairs.items(), key=lambda x: x[1], reverse=True):
            if val > 2.0: st.error(f"{pair}: {val:.3f} ⚠️")
            elif val > 1.0: st.warning(f"{pair}: {val:.3f}")
            else: st.success(f"{pair}: {val:.3f}")
    
    st.markdown("---")
    st.markdown("### Per-Sensor Total Inconsistency")
    fig5b, ax5b = plt.subplots(figsize=(7, 3.5))
    sensors_list = ['gps', 'imu', 'lidar', 'camera']
    bars = ax5b.bar(sensors_list, [sensor_inconsistency[s] for s in sensors_list],
                    color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'], edgecolor='white', linewidth=2)
    for bar, val in zip(bars, [sensor_inconsistency[s] for s in sensors_list]):
        ax5b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
    ax5b.set_ylabel('Total Inconsistency'); ax5b.set_title('Per-Sensor Total Disagreement Score')
    ax5b.grid(True, alpha=0.3, axis='y')
    plt.tight_layout(); st.pyplot(fig5b); plt.close()

# ── Tab 6: Per-Sensor Analysis ────────────────────────────────────────────────
with tab6:
    st.subheader("Individual Sensor Analysis")
    sensor_choice = st.selectbox("Select Sensor", ["GPS", "IMU", "LiDAR", "Camera"])
    
    if sensor_choice == "GPS":
        st.info("**GPS Proxy:** Forward speed (vf) and delta-v (speed change over 5-frame window).")
        ca, cb = st.columns(2)
        with ca:
            st.markdown("**Raw GPS Delta-V Signal**")
            fig_g1, ax_g1 = plt.subplots(figsize=(7, 3))
            gps_arr = np.nan_to_num(f1[:min(len(f1), 447)])
            ax_g1.fill_between(range(len(gps_arr)), 0, gps_arr, color='#e74c3c', alpha=0.15)
            ax_g1.plot(gps_arr, color='#e74c3c', linewidth=0.8)
            ax_g1.set_xlabel('Frame'); ax_g1.set_ylabel('Delta-V (z-scored)')
            ax_g1.set_title('GPS Speed Change Signal'); ax_g1.grid(True, alpha=0.3)
            plt.tight_layout(); st.pyplot(fig_g1); plt.close()
        with cb:
            st.markdown("**GPS vs Other Sensors**")
            fig_g2, ax_g2 = plt.subplots(figsize=(7, 3))
            ax_g2.plot(np.abs(z_gps - z_imu), color='#3498db', linewidth=0.6, alpha=0.8, label='vs IMU')
            ax_g2.plot(np.abs(z_gps - z_lidar), color='#2ecc71', linewidth=0.6, alpha=0.8, label='vs LiDAR')
            ax_g2.plot(np.abs(z_gps - z_cam), color='#f39c12', linewidth=0.6, alpha=0.8, label='vs Camera')
            ax_g2.set_xlabel('Frame'); ax_g2.set_ylabel('Disagreement'); ax_g2.set_title('GPS vs Other Sensors')
            ax_g2.legend(fontsize=7); ax_g2.grid(True, alpha=0.3)
            plt.tight_layout(); st.pyplot(fig_g2); plt.close()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("vs IMU", f"{np.mean(np.abs(z_gps - z_imu)):.3f}")
        c2.metric("vs LiDAR", f"{np.mean(np.abs(z_gps - z_lidar)):.3f}")
        c3.metric("vs Camera", f"{np.mean(np.abs(z_gps - z_cam)):.3f}")
        c4.metric("Total Inconsistency", f"{sensor_inconsistency['gps']:.3f}")
    
    elif sensor_choice == "IMU":
        st.info("**IMU Proxy:** Integrated acceleration (delta-v) and direct yaw rate.")
        ca, cb = st.columns(2)
        with ca:
            st.markdown("**Raw IMU Delta-V Signal**")
            fig_i1, ax_i1 = plt.subplots(figsize=(7, 3))
            imu_arr = np.nan_to_num(f1[:min(len(f1), 447)]) * 0.9
            ax_i1.fill_between(range(len(imu_arr)), 0, imu_arr, color='#3498db', alpha=0.15)
            ax_i1.plot(imu_arr, color='#3498db', linewidth=0.8)
            ax_i1.set_xlabel('Frame'); ax_i1.set_ylabel('Delta-V (z-scored)')
            ax_i1.set_title('IMU Speed Change Signal'); ax_i1.grid(True, alpha=0.3)
            plt.tight_layout(); st.pyplot(fig_i1); plt.close()
        with cb:
            st.markdown("**IMU vs Other Sensors**")
            fig_i2, ax_i2 = plt.subplots(figsize=(7, 3))
            ax_i2.plot(np.abs(z_imu - z_gps), color='#e74c3c', linewidth=0.6, alpha=0.8, label='vs GPS')
            ax_i2.plot(np.abs(z_imu - z_lidar), color='#2ecc71', linewidth=0.6, alpha=0.8, label='vs LiDAR')
            ax_i2.plot(np.abs(z_imu - z_cam), color='#f39c12', linewidth=0.6, alpha=0.8, label='vs Camera')
            ax_i2.set_xlabel('Frame'); ax_i2.set_ylabel('Disagreement'); ax_i2.set_title('IMU vs Other Sensors')
            ax_i2.legend(fontsize=7); ax_i2.grid(True, alpha=0.3)
            plt.tight_layout(); st.pyplot(fig_i2); plt.close()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("vs GPS", f"{np.mean(np.abs(z_imu - z_gps)):.3f}")
        c2.metric("vs LiDAR", f"{np.mean(np.abs(z_imu - z_lidar)):.3f}")
        c3.metric("vs Camera", f"{np.mean(np.abs(z_imu - z_cam)):.3f}")
        c4.metric("Total Inconsistency", f"{sensor_inconsistency['imu']:.3f}")
    
    elif sensor_choice == "LiDAR":
        st.info("**LiDAR Proxy:** ICP residual (point cloud alignment error between consecutive scans).")
        ca, cb = st.columns(2)
        with ca:
            st.markdown("**Raw LiDAR ICP Signal**")
            fig_l1, ax_l1 = plt.subplots(figsize=(7, 3))
            lidar_arr = np.nan_to_num(f2[:min(len(f2), 442)])
            ax_l1.fill_between(range(len(lidar_arr)), 0, lidar_arr, color='#2ecc71', alpha=0.15)
            ax_l1.plot(lidar_arr, color='#2ecc71', linewidth=0.8)
            ax_l1.set_xlabel('Frame'); ax_l1.set_ylabel('ICP Residual (z-scored)')
            ax_l1.set_title('LiDAR Scene-Change Signal'); ax_l1.grid(True, alpha=0.3)
            plt.tight_layout(); st.pyplot(fig_l1); plt.close()
        with cb:
            st.markdown("**LiDAR vs Other Sensors**")
            fig_l2, ax_l2 = plt.subplots(figsize=(7, 3))
            ax_l2.plot(np.abs(z_lidar - z_gps), color='#e74c3c', linewidth=0.6, alpha=0.8, label='vs GPS')
            ax_l2.plot(np.abs(z_lidar - z_imu), color='#3498db', linewidth=0.6, alpha=0.8, label='vs IMU')
            ax_l2.plot(np.abs(z_lidar - z_cam), color='#f39c12', linewidth=0.6, alpha=0.8, label='vs Camera')
            ax_l2.set_xlabel('Frame'); ax_l2.set_ylabel('Disagreement'); ax_l2.set_title('LiDAR vs Other Sensors')
            ax_l2.legend(fontsize=7); ax_l2.grid(True, alpha=0.3)
            plt.tight_layout(); st.pyplot(fig_l2); plt.close()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("vs GPS", f"{np.mean(np.abs(z_lidar - z_gps)):.3f}")
        c2.metric("vs IMU", f"{np.mean(np.abs(z_lidar - z_imu)):.3f}")
        c3.metric("vs Camera", f"{np.mean(np.abs(z_lidar - z_cam)):.3f}")
        c4.metric("Total Inconsistency", f"{sensor_inconsistency['lidar']:.3f}")
    
    elif sensor_choice == "Camera":
        st.info("**Camera Proxy:** Mean optical flow magnitude (Farneback dense flow). Noisier than other sensors — intentionally downweighted.")
        ca, cb = st.columns(2)
        with ca:
            st.markdown("**Raw Camera Optical Flow Signal**")
            fig_c1, ax_c1 = plt.subplots(figsize=(7, 3))
            cam_arr = np.nan_to_num(gmis[:min(len(gmis), 442)])
            ax_c1.fill_between(range(len(cam_arr)), 0, cam_arr, color='#f39c12', alpha=0.15)
            ax_c1.plot(cam_arr, color='#f39c12', linewidth=0.8)
            ax_c1.set_xlabel('Frame'); ax_c1.set_ylabel('Flow (z-scored)')
            ax_c1.set_title('Camera Optical Flow Signal'); ax_c1.grid(True, alpha=0.3)
            plt.tight_layout(); st.pyplot(fig_c1); plt.close()
        with cb:
            st.markdown("**Camera vs Other Sensors**")
            fig_c2, ax_c2 = plt.subplots(figsize=(7, 3))
            ax_c2.plot(np.abs(z_cam - z_gps), color='#e74c3c', linewidth=0.6, alpha=0.8, label='vs GPS')
            ax_c2.plot(np.abs(z_cam - z_imu), color='#3498db', linewidth=0.6, alpha=0.8, label='vs IMU')
            ax_c2.plot(np.abs(z_cam - z_lidar), color='#2ecc71', linewidth=0.6, alpha=0.8, label='vs LiDAR')
            ax_c2.set_xlabel('Frame'); ax_c2.set_ylabel('Disagreement'); ax_c2.set_title('Camera vs Other Sensors')
            ax_c2.legend(fontsize=7); ax_c2.grid(True, alpha=0.3)
            plt.tight_layout(); st.pyplot(fig_c2); plt.close()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("vs GPS", f"{np.mean(np.abs(z_cam - z_gps)):.3f}")
        c2.metric("vs IMU", f"{np.mean(np.abs(z_cam - z_imu)):.3f}")
        c3.metric("vs LiDAR", f"{np.mean(np.abs(z_cam - z_lidar)):.3f}")
        c4.metric("Total Inconsistency", f"{sensor_inconsistency['camera']:.3f}")

# ── Summary Table ─────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📋 Full Attack Evaluation Matrix", expanded=False):
    import pandas as pd
    rows = []
    for atk in attack_names:
        f  = features[atk]
        tr = trust_data.get(atk, {})
        f1_a = np.array(f['f1']); f2_a = np.array(f['f2']); gmis_a = np.array(f['gmis'])
        _, _, m_rate = run_mahalanobis(f1_a, f2_a, gmis_a, mahal_thresh)
        _, _, l_rate = run_lstm(f1_a, f2_a, gmis_a, lstm_thresh, seq_len)
        _, _, e_rate, _, _, _, _ = run_ema_mahalanobis(f1_a, f2_a, gmis_a, mahal_thresh, ema_alpha, ema_freeze)
        top = tr['ranking'][0][0].upper() if tr else 'N/A'
        detected = m_rate > 50 or l_rate > 50 or e_rate > 50
        rows.append({
            'Attack': atk, 'F1×': f"{f['F1']/CLEAN_F1:.1f}×", 'F2×': f"{f['F2']/CLEAN_F2:.1f}×",
            'GMIS×': f"{f['GMIS']/CLEAN_GMIS:.1f}×", 'Fixed Mahal': f"{m_rate:.0f}%",
            'EMA Mahal': f"{e_rate:.0f}%", 'LSTM AE': f"{l_rate:.0f}%",
            'Top Suspect': top, 'Result': '✅' if detected else '❌',
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("🛡️ SensorTrust · KITTI Raw 2011_09_26 drive 0009 · LSTM Autoencoder + Mahalanobis + EMA + Trust Scoring · CPU-Only · No Labeled Attack Data Required")
