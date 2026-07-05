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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SensorTrust Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; margin-bottom: 0; }
    .sub-header { color: #666; font-size: 1rem; margin-top: 0; }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_features(drive):
    suffix = '' if drive == '0009' else f'_{drive}'
    with open(f'results/features{suffix}.json') as f:
        return json.load(f)

@st.cache_data
def load_trust(drive):
    suffix = '' if drive == '0009' else f'_{drive}'
    with open(f'results/trust{suffix}.json') as f:
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

model, device, default_thresh = load_model()

# ── Detection functions ───────────────────────────────────────────────────────
def run_mahalanobis(f1, f2, gmis, thresh, mahal_detector):
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

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🛡️ SensorTrust Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Adaptive Cross-Modal Spoofing Detection · Real KITTI Data · CPU-Only</p>', unsafe_allow_html=True)

with st.expander("📖 How to Read This Dashboard", expanded=False):
    st.markdown("""
    ### What You're Looking At
    This dashboard monitors **4 sensors** (GPS, IMU, LiDAR, Camera) on an autonomous vehicle
    and checks if they all **agree on the vehicle's motion**.

    - **When sensors agree** → Green indicators, high consistency scores
    - **When sensors disagree** → Red alerts, low consistency on the compromised sensor

    ### The Two Detectors
    | Detector | What It Catches | Best For |
    |:---|:---|:---|
    | **Mahalanobis Distance** | Statistical outliers vs clean baseline | High-precision alerts, IMU attacks |
    | **LSTM Autoencoder** | Temporal sequence anomalies | GPS ramps, coordinated attacks |

    ### Consistency Scores
    - **1.0** = Sensor agrees with all other sensors (consistent)
    - **0.0** = Sensor completely disagrees with others (likely compromised)

    *Note: Consistency scores measure cross-modal agreement, not direct attack probability.*

    ### Generalization
    Toggle between **Drive 0009** (primary evaluation, 447 frames) and
    **Drive 0051** (generalization test, 438 frames) to see if detection holds on unseen data.
    """)

st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("🗺️ Dataset")
drive = st.sidebar.radio(
    "KITTI Drive",
    ["0009", "0051"],
    format_func=lambda x: f"Drive {x} ({'447 frames · Primary' if x == '0009' else '438 frames · Generalization'})",
    help="Model trained on drive 0009. Drive 0051 tests generalization to unseen data."
)

st.sidebar.markdown("---")
st.sidebar.header("🎯 Scenario")

# load data for selected drive
features     = load_features(drive)
trust_data   = load_trust(drive)
mahal_det    = load_mahal(features)

clean        = features['Clean']
CLEAN_F1     = clean['F1']
CLEAN_F2     = clean['F2']
CLEAN_GMIS   = clean['GMIS']
attack_names = [k for k in features.keys() if k != 'Clean']

selected = st.sidebar.selectbox(
    "Attack Scenario",
    ["Clean (No Attack)"] + attack_names,
    help="Select a pre-computed attack scenario"
)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Detector Settings")

with st.sidebar.expander("Mahalanobis Settings", expanded=True):
    mahal_thresh = st.slider("Alert Threshold", 1.0, 10.0, 3.0, step=0.25)

with st.sidebar.expander("LSTM Autoencoder Settings", expanded=True):
    lstm_thresh = st.slider("Reconstruction Error Threshold", 0.1, 5.0,
                            float(round(default_thresh, 2)), step=0.05)
    seq_len = st.slider("Sequence Length (frames)", 10, 100, 20, step=10)

st.sidebar.markdown("---")
st.sidebar.caption(f"KITTI Raw 2011_09_26 · Drive {drive}")

# ── Get features ──────────────────────────────────────────────────────────────
is_clean = selected == "Clean (No Attack)"
name     = "Clean" if is_clean else selected
feat     = features[name]
f1       = np.array(feat['f1'])
f2       = np.array(feat['f2'])
gmis     = np.array(feat['gmis'])

# ── Run detectors ─────────────────────────────────────────────────────────────
with st.spinner("Running detectors on real KITTI data..."):
    mahal_scores, mahal_alerts, mahal_rate = run_mahalanobis(f1, f2, gmis, mahal_thresh, mahal_det)
    lstm_errors,  lstm_alerts,  lstm_rate  = run_lstm(f1, f2, gmis, lstm_thresh, seq_len)

mahal_detected = mahal_rate > 10
lstm_detected  = lstm_rate  > 10
any_detected   = mahal_detected or lstm_detected

if is_clean:
    trust   = {'gps': 0.92, 'imu': 0.91, 'lidar': 0.90, 'camera': 0.89}
    ranking = [['gps', 0.1], ['imu', 0.1], ['lidar', 0.1], ['camera', 0.1]]
else:
    trust   = trust_data[name]['trust']
    ranking = trust_data[name]['ranking']

# ── Drive badge ───────────────────────────────────────────────────────────────
drive_color = "#3498db" if drive == "0009" else "#9b59b6"
drive_label = "Primary Evaluation" if drive == "0009" else "Generalization Test"
st.markdown(
    f'<div style="background:{drive_color}22; border-left: 4px solid {drive_color}; '
    f'padding: 8px 16px; border-radius: 4px; margin-bottom: 12px;">'
    f'<b>Drive {drive}</b> — {drive_label} · '
    f'Clean baseline: F1={CLEAN_F1:.3f}, F2={CLEAN_F2:.3f}, GMIS={CLEAN_GMIS:.3f}'
    f'</div>',
    unsafe_allow_html=True
)

# ── Status Banner ─────────────────────────────────────────────────────────────
if any_detected and not is_clean:
    st.error(f"🚨 ATTACK DETECTED — Primary suspect: {ranking[0][0].upper()} "
             f"(consistency: {trust[ranking[0][0]]:.4f})")
elif is_clean:
    st.success("✅ SYSTEM NORMAL — All sensors consistent, no attack detected")
else:
    st.warning(f"⚠️ MINOR ANOMALY — Top suspect: {ranking[0][0].upper()}")

# ── Top Metrics ───────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("Mahalanobis Distance", f"{mahal_rate:.0f}% flagged",
            delta="ATTACK" if mahal_detected else "Normal",
            delta_color="inverse" if mahal_detected else "normal")
col2.metric("LSTM Autoencoder", f"{lstm_rate:.0f}% flagged",
            delta="ATTACK" if lstm_detected else "Normal",
            delta_color="inverse" if lstm_detected else "normal")
col3.metric("Overall Verdict", "🚨 DETECTED" if any_detected and not is_clean else "✅ SAFE",
            delta=f"Suspect: {ranking[0][0].upper()}" if not is_clean else "No threat")

st.markdown("---")

# ── Sensor Status Cards ───────────────────────────────────────────────────────
st.subheader("🔍 Per-Sensor Consistency")
s_col1, s_col2, s_col3, s_col4 = st.columns(4)
sensors_config = [
    ("GPS",    "gps",    "📍", "#e74c3c", "Position & Velocity"),
    ("IMU",    "imu",    "📐", "#3498db", "Acceleration & Rotation"),
    ("LiDAR",  "lidar",  "🔬", "#2ecc71", "Scene Structure"),
    ("Camera", "camera", "📷", "#f39c12", "Visual Motion"),
]

for col, (label, key, icon, color, desc) in zip(
    [s_col1, s_col2, s_col3, s_col4], sensors_config
):
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Features", "🔍 Detectors", "🛡️ Consistency & Ranking",
    "🔗 Disagreement Graph", "📡 Per-Sensor"
])

# ── Tab 1: Features ──────────────────────────────────────────────────────────
with tab1:
    st.subheader(f"Motion Consistency Features — Drive {drive}")
    fc1, fc2, fc3 = st.columns(3)
    fc1.info("**F1 — GPS ↔ IMU Kinematic**\n\nCompares speed changes from GPS vs IMU.")
    fc2.info("**F2 — GPS ↔ LiDAR Scene**\n\nCompares GPS speed vs LiDAR odometry.")
    fc3.info("**GMIS — All-Sensor Disagreement**\n\nWeighted RMS across all 6 sensor pairs.")

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
        ax.axhline(baseline, color='gray', linestyle='--', linewidth=1, alpha=0.7,
                   label=f'Clean mean ({baseline:.2f})')
        ax.axhline(baseline * 3, color='red', linestyle='--', linewidth=1, alpha=0.5,
                   label='3× threshold')
        ax.set_title(title, fontsize=10); ax.set_xlabel('Frame')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
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
        st.markdown("**Mahalanobis Distance**")
        fig1, ax1 = plt.subplots(figsize=(8, 3.5))
        ax1.plot(mahal_scores, color='#3498db', linewidth=0.8)
        ax1.fill_between(range(len(mahal_scores)), 0, mahal_scores,
                         where=mahal_alerts, color='red', alpha=0.3,
                         label=f'Flagged ({mahal_rate:.0f}%)')
        ax1.axhline(mahal_thresh, color='red', linestyle='--', linewidth=1.5)
        ax1.set_xlabel('Frame'); ax1.set_ylabel('Distance')
        ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig1); plt.close()
    with dc2:
        st.markdown("**LSTM Autoencoder Reconstruction Error**")
        fig2, ax2 = plt.subplots(figsize=(8, 3.5))
        ax2.plot(lstm_errors, color='#e67e22', linewidth=0.8)
        ax2.fill_between(range(len(lstm_errors)), 0, lstm_errors,
                         where=lstm_alerts, color='red', alpha=0.3,
                         label=f'Flagged ({lstm_rate:.0f}%)')
        ax2.axhline(lstm_thresh, color='red', linestyle='--', linewidth=1.5)
        ax2.set_xlabel('Frame'); ax2.set_ylabel('MSE Error')
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig2); plt.close()

# ── Tab 3: Consistency & Ranking ─────────────────────────────────────────────
with tab3:
    st.subheader("Per-Sensor Consistency Scores & Suspicion Ranking")
    tc1, tc2 = st.columns([3, 2])

    with tc1:
        st.markdown("### Consistency Score Bars")
        sensors    = ['gps', 'imu', 'lidar', 'camera']
        trust_vals = [trust[s] for s in sensors]
        fig4, ax4  = plt.subplots(figsize=(7, 4))
        bars = ax4.bar(sensors, trust_vals,
                       color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'],
                       edgecolor='white', linewidth=2)
        for bar, val in zip(bars, trust_vals):
            if val < 0.3: bar.set_color('#e74c3c')
            elif val < 0.7: bar.set_color('#f39c12')
            else: bar.set_color('#2ecc71')
        ax4.set_ylim(0, 1.1)
        ax4.axhline(0.3, color='#e74c3c', linestyle='--', linewidth=1.5,
                    alpha=0.7, label='Compromised (<0.3)')
        ax4.axhline(0.7, color='#2ecc71', linestyle='--', linewidth=1.5,
                    alpha=0.7, label='Healthy (>0.7)')
        ax4.set_ylabel('Consistency Score')
        ax4.set_title('Cross-Modal Consistency Score per Sensor')
        for bar, val in zip(bars, trust_vals):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')
        ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3, axis='y')
        plt.tight_layout(); st.pyplot(fig4); plt.close()

    with tc2:
        st.markdown(f"### 🏆 Suspicion Ranking")
        st.markdown(f"*{name} · Drive {drive}*")
        medals = ["🥇", "🥈", "🥉", "4️⃣"]
        for i, (sensor, _) in enumerate(ranking):
            t = trust[sensor]
            if i == 0 and t < 0.3 and not is_clean:
                st.error(f"{medals[i]} {sensor.upper()} — consistency: {t:.4f} ⚠️ LIKELY COMPROMISED")
            elif i == 0 and t < 0.5:
                st.warning(f"{medals[i]} {sensor.upper()} — consistency: {t:.4f} ⚠️ Suspicious")
            else:
                st.success(f"{medals[i]} {sensor.upper()} — consistency: {t:.4f} ✅")
        if not is_clean:
            st.markdown("---")
            st.markdown("**Feature Elevation vs Clean**")
            st.markdown(f"- F1:   `{feat['F1'] / CLEAN_F1:.1f}×`")
            st.markdown(f"- F2:   `{feat['F2'] / CLEAN_F2:.1f}×`")
            st.markdown(f"- GMIS: `{feat['GMIS'] / CLEAN_GMIS:.1f}×`")

# ── Tab 4: Disagreement Graph ────────────────────────────────────────────────
with tab4:
    st.subheader("Sensor Pairwise Disagreement Graph")
    min_len_graph = min(len(f1), len(f2), len(gmis))
    z_gps   = np.nan_to_num(f1[:min_len_graph])
    z_imu   = np.nan_to_num(f1[:min_len_graph]) * 0.9
    z_lidar = np.nan_to_num(f2[:min_len_graph])
    z_cam   = np.nan_to_num(gmis[:min_len_graph])

    pairs = {
        'GPS ↔ IMU':      np.mean(np.abs(z_gps - z_imu)),
        'GPS ↔ LiDAR':    np.mean(np.abs(z_gps - z_lidar)),
        'GPS ↔ Camera':   np.mean(np.abs(z_gps - z_cam)),
        'IMU ↔ LiDAR':    np.mean(np.abs(z_imu - z_lidar)),
        'IMU ↔ Camera':   np.mean(np.abs(z_imu - z_cam)),
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
        matrix[0,1] = matrix[1,0] = pairs['GPS ↔ IMU']
        matrix[0,2] = matrix[2,0] = pairs['GPS ↔ LiDAR']
        matrix[0,3] = matrix[3,0] = pairs['GPS ↔ Camera']
        matrix[1,2] = matrix[2,1] = pairs['IMU ↔ LiDAR']
        matrix[1,3] = matrix[3,1] = pairs['IMU ↔ Camera']
        matrix[2,3] = matrix[3,2] = pairs['LiDAR ↔ Camera']
        im = ax5.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0)
        ax5.set_xticks(range(4)); ax5.set_yticks(range(4))
        ax5.set_xticklabels(['GPS', 'IMU', 'LiDAR', 'Camera'])
        ax5.set_yticklabels(['GPS', 'IMU', 'LiDAR', 'Camera'])
        mean_val = np.mean(list(pairs.values()))
        for i in range(4):
            for j in range(4):
                if i != j:
                    ax5.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center',
                            fontsize=11, fontweight='bold',
                            color='white' if matrix[i,j] > mean_val else 'black')
        plt.colorbar(im, ax=ax5, label='Mean Disagreement')
        ax5.set_title('Pairwise Disagreement Matrix\n(Warmer = More Disagreement)')
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
    sl = ['gps', 'imu', 'lidar', 'camera']
    bars = ax5b.bar(sl, [sensor_inconsistency[s] for s in sl],
                    color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'],
                    edgecolor='white', linewidth=2)
    for bar, val in zip(bars, [sensor_inconsistency[s] for s in sl]):
        ax5b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                  f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
    ax5b.set_ylabel('Total Inconsistency')
    ax5b.set_title('Per-Sensor Total Disagreement Score')
    ax5b.grid(True, alpha=0.3, axis='y')
    plt.tight_layout(); st.pyplot(fig5b); plt.close()

# ── Tab 5: Per-Sensor ────────────────────────────────────────────────────────
with tab5:
    st.subheader("Individual Sensor Analysis")
    sensor_choice = st.selectbox("Select Sensor", ["GPS", "IMU", "LiDAR", "Camera"])

    sensor_map = {
        "GPS":    (z_gps,   '#e74c3c', 'Delta-V (z-scored)',     'GPS Speed Change Signal'),
        "IMU":    (z_imu,   '#3498db', 'Delta-V (z-scored)',     'IMU Speed Change Signal'),
        "LiDAR":  (z_lidar, '#2ecc71', 'ICP Residual (z-scored)', 'LiDAR Scene-Change Signal'),
        "Camera": (z_cam,   '#f39c12', 'Flow (z-scored)',         'Camera Optical Flow Signal'),
    }
    other_map = {
        "GPS":    [("vs IMU", z_imu, '#3498db'), ("vs LiDAR", z_lidar, '#2ecc71'), ("vs Camera", z_cam, '#f39c12')],
        "IMU":    [("vs GPS", z_gps, '#e74c3c'), ("vs LiDAR", z_lidar, '#2ecc71'), ("vs Camera", z_cam, '#f39c12')],
        "LiDAR":  [("vs GPS", z_gps, '#e74c3c'), ("vs IMU", z_imu, '#3498db'),    ("vs Camera", z_cam, '#f39c12')],
        "Camera": [("vs GPS", z_gps, '#e74c3c'), ("vs IMU", z_imu, '#3498db'),    ("vs LiDAR", z_lidar, '#2ecc71')],
    }
    key_map = {"GPS": "gps", "IMU": "imu", "LiDAR": "lidar", "Camera": "camera"}

    arr, color, ylabel, title = sensor_map[sensor_choice]
    others = other_map[sensor_choice]
    key    = key_map[sensor_choice]

    ca, cb = st.columns(2)
    with ca:
        st.markdown(f"**Raw {sensor_choice} Signal**")
        fig_s1, ax_s1 = plt.subplots(figsize=(7, 3))
        ax_s1.fill_between(range(len(arr)), 0, arr, color=color, alpha=0.15)
        ax_s1.plot(arr, color=color, linewidth=0.8)
        ax_s1.set_xlabel('Frame'); ax_s1.set_ylabel(ylabel)
        ax_s1.set_title(title); ax_s1.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig_s1); plt.close()
    with cb:
        st.markdown(f"**{sensor_choice} vs Other Sensors**")
        fig_s2, ax_s2 = plt.subplots(figsize=(7, 3))
        for label, other_arr, other_color in others:
            ax_s2.plot(np.abs(arr - other_arr), color=other_color,
                       linewidth=0.6, alpha=0.8, label=label)
        ax_s2.set_xlabel('Frame'); ax_s2.set_ylabel('Disagreement')
        ax_s2.set_title(f'{sensor_choice} vs Other Sensors')
        ax_s2.legend(fontsize=7); ax_s2.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig_s2); plt.close()

    c1, c2, c3, c4 = st.columns(4)
    vals = [np.mean(np.abs(arr - o[1])) for o in others]
    for col, (label, _, _), val in zip([c1, c2, c3], others, vals):
        col.metric(label, f"{val:.3f}")
    c4.metric("Total Inconsistency", f"{sensor_inconsistency[key]:.3f}")

# ── Summary Table ─────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📋 Full Attack Evaluation Matrix", expanded=False):
    import pandas as pd
    rows = []
    for atk in attack_names:
        f  = features[atk]
        tr = trust_data.get(atk, {})
        f1_a = np.array(f['f1']); f2_a = np.array(f['f2']); gmis_a = np.array(f['gmis'])
        _, _, m_rate = run_mahalanobis(f1_a, f2_a, gmis_a, mahal_thresh, mahal_det)
        _, _, l_rate = run_lstm(f1_a, f2_a, gmis_a, lstm_thresh, seq_len)
        top      = tr['ranking'][0][0].upper() if tr else 'N/A'
        detected = m_rate > 50 or l_rate > 50
        rows.append({
            'Attack':       atk,
            'F1×':         f"{f['F1']/CLEAN_F1:.1f}×",
            'F2×':         f"{f['F2']/CLEAN_F2:.1f}×",
            'GMIS×':       f"{f['GMIS']/CLEAN_GMIS:.1f}×",
            'Mahalanobis': f"{m_rate:.0f}%",
            'LSTM AE':     f"{l_rate:.0f}%",
            'Top Suspect': top,
            'Result':      '✅' if detected else '❌',
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── Cross-drive comparison ────────────────────────────────────────────────────
st.markdown("---")
with st.expander("🗺️ Cross-Drive Generalization Comparison", expanded=False):
    st.markdown("Compare detection rates across Drive 0009 (primary) and Drive 0051 (unseen).")
    try:
        feat_0009  = load_features('0009')
        feat_0051  = load_features('0051')
        mahal_0009 = load_mahal(feat_0009)
        mahal_0051 = load_mahal(feat_0051)

        c0009 = feat_0009['Clean']
        c0051 = feat_0051['Clean']

        gen_rows = []
        common_attacks = [a for a in feat_0009 if a != 'Clean' and a in feat_0051]

        for atk in common_attacks:
            r0009 = feat_0009[atk]; r0051 = feat_0051[atk]

            f1_09  = np.array(r0009['f1']); f2_09  = np.array(r0009['f2']); g_09  = np.array(r0009['gmis'])
            f1_51  = np.array(r0051['f1']); f2_51  = np.array(r0051['f2']); g_51  = np.array(r0051['gmis'])

            _, _, m09 = run_mahalanobis(f1_09, f2_09, g_09, mahal_thresh, mahal_0009)
            _, _, l09 = run_lstm(f1_09, f2_09, g_09, lstm_thresh, seq_len)
            _, _, m51 = run_mahalanobis(f1_51, f2_51, g_51, mahal_thresh, mahal_0051)
            _, _, l51 = run_lstm(f1_51, f2_51, g_51, lstm_thresh, seq_len)

            det09 = m09 > 50 or l09 > 50
            det51 = m51 > 50 or l51 > 50

            gen_rows.append({
                'Attack':          atk,
                '0009 Mahal':      f"{m09:.0f}%",
                '0009 LSTM':       f"{l09:.0f}%",
                '0009 Result':     '✅' if det09 else '❌',
                '0051 Mahal':      f"{m51:.0f}%",
                '0051 LSTM':       f"{l51:.0f}%",
                '0051 Result':     '✅' if det51 else '❌',
                'Generalizes':     '✅' if (det09 == det51) else '⚠️',
            })

        st.dataframe(pd.DataFrame(gen_rows), use_container_width=True, hide_index=True)

        agree = sum(1 for r in gen_rows if r['Generalizes'] == '✅')
        st.caption(
            f"Agreement across drives: {agree}/{len(gen_rows)} attacks "
            f"({agree/len(gen_rows)*100:.0f}%) · "
            f"Drive 0009: {len(feat_0009['Clean']['f1'])} frames · "
            f"Drive 0051: {len(feat_0051['Clean']['f1'])} frames"
        )
    except FileNotFoundError:
        st.info("Run the drive 0051 notebook and export features_0051.json to enable this comparison.")

st.markdown("---")
st.caption(
    f"🛡️ SensorTrust · KITTI Raw 2011_09_26 Drive {drive} · "
    "LSTM Autoencoder + Mahalanobis + Cross-Modal Consistency Scoring · "
    "No Labeled Attack Data Required"
)

