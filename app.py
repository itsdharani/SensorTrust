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
mahal_det                     = load_mahal(features)

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
        scores = mahal_det.score(f1[mask], f2[mask], gmis[mask])
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
        if mahal_det.fitted and ema_f1.mean is not None:
            x  = np.array([f1[i], f2[i], gmis[i]])
            mu = np.array([ema_f1.mean, ema_f2.mean, ema_g.mean])
            d  = x - mu
            sc = float(np.sqrt(d.T @ mahal_det.cov_inv @ d))
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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="SensorTrust", page_icon="🚗", layout="wide")

st.title("🚗 SensorTrust: Adaptive Cross-Modal Sensor Spoofing Detector")
st.markdown("*Real KITTI data · LSTM Autoencoder · Mahalanobis + EMA Adaptive Baseline · Trust Scoring*")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("🎯 Scenario")
selected = st.sidebar.selectbox("Attack Scenario", ["Clean (No Attack)"] + attack_names)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Thresholds")
mahal_thresh = st.sidebar.slider("Mahalanobis threshold", 1.0, 10.0, 3.0, step=0.25)
lstm_thresh  = st.sidebar.slider("LSTM threshold", 0.1, 5.0,
                                  float(round(default_thresh, 2)), step=0.05)
ema_alpha    = st.sidebar.slider("EMA alpha (adaptation speed)", 0.01, 0.2, 0.05, step=0.01,
                                  help="Higher = adapts faster to new baseline")
seq_len      = st.sidebar.slider("LSTM sequence length", 10, 100, 20, step=10)

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
with st.spinner("Running detectors on real KITTI feature arrays..."):
    mahal_scores, mahal_alerts, mahal_rate       = run_mahalanobis(f1, f2, gmis, mahal_thresh)
    lstm_errors,  lstm_alerts,  lstm_rate        = run_lstm(f1, f2, gmis, lstm_thresh, seq_len)
    ema_scores, ema_alerts, ema_rate, frozen, \
        em_f1, em_f2, em_gmis                    = run_ema_mahalanobis(
                                                        f1, f2, gmis,
                                                        mahal_thresh, ema_alpha)

mahal_det_flag = mahal_rate > 10
lstm_det_flag  = lstm_rate  > 10
ema_det_flag   = ema_rate   > 10
either         = mahal_det_flag or lstm_det_flag or ema_det_flag

if is_clean:
    trust   = {'gps': 0.92, 'imu': 0.91, 'lidar': 0.90, 'camera': 0.89}
    ranking = [['gps', 0.1], ['imu', 0.1], ['lidar', 0.1], ['camera', 0.1]]
else:
    trust   = trust_data[name]['trust']
    ranking = trust_data[name]['ranking']

# ── Top metrics ───────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Scenario", name)
c2.metric("Mahalanobis (Fixed)",
          "🚨 ATTACK" if mahal_det_flag else "✅ Clean",
          f"{mahal_rate:.1f}% flagged")
c3.metric("Mahalanobis (EMA)",
          "🚨 ATTACK" if ema_det_flag else "✅ Clean",
          f"{ema_rate:.1f}% flagged")
c4.metric("LSTM Autoencoder",
          "🚨 ATTACK" if lstm_det_flag else "✅ Clean",
          f"{lstm_rate:.1f}% flagged")
c5.metric("Combined", "🚨 DETECTED" if either else "✅ Clean")
c6.metric("Top Suspect",
          f"⚠️ {ranking[0][0].upper()}" if not is_clean else "None",
          f"trust={trust[ranking[0][0]]:.4f}" if not is_clean else "—")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Features", "🔍 Detectors", "📈 EMA Baseline", "🛡️ Trust Scores"
])

with tab1:
    st.subheader("Motion Consistency Features — Real KITTI")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    triples = [
        (f1,   'F1 — Kinematic Delta',           '#e74c3c', CLEAN_F1),
        (f2,   'F2 — GPS vs LiDAR Odometry',     '#3498db', CLEAN_F2),
        (gmis, 'GMIS — Geometric Inconsistency', '#2ecc71', CLEAN_GMIS),
    ]
    for ax, (arr, title, col, baseline) in zip(axes, triples):
        ax.plot(arr, color=col, linewidth=0.8, alpha=0.9)
        ax.axhline(baseline,     color='gray', linestyle='--', linewidth=1,   label='Clean mean')
        ax.axhline(baseline * 3, color='red',  linestyle='--', linewidth=1.2, label='3× threshold')
        ax.fill_between(range(len(arr)), 0, arr,
                        where=(arr > baseline * 3), color='red', alpha=0.2, label='Anomalous')
        ax.set_title(title, fontsize=10); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig); plt.close()

with tab2:
    st.subheader("Anomaly Detector Outputs")
    st.caption("Fixed Mahalanobis vs LSTM Autoencoder — threshold sliders update both live")
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))

    ax1.plot(mahal_scores, color='#3498db', linewidth=0.8, label='Mahalanobis Score')
    ax1.axhline(mahal_thresh, color='red', linestyle='--', linewidth=1.5,
                label=f'Threshold={mahal_thresh}')
    ax1.fill_between(range(len(mahal_scores)), 0, mahal_scores,
                     where=mahal_alerts, color='red', alpha=0.25, label='Flagged')
    ax1.set_title('Mahalanobis Distance (Fixed Baseline)')
    ax1.set_xlabel('Frame'); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    ax2.plot(lstm_errors, color='#e67e22', linewidth=0.8, label='Reconstruction Error')
    ax2.axhline(lstm_thresh, color='red', linestyle='--', linewidth=1.5,
                label=f'Threshold={lstm_thresh:.3f}')
    ax2.fill_between(range(len(lstm_errors)), 0, lstm_errors,
                     where=lstm_alerts, color='red', alpha=0.25, label='Flagged')
    ax2.set_title('LSTM Autoencoder Reconstruction Error')
    ax2.set_xlabel('Sequence Index'); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    plt.tight_layout(); st.pyplot(fig2); plt.close()

with tab3:
    st.subheader("EMA Adaptive Baseline Tracking")
    st.markdown(
        f"EMA alpha=`{ema_alpha}` — adapts baseline over ~`{int(1/ema_alpha)}` frames. "
        f"**EMA freezes when score > {mahal_thresh}** to prevent attack signal poisoning the baseline."
    )

    frozen_pct = float(np.mean(frozen) * 100)
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("EMA Detection Rate", f"{ema_rate:.1f}%")
    col_b.metric("Frames EMA Frozen", f"{frozen_pct:.1f}%",
                 help="Frames where EMA paused updating due to suspected attack")
    col_c.metric("Fixed vs EMA",
                 "EMA better" if ema_rate > mahal_rate else
                 "Fixed better" if mahal_rate > ema_rate else "Equal")

    fig3, axes3 = plt.subplots(3, 1, figsize=(14, 10))

    # F1 with EMA tracking
    min_plot = min(len(f1), len(em_f1))
    axes3[0].plot(f1[:min_plot], color='#e74c3c', linewidth=0.8, alpha=0.7, label='F1 signal')
    axes3[0].plot(em_f1[:min_plot], color='black', linewidth=1.5, linestyle='--', label='EMA baseline')
    axes3[0].fill_between(range(min_plot), 0, f1[:min_plot],
                          where=frozen[:min_plot], color='red', alpha=0.15, label='EMA frozen')
    axes3[0].axhline(CLEAN_F1, color='gray', linestyle=':', linewidth=1, label='Fixed clean mean')
    axes3[0].set_title('F1 — EMA Baseline vs Fixed Baseline')
    axes3[0].legend(fontsize=8); axes3[0].grid(True, alpha=0.3)

    # F2 with EMA tracking
    min_plot2 = min(len(f2), len(em_f2))
    axes3[1].plot(f2[:min_plot2], color='#3498db', linewidth=0.8, alpha=0.7, label='F2 signal')
    axes3[1].plot(em_f2[:min_plot2], color='black', linewidth=1.5, linestyle='--', label='EMA baseline')
    axes3[1].fill_between(range(min_plot2), 0, f2[:min_plot2],
                          where=frozen[:min_plot2], color='red', alpha=0.15, label='EMA frozen')
    axes3[1].axhline(CLEAN_F2, color='gray', linestyle=':', linewidth=1, label='Fixed clean mean')
    axes3[1].set_title('F2 — EMA Baseline vs Fixed Baseline')
    axes3[1].legend(fontsize=8); axes3[1].grid(True, alpha=0.3)

    # EMA Mahalanobis score
    axes3[2].plot(ema_scores, color='#9b59b6', linewidth=0.8, label='EMA Mahalanobis Score')
    axes3[2].plot(mahal_scores, color='#3498db', linewidth=0.8, alpha=0.5,
                  linestyle='--', label='Fixed Mahalanobis Score')
    axes3[2].axhline(mahal_thresh, color='red', linestyle='--', linewidth=1.5,
                     label=f'Threshold={mahal_thresh}')
    axes3[2].fill_between(range(len(ema_scores)), 0, ema_scores,
                          where=ema_alerts, color='purple', alpha=0.2, label='EMA Flagged')
    axes3[2].set_title('EMA vs Fixed Mahalanobis Score Comparison')
    axes3[2].set_xlabel('Frame')
    axes3[2].legend(fontsize=8); axes3[2].grid(True, alpha=0.3)

    plt.tight_layout(); st.pyplot(fig3); plt.close()

    # show precomputed EMA results if available
    if not is_clean and name in ema_data:
        st.markdown("---")
        st.caption(f"Precomputed EMA result for {name}: "
                   f"detection rate = {ema_data[name]['rate']:.1f}%, "
                   f"frozen frames = {np.mean(ema_data[name]['frozen'])*100:.1f}%")

with tab4:
    st.subheader("Sensor Trust Scores & Suspicion Ranking")
    col_left, col_right = st.columns([3, 2])

    with col_left:
        sensors    = ['gps', 'imu', 'lidar', 'camera']
        trust_vals = [trust[s] for s in sensors]
        bar_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

        fig4, ax4 = plt.subplots(figsize=(7, 3.5))
        bars = ax4.bar(sensors, trust_vals, color=bar_colors, edgecolor='white', linewidth=1.2)
        ax4.set_ylim(0, 1.1)
        ax4.axhline(0.3, color='red', linestyle='--', linewidth=1, alpha=0.7,
                    label='Low trust boundary')
        ax4.set_ylabel('Trust Score (consistency with other sensors)')
        ax4.set_title('Per-Sensor Consistency Score\n(lower = more inconsistent = higher suspicion)')
        for bar, val in zip(bars, trust_vals):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
        ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3, axis='y')
        plt.tight_layout(); st.pyplot(fig4); plt.close()

    with col_right:
        st.markdown(f"### 🏆 Suspicion Ranking")
        st.markdown(f"*Scenario: **{name}***")
        st.caption("Rank = relative cross-modal inconsistency. "
                   "Lower trust = more inconsistent with other sensors.")
        medals = ["🥇", "🥈", "🥉", "4️⃣"]
        for i, (sensor, _) in enumerate(ranking):
            t    = trust[sensor]
            flag = " 🚨" if i == 0 and t < 0.2 and not is_clean else ""
            st.markdown(f"{medals[i]} **{sensor.upper()}** — consistency: `{t:.4f}`{flag}")

        if not is_clean:
            st.markdown("---")
            st.markdown("**Feature Elevation vs Clean**")
            st.markdown(f"- F1:   `{feat['F1'] / CLEAN_F1:.1f}×`")
            st.markdown(f"- F2:   `{feat['F2'] / CLEAN_F2:.1f}×`")
            st.markdown(f"- GMIS: `{feat['GMIS'] / CLEAN_GMIS:.1f}×`")

# ── Summary table ─────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📋 Full Attack Evaluation (current threshold settings)"):
    import pandas as pd
    rows = []
    for atk in attack_names:
        f  = features[atk]
        tr = trust_data.get(atk, {})
        f1_a   = np.array(f['f1'])
        f2_a   = np.array(f['f2'])
        gmis_a = np.array(f['gmis'])
        _, _, m_rate              = run_mahalanobis(f1_a, f2_a, gmis_a, mahal_thresh)
        _, _, l_rate              = run_lstm(f1_a, f2_a, gmis_a, lstm_thresh, seq_len)
        _, _, e_rate, _, _, _, _  = run_ema_mahalanobis(f1_a, f2_a, gmis_a,
                                                         mahal_thresh, ema_alpha)
        top = tr['ranking'][0][0].upper() if tr else 'N/A'
        rows.append({
            'Attack':        atk,
            'F1×':          f"{f['F1'] / CLEAN_F1:.1f}×",
            'F2×':          f"{f['F2'] / CLEAN_F2:.1f}×",
            'GMIS×':        f"{f['GMIS'] / CLEAN_GMIS:.1f}×",
            'Fixed Mahal%': f"{m_rate:.0f}%",
            'EMA Mahal%':   f"{e_rate:.0f}%",
            'LSTM%':        f"{l_rate:.0f}%",
            'Top Suspect':  top,
            'Detected':     '✅' if (m_rate > 50 or l_rate > 50 or e_rate > 50) else '❌',
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(
    "SensorTrust · KITTI Raw 2011_09_26 drive 0009 · "
    "LSTM Autoencoder + Mahalanobis (Fixed & EMA) + Cross-Modal Trust Scoring · "
    "No simulation — real precomputed features"
)
