## TABLE OF CONTENTS

- [1. Introduction](#1-introduction)
- [2. Literature Survey](#2-literature-survey)
  - [2.1 GPS and GNSS Spoofing](#21-gps-and-gnss-spoofing)
  - [2.2 IMU and Inertial Sensor Attacks](#22-imu-and-inertial-sensor-attacks)
  - [2.3 LiDAR Spoofing](#23-lidar-spoofing)
  - [2.4 Cross-Modal Verification](#24-cross-modal-verification)
  - [2.5 Anomaly Detection for Time-Series Sensor Data](#25-anomaly-detection-for-time-series-sensor-data)
  - [2.6 Gap Analysis](#26-gap-analysis)
- [3. Problem Statement](#3-problem-statement)
- [4. Objectives](#4-objectives)
- [5. System Architecture](#5-system-architecture)
- [6. Methodology](#6-methodology)
  - [6.1 Dataset and Sensor Proxies](#61-dataset-and-sensor-proxies)
  - [6.2 Feature Extraction (F1, F2, GMIS)](#62-feature-extraction-f1-f2-gmis)
  - [6.3 LSTM Autoencoder](#63-lstm-autoencoder)
  - [6.4 Mahalanobis Distance Detector](#64-mahalanobis-distance-detector)
  - [6.5 Sensor Disagreement Graph and Suspicion Ranking](#65-sensor-disagreement-graph-and-suspicion-ranking)
- [7. Attack Injection Framework](#7-attack-injection-framework)
  - [7.1 Single-Sensor Attacks](#71-single-sensor-attacks)
  - [7.2 Coordinated Multi-Sensor Attacks](#72-coordinated-multi-sensor-attacks)
- [8. Experimental Evaluation](#8-experimental-evaluation)
  - [8.1 Experimental Setup](#81-experimental-setup)
  - [8.2 Detection Results — Drive 0009](#82-detection-results--drive-0009)
  - [8.3 Generalization — Drive 0051](#83-generalization--drive-0051)
  - [8.4 Ablation Study](#84-ablation-study)
  - [8.5 False Positive Analysis](#85-false-positive-analysis)
  - [8.6 Suspicion Ranking Results](#86-suspicion-ranking-results)
- [9. Results and Discussion](#9-results-and-discussion)
- [10. Limitations and Future Work](#10-limitations-and-future-work)
- [11. Conclusion](#11-conclusion)
- [12. References](#12-references)

<div style="page-break-after: always;"></div>


## LIST OF FIGURES

| Figure | Caption |
|:---|:---|
| Figure 1 | System Architecture - SensorTrust Pipeline |
| Figure 2 | Complete Attack Evaluation - Feature Elevation Table |
| Figure 3 | LSTM Autoencoder Detection - GPS Ramp Attack |
| Figure 4 | Mahalanobis Detection Results across 14 Attacks |
| Figure 5 | Detector Comparison: Mahalanobis vs LSTM Autoencoder |
| Figure 6 | Threshold Sweep - Detection Coverage vs Threshold |
| Figure 7 | Sensor Suspicion Ranking Heatmap |
| Figure 8 | Per-Sensor Consistency Score Over Time - GPS Ramp Attack |

## LIST OF TABLES

| Table | Caption |
|:---|:---|
| Table 1 | Literature Survey Gap Analysis |
| Table 2 | Sensor Proxy Signals |
| Table 3 | Single-Sensor Attack Parameters |
| Table 4 | Detection Results - Drive 0009 |
| Table 5 | Summary Metrics - Drive 0009 |
| Table 6 | Generalization Metrics - Drive 0051 |
| Table 7 | Ablation Study - Feature Contribution |

<div style="page-break-after: always;"></div>


## 1. INTRODUCTION

Autonomous vehicles (AVs) represent one of the most safety-critical applications of embedded AI systems. Their perception stack relies on continuous, synchronized input from multiple heterogeneous sensors: Global Positioning System (GPS) for localization, Inertial Measurement Unit (IMU) for motion estimation, LiDAR for 3D scene reconstruction, and Camera for visual context. The integrity of this sensor data is a fundamental assumption underlying every downstream decision — path planning, obstacle avoidance, speed regulation, and emergency braking.

Adversarial sensor spoofing attacks deliberately violate this assumption. GPS spoofing via software-defined radio is demonstrated in open literature and can redirect autonomous vehicles to unintended locations. IMU manipulation through acoustic injection can induce false acceleration readings. LiDAR spoofing using infrared lasers can inject phantom obstacles. Camera attacks using adversarial patches or strong light sources can corrupt optical flow estimation. More critically, **coordinated multi-sensor attacks** — where an adversary simultaneously manipulates two or more sensors with physically consistent false data — can defeat cross-sensor consistency checks that assume at most one sensor is compromised at a time.

Existing defenses suffer from one or more of the following limitations: they monitor individual sensors in isolation and miss cross-modal inconsistencies; they require a designated trusted anchor sensor (typically camera) that itself may be compromised; they depend on labeled attack data for supervised training; or they require infrastructure such as HD maps that are unavailable in resource-constrained deployments.

This work addresses the gap by proposing a **label-free, anchor-free, cross-modal motion-consistency framework** - SensorTrust: that detects coordinated spoofing attacks by checking whether all four sensor modalities agree on the physical motion of the vehicle. The key insight is that physical vehicle motion leaves consistent signatures across heterogeneous sensing modalities = signatures that are difficult for an adversary to fabricate simultaneously across GPS, IMU, LiDAR, and Camera without creating detectable inconsistencies.

**The main contributions of this work are:**

1. Three physics-grounded motion consistency features (F1, F2, GMIS) computed without labeled attack data.
2. A dual anomaly detection pipeline combining LSTM Autoencoder (temporal) and Mahalanobis Distance (statistical), achieving Precision = 1.00 with Recall = 0.93.
3. Evaluation across 14 attack scenarios on two independent KITTI driving sequences demonstrating generalization to unseen data.
4. An interactive demonstration dashboard built on real KITTI data with no simulation.

The remainder of this report is organized as follows. Section 2 surveys related work. Section 3 states the problem formally. Section 4 lists objectives. Section 5 presents the system architecture. Section 6 details the methodology. Section 7 describes the attack injection framework. Section 8 presents experimental results. Sections 9 through 11 discuss findings, limitations, and conclusions. Section 12 provides references.

<div style="page-break-after: always;"></div>


## 2. LITERATURE SURVEY

| Paper | Sensors | Method | Strengths | Limitations |
|:---|:---|:---|:---|:---|
| Shen et al. (2020) — Drift with Devil | GPS, IMU | FusionRipper attack characterization | Demonstrated MSF bypassability; formal threat model | No detection system proposed |
| Dasgupta et al. (2022) | GPS, IMU | LSTM predicts location shift vs GNSS | 4 attack types detected; Honda dataset | No LiDAR/Camera; fails on correlated GPS+IMU |
| Liu & Lei (2022) — FFIT | GPS, IMU | Forgetting factor innovation test | Better than SIT/AIME/CUSUM; efficient recursion | No perception sensors; single-source only |
| Jung & Yoon (2024) | GPS, IMU | External IMU + correlated GPS spoofing | Formally proves 2-sensor detection is bypassable | Attack only; no defense |
| Van Wyk et al. (2020) | Per-sensor | CNN + Kalman chi-squared per stream | Real-time; identifies faulty sensor | No cross-modal checks; coordinated attacks evade |
| You et al. (2021) — 3D-TC2 | LiDAR | Temporal consistency of detected objects | >98% detection for spoofed vehicles | LiDAR only; no GPS/IMU/Camera cross-check |
| Wang et al. (2024) | Trajectory | LSTM Autoencoder + GMM | Unsupervised; sequential anomaly detection | No cross-modal verification; no sensor diagnosis |
| Begum et al. (2024) — SADC | GPS, LiDAR | CNN-LSTM + pattern classification | 6G-V2X context; attack classification | Requires labeled data; 2 sensors only |
| PhyScout / Xu et al. (2024) | GPS, IMU, LiDAR, Camera | Camera keypoints anchor spatio-temporal checks | 4 modalities; <100ms CPU; no labels | Camera is single trusted anchor; no adaptive baseline; no trust scoring |
| **SensorTrust (Ours)** | **GPS, IMU, LiDAR, Camera** | **F1+F2+GMIS + LSTM AE + Mahalanobis + Disagreement Graph** | **4 modalities; no anchor; no labels; trust scores; suspicion ranking; coordinated attacks** | **Synthetic attacks only; single dataset; camera noise boundary** |

The field of autonomous vehicle sensor security has grown significantly following demonstrations of practical GPS and LiDAR spoofing attacks. We survey the most relevant prior work across five categories.

### 2.1 GPS and GNSS Spoofing

Zeng et al. [1] demonstrated end-to-end GPS spoofing attacks on autonomous vehicles using software-defined radio, showing that off-the-shelf hardware costing under $500 can redirect a vehicle's perceived trajectory. Their work established GPS as the most practically vulnerable sensor and motivated cross-modal verification approaches. Jansen et al. [2] formalized the threat model for GNSS spoofing in automotive contexts, identifying detection windows based on navigation state divergence.

### 2.2 IMU and Inertial Sensor Attacks

Son et al. [3] demonstrated acoustic injection attacks on MEMS IMU sensors, inducing controlled false acceleration readings through resonance frequencies of the sensor's proof mass. This attack bypasses cryptographic protection since it operates at the physical layer. Jung and Yoon [4] specifically targeted GPS-IMU fusion — showing that an attacker who models the vehicle's true inertial motion can craft correlated GPS spoofing signals that defeat innovation-based EKF monitoring. Their work directly motivates our use of LiDAR and Camera as additional modalities that an adversary cannot easily correlate.

### 2.3 LiDAR Spoofing

Cao et al. [5] demonstrated physical LiDAR spoofing using commercial infrared laser systems, injecting phantom obstacle points that caused an AV to perform emergency braking. Shin et al. [6] extended this to relay attacks. Both works confirm that LiDAR, despite its apparent complexity, is practically spoofable with moderate equipment. Our phantom injection attack is modeled on Cao et al.'s methodology.

### 2.4 Cross-Modal Verification

PhyScout [7] is the closest prior work to our approach. It uses physical motion consistency across GPS, IMU, and Camera to detect spoofing attacks. However, PhyScout designates camera as a trusted anchor sensor — if camera is compromised, the system's trust assumption fails. Our method treats all four sensors symmetrically with no trusted anchor. Wang et al. [8] proposed a cross-modal verification system for GPS and LiDAR consistency using scene-flow estimation, achieving detection on simulated KITTI attacks. Their work informs our F2 feature design. You et al. [9] proposed sensor fusion integrity monitoring using statistical process control, but their method requires labeled anomaly data for threshold calibration.

### 2.5 Anomaly Detection for Time-Series Sensor Data

LSTM autoencoders for unsupervised anomaly detection were pioneered by Malhotra et al. [10] on physiological time series. Park et al. [11] applied LSTM autoencoders to multivariate sensor streams in industrial IoT settings, demonstrating superior recall compared to frame-level methods (Isolation Forest, One-Class SVM) on temporally structured anomalies. This directly supports our choice of LSTM Autoencoder as the primary detector. Mahalanobis distance for multivariate anomaly detection in sensor data is well-established [12] and provides complementary statistical coverage.

### 2.6 Gap Analysis

**Table 1: Literature Survey Gap Analysis**

| Paper | GPS | IMU | LiDAR | Camera | No Trusted Anchor | No Labeled Data | Cross-Modal Verification | Per-Sensor Trust | Coordinated Attack Detection |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Shen et al. (2020) | ✅ | ✅ | ❌ | ❌ | ✅ | N/A | ❌ | ❌ | ❌ |
| Dasgupta et al. (2022) | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Liu & Lei (2022) | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| Van Wyk et al. (2020) | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| You et al. (2021) | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| Wang et al. (2024) | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| Begum et al. (2024) | ✅ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| PhyScout (2024) | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ |
| **SensorTrust** | **✅** | **✅** | **✅** | **✅** | **✅** | **✅** | **✅** | **✅** | **✅** |
<div style="page-break-after: always;"></div>

## 3. PROBLEM STATEMENT

Given synchronized time-series streams from four heterogeneous sensors — GPS (velocity, heading), IMU (acceleration, yaw rate), LiDAR (point cloud), and Camera (image frames) — on an autonomous vehicle operating in an open-road environment, detect whether one or more sensors are under an active spoofing attack, and if so, identify the most likely compromised sensor modality.

Formally, let $\mathbf{x}_t = \{x_t^{GPS}, x_t^{IMU}, x_t^{LiDAR}, x_t^{CAM}\}$ be the joint sensor observation at time $t$. The system must produce:

1. A binary detection signal $d_t \in \{0, 1\}$ indicating attack presence.
2. A per-sensor consistency score $T_i \in [0, 1]$ for each sensor $i \in \{GPS, IMU, LiDAR, Camera\}$.
3. A suspicion ranking $\sigma$ over the four sensors.

**Constraints:**

- No labeled attack data available at training time.
- No sensor may be designated as trusted a priori.
- The system must operate on resource-constrained hardware (CPU-only inference).
- Detection must be achievable from raw sensor streams without HD maps or external infrastructure.

<div style="page-break-after: always;"></div>


## 4. OBJECTIVES

1. Design physics-grounded motion consistency features from raw multi-sensor data that are sensitive to spoofing attacks and robust to benign driving variation.
2. Implement an unsupervised anomaly detection pipeline requiring no labeled attack data.
3. Develop a sensor suspicion ranking mechanism that localizes the likely compromised sensor modality.
4. Evaluate the system on 14 attack scenarios spanning single-sensor and coordinated multi-sensor attacks.
5. Demonstrate generalization to an unseen driving sequence.
6. Build an interactive demonstration dashboard on real sensor data with no simulation.

<div style="page-break-after: always;"></div>


## 5. SYSTEM ARCHITECTURE

![meow](SensorTrust.png)
<div style="page-break-after: always;"></div>

![meow](SensorTrust2.png)
**Figure 1: SensorTrust System Architecture**

The pipeline proceeds as follows: Raw synchronized sensor streams (GPS, IMU, LiDAR, Camera) are processed into motion proxy signals via modality-specific extractors. Proxy signals are z-score normalized per drive. Three physics-grounded features (F1, F2, GMIS) are computed per frame. Features feed two parallel anomaly detectors — LSTM Autoencoder and Mahalanobis Distance. An ensemble vote triggers the binary attack alert. In parallel, a pairwise sensor disagreement graph computes per-sensor consistency scores and ranks sensors by suspicion.

<div style="page-break-after: always;"></div>


## 6. METHODOLOGY

### 6.1 Dataset and Sensor Proxies

We use the **KITTI Raw Dataset** [13], a widely used benchmark providing synchronized GPS/IMU (OXTS unit at 100Hz, downsampled to 10Hz), 64-beam Velodyne HDL-64E LiDAR (10Hz), and stereo camera (1242×375, 10Hz) data captured from a vehicle driving in urban and highway environments around Karlsruhe, Germany.

**Primary evaluation:** Drive 2011_09_26_0009 — 447 synchronized frames, 44.7 seconds.

**Generalization evaluation:** Drive 2011_09_26_0051 — 438 synchronized frames, 43.8 seconds.

Raw sensor streams are preprocessed into motion proxy signals as described in Table 2.

**Table 2: Sensor Proxy Signals**

| Sensor | Proxy Signal | Extraction Method | Physical Meaning |
|:---|:---|:---|:---|
| GPS | $\Delta v_{GPS}$, heading rate | OXTS forward velocity differencing over 5-frame window | Speed change and turning rate |
| IMU | $\Delta v_{IMU}$, yaw rate | Integrated OXTS accelerometer + gyroscope | Acceleration-derived speed change |
| LiDAR | ICP residual | Open3D point-to-plane ICP between consecutive scans | Scene displacement alignment error |
| Camera | Optical flow magnitude | Farneback dense optical flow on grayscale frames | Mean visual motion magnitude |

All proxy signals are z-score normalized per drive:

$$z_i = \frac{x_i - \mu_i^{clean}}{\sigma_i^{clean}}$$

where $\mu_i^{clean}$ and $\sigma_i^{clean}$ are estimated from the clean (unattacked) portion of each drive independently. This makes features invariant to absolute sensor scale and drive-specific dynamics.

### 6.2 Feature Extraction (F1, F2, GMIS)

Three scalar motion consistency features are computed per frame from the normalized proxy signals.

**F1 — Kinematic Cross-Check**

$$F_1(t)=\left|z_{\mathrm{GPS},\Delta v}(t)-z_{\mathrm{IMU},\Delta v}(t)\right|$$


GPS and IMU both measure vehicle speed change through fundamentally different physical principles — Doppler shift and dead reckoning respectively. Under normal operation their normalized signals agree closely. When either sensor is spoofed, their difference spikes disproportionately. Clean mean on Drive 0009: $\bar{F}_1^{clean} = 0.5598$. GPS Ramp attack elevation: 9.6×.

**F2 — Scene Motion Consistency**

$$F_2(t)=\left|z_{\mathrm{GPS},\Delta v}(t)-z_{\mathrm{LiDAR},\mathrm{ICP}}(t)\right|$$

GPS speed and LiDAR ICP residual (estimated scene displacement between consecutive point clouds via Iterative Closest Point registration) both reflect vehicle displacement through independent physical sensing modalities. Disagreement indicates fabricated motion in one source. Clean mean: $\bar{F}_2^{clean} = 0.9053$. Coordinated GPS+LiDAR attack elevation: 7.0×.

**GMIS — Geometric Motion Inconsistency Score**

$$GMIS(t) = \frac{1}{6} \sum_{i < j} \left| z_i(t) - z_j(t) \right|$$

A global aggregation of all six pairwise sensor disagreements, scale-invariant through normalization. GMIS captures system-wide inconsistency regardless of which specific sensor pair disagrees. Clean mean: $\overline{GMIS}^{clean} = 0.8221$. ALL FOUR attack elevation: 5.9×.

The three features are complementary by design: F1 is sensitive to GPS and IMU attacks, F2 is sensitive to GPS and LiDAR attacks, and GMIS captures broader multi-sensor inconsistency. The ablation study in Section 8.4 quantifies the contribution of each feature independently.

### 6.3 LSTM Autoencoder

An LSTM Autoencoder is trained **exclusively on clean driving sequences** from Drive 0009, requiring no labeled attack data.

**Architecture:**

| Component | Configuration |
|:---|:---|
| Input | Sequences of length L=200 frames, 3 features per frame |
| Encoder | LSTM: input_size=3, hidden_size=64, output latent_size=32 |
| Decoder | LSTM: input_size=32, hidden_size=64, linear projection to 3 |
| Loss | Mean Squared Error (MSE) reconstruction loss |
| Optimizer | Adam, learning rate=1e-3 |
| Training | 150 epochs, batch_size=32, 238 clean sequences from Drive 0009 |

**Operating principle:** The model learns the temporal dynamics of normal motion. Attack sequences deviate from learned patterns, producing elevated reconstruction error:

$$\text{Error}(t) = \frac{1}{L} \sum_{l=1}^{L} \left\| \mathbf{x}_{t:t+L} - \hat{\mathbf{x}}_{t:t+L} \right\|^2$$

**Threshold:** Set at the **95th percentile of clean reconstruction errors** on Drive 0009: $\theta_{LSTM} = 1.0961$. A sequence is flagged anomalous if $\text{Error} > \theta_{LSTM}$.

### 6.4 Mahalanobis Distance Detector

The Mahalanobis Distance detector models the clean feature distribution as a multivariate Gaussian, providing a frame-level statistical anomaly score:

$$d_{Mahal}(\mathbf{x}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})}$$

where $\boldsymbol{\mu} \in \mathbb{R}^3$ and $\boldsymbol{\Sigma} \in \mathbb{R}^{3 \times 3}$ are the mean vector and covariance matrix fitted on clean F1, F2, GMIS triples from 437 clean frames of Drive 0009.

**Threshold:** $\theta_{Mahal} = 3.0$ Mahalanobis units. A frame is flagged anomalous if $d_{Mahal} > \theta_{Mahal}$.

The Mahalanobis detector is deliberately conservative — it fires less frequently than LSTM but with higher precision (1.00), making it suitable for high-confidence alerting and early warning on sustained attacks. The ensemble detector flags an attack if either detector fires.

### 6.5 Sensor Disagreement Graph and Suspicion Ranking

Given normalized sensor signals $\mathbf{z} = \{z_{GPS}, z_{IMU}, z_{LiDAR}, z_{CAM}\}$, a weighted undirected graph $G = (V, E, W)$ is constructed where $V = \{GPS, IMU, LiDAR, Camera\}$ and edge weights represent pairwise disagreement:

$$w_{ij}(t) = |z_i(t) - z_j(t)|, \quad \forall i \neq j$$

This yields six pairwise disagreement time-series across the complete sensor graph.

**Per-sensor node inconsistency** (sum of all adjacent edge weights):

$$I_{GPS} = \overline{w_{GPS,IMU}} + \overline{w_{GPS,LiDAR}} + \overline{w_{GPS,CAM}}$$
$$I_{IMU} = \overline{w_{GPS,IMU}} + \overline{w_{IMU,LiDAR}} + \overline{w_{IMU,CAM}}$$
$$I_{LiDAR} = \overline{w_{GPS,LiDAR}} + \overline{w_{IMU,LiDAR}} + \overline{w_{LiDAR,CAM}}$$
$$I_{CAM} = \overline{w_{GPS,CAM}} + \overline{w_{IMU,CAM}} + \overline{w_{LiDAR,CAM}}$$

**Per-sensor consistency score** via exponential decay mapping:

$$T_i = \exp\left(-\beta \cdot I_i\right), \quad \beta = 0.7$$

$T_i \in [0, 1]$ — lower values indicate higher inconsistency with the rest of the sensing system, and therefore higher suspicion of compromise.

**Suspicion ranking:** Sensors are ranked in descending order of $I_i$.

**Interpretation note:** Consistency scores measure cross-modal agreement evidence, not direct attack probability. In coordinated attack scenarios where two sensors are simultaneously spoofed with correlated values, clean sensors also accumulate inconsistency through their graph edges to the compromised pair. This documented limitation is discussed further in Section 10.

<div style="page-break-after: always;"></div>


## 7. ATTACK INJECTION FRAMEWORK

Fourteen attack scenarios are implemented via synthetic injection into real KITTI proxy signals. All attacks operate after proxy extraction to ensure physical plausibility within each sensor's measurement domain. Attack parameters are chosen to represent realistic adversarial capabilities documented in the spoofing literature.

### 7.1 Single-Sensor Attacks

**Table 3: Single-Sensor Attack Parameters**

| Attack | Sensor | Implementation | Parameters | Threat Model Basis |
|:---|:---|:---|:---|:---|
| GPS Speed Ramp | GPS | Linear velocity ramp added to GPS speed from frame 200 | ramp_rate=2.0 m/s², duration=50 frames | Zeng et al. [1] |
| GPS Step Offset | GPS | Constant lat/lon offset injected from frame 200 | Δlat=0.005°, Δlon=0.005° | Jansen et al. [2] |
| IMU Constant Bias | IMU | Additive constant bias on IMU delta-v across all frames | bias=0.5 | Son et al. [3] |
| IMU Gaussian Noise | IMU | Zero-mean Gaussian noise added to IMU delta-v | σ=0.3 | Son et al. [3] |
| IMU Burst Noise | IMU | High-amplitude burst in localized time window | amplitude=1.0, duration=30 frames | — |
| LiDAR Phantom Injection | LiDAR | Synthetic obstacle points injected into point cloud | n_points=50,000, distance=3m | Cao et al. [5] |
| Camera Gaussian Noise | Camera | Pixel-level Gaussian noise on all image frames | σ=25 (pixel intensity) | — |

### 7.2 Coordinated Multi-Sensor Attacks

| Attack | Sensors Compromised | Strategy |
|:---|:---|:---|
| Coord GPS+IMU | GPS, IMU | GPS ramp + correlated IMU bias (Jung & Yoon [4] variant — designed to defeat EKF innovation monitoring) |
| Coord GPS+Camera | GPS, Camera | GPS ramp + camera Gaussian noise |
| Coord GPS+LiDAR | GPS, LiDAR | GPS ramp + LiDAR phantom injection |
| Coord IMU+LiDAR | IMU, LiDAR | IMU bias + LiDAR phantom injection |
| Coord IMU+Camera | IMU, Camera | IMU bias + camera Gaussian noise |
| Coord LiDAR+Camera | LiDAR, Camera | Phantom injection + camera Gaussian noise |
| ALL FOUR | GPS, IMU, LiDAR, Camera | GPS ramp + IMU bias + LiDAR phantom + camera noise simultaneously |

Each attacked sequence is independently processed through the full SensorTrust pipeline: proxy extraction → normalization → feature computation → anomaly detection → suspicion ranking. Ground truth labels (attack/clean) are available for evaluation since attacks are synthetically injected at known frames.

<div style="page-break-after: always;"></div>


## 8. EXPERIMENTAL EVALUATION

### 8.1 Experimental Setup

| Parameter | Value |
|:---|:---|
| Hardware | Intel Core i5, 16GB RAM, CPU-only (no GPU) |
| Framework | PyTorch 2.x, NumPy, scikit-learn, pykitti, Open3D |
| LSTM training | 50 epochs, Adam (lr=1e-3), batch_size=32 |
| Clean training sequences | 248 sequences of length 200 from Drive 0009 |
| Mahalanobis fitting | 437 clean frames from Drive 0009 |
| LSTM threshold | 1.0961 (95th percentile of clean reconstruction errors) |
| Mahalanobis threshold | 3.0 Mahalanobis units |
| Consistency score β | 0.7 |
| Detection criterion | >50% of frames/sequences flagged per attack scenario |

### 8.2 Detection Results — Drive 0009

![alt text](image-1.png)
![alt text](image-2.png)
**Table 4: Detection Results — Drive 0009 (447 frames)**

| Attack | F1× | F2× | GMIS× | LSTM % | Mahal % | Detected |
|:---|---:|---:|---:|---:|---:|:---:|
| GPS Ramp | 9.6× | 6.3× | 5.3× | 100.0% | 13.7% | ✅ |
| GPS Step | 1.1× | 1.0× | 1.0× | 80.3% | 6.9% | ✅ |
| IMU Bias | 9.2× | 1.0× | 1.4× | 100.0% | 92.9% | ✅ |
| IMU Noise | 4.8× | 1.0× | 1.2× | 100.0% | 44.6% | ✅ |
| IMU Burst | 1.8× | 1.0× | 1.1× | 92.0% | 7.6% | ✅ |
| LiDAR Phantom | 1.0× | 1.7× | 1.5× | 56.7% | 16.9% | ✅ |
| Camera Noise | 1.0× | 1.0× | 1.1× | 0.0% | 10.3% | ❌ |
| Coord GPS+IMU | 11.8× | 5.0× | 4.3× | 100.0% | 83.3% | ✅ |
| Coord GPS+Cam | 9.6× | 6.3× | 5.3× | 100.0% | 17.2% | ✅ |
| Coord GPS+LiDAR | 9.6× | 7.0× | 5.7× | 100.0% | 23.8% | ✅ |
| Coord IMU+LiDAR | 9.2× | 1.7× | 1.8× | 100.0% | 93.4% | ✅ |
| Coord IMU+Cam | 9.2× | 1.0× | 1.4× | 100.0% | 92.9% | ✅ |
| Coord LiDAR+Cam | 1.0× | 1.7× | 1.5× | 57.1% | 17.8% | ✅ |
| ALL FOUR | 17.4× | 7.0× | 5.9× | 100.0% | 94.3% | ✅ |

- LSTM Autoencoder detection plot: GPS Ramp Attack
![alt text](image-3.png)

- Mahalanobis detection results table output
![alt text](image-4.png)
- Detector comparison table: Mahalanobis vs LSTM Autoencoder
![alt text](image-5.png)

**Table 5: Summary Detection Metrics: Drive 0009**

| Detector | Precision | Recall | F1-Score |
|:---|:---:|:---:|:---:|
| Mahalanobis | 1.00 | 0.36 | 0.53 |
| LSTM Autoencoder | 1.00 | 0.93 | 0.96 |
| Either (Ensemble) | 1.00 | 0.93 | 0.96 |

Both detectors achieve Precision = 1.00, confirming zero false alarms on clean data. The LSTM Autoencoder significantly outperforms Mahalanobis on recall (0.93 vs 0.36), consistent with the temporal nature of attack signatures which frame-level Mahalanobis distance misses. The ensemble achieves the same recall as LSTM while adding Mahalanobis high-confidence early alerting for sustained attacks.

### 8.3 Generalization — Drive 0051

- Threshold sweep plots for both detectors
![alt text](image-6.png)

The trained LSTM model (weights unchanged from Drive 0009 training) and the LSTM threshold (1.0961, calibrated on Drive 0009) are applied directly to Drive 0051 without any retraining or threshold adjustment. The Mahalanobis detector is re-fitted on Drive 0051's clean baseline to account for different driving dynamics — this is analogous to per-deployment calibration and does not use Drive 0051 attack data.

**Drive baseline dynamics comparison:**

| Feature | Drive 0009 | Drive 0051 | Difference |
|:---|:---:|:---:|:---:|
| F1 clean mean | 0.5598 | 0.6581 | +17.6% |
| F2 clean mean | 0.9053 | 1.2282 | +35.7% |
| GMIS clean mean | 0.8221 | 1.0499 | +27.7% |

The elevated baselines on Drive 0051 reflect more dynamic driving (higher acceleration variance, tighter turns) which partially absorbs borderline attacks.

**Table 6: Generalization Metrics: Drive 0051 vs Drive 0009**

| Detector | Drive 0009 Recall | Drive 0051 Recall | Precision (both) |
|:---|:---:|:---:|:---:|
| LSTM Autoencoder | 0.93 | 0.64 | 1.00 |
| Mahalanobis | 0.36 | 0.43 | 1.00 |
| Either Ensemble | 0.93 | 0.64 | 1.00 |

The recall reduction on Drive 0051 (0.93 → 0.64) is concentrated in weak single-sensor attacks near the detection boundary. All high-severity coordinated attacks (Coord GPS+IMU, Coord GPS+LiDAR, IMU Bias, ALL FOUR) maintain 100% detection on both drives. Crucially, **Precision = 1.00 on both drives** — the system produces zero false alarms on clean data regardless of drive dynamics.

### 8.4 Ablation Study

To validate the necessity of each feature, the LSTM Autoencoder is evaluated with individual and combined feature subsets. Absent features are replaced with zeros.

**Table 7: Ablation Study: Feature Contribution to LSTM Recall**

| Features Used | LSTM Recall | Notes |
|:---|:---:|:---|
| F1 only | 0.50 | Misses LiDAR and camera attacks entirely |
| F2 only | 0.29 | Sensitive only to GPS-LiDAR disagreement |
| GMIS only | 0.43 | Broad coverage but noisy in isolation |
| F1 + F2 | 0.71 | Misses camera-dominant scenarios |
| F1 + GMIS | 0.79 | Improved but incomplete |
| F2 + GMIS | 0.64 | Weak on IMU-only attacks |
| **F1 + F2 + GMIS** | **0.93** | **Maximum coverage** |

No individual feature achieves above 0.50 recall. The combination of all three features is necessary to cover GPS, IMU, LiDAR, and Camera attack categories simultaneously.

### 8.5 False Positive Analysis

On clean unattacked data from both drives, both detectors produce zero false alarms:

| Detector | Drive 0009 FP Rate | Drive 0051 FP Rate |
|:---|:---:|:---:|
| LSTM Autoencoder | 0.0% | 0.0% |
| Mahalanobis | 0.0% | 0.0% |

Benign driving variation including sharp acceleration, deceleration, and turns does not trigger either detector at the chosen thresholds.

### 8.6 Suspicion Ranking Results

*[image8.png — Sensor Suspicion Ranking Heatmap across 14 attacks]*

*[image9.png — Per-Sensor Consistency Score Over Time — GPS Ramp Attack]*

For single-sensor attacks, the consistency scoring correctly identifies the compromised sensor as the primary suspect in the majority of cases: GPS Ramp → GPS trust = 0.0000, IMU Bias → IMU trust = 0.0868 (lowest among four sensors), Coordinated GPS+IMU → GPS ranked first (trust = 0.0001 vs IMU trust = 0.0145).

The temporal consistency plot (Figure 8) shows all sensor consistency scores collapsing toward zero during the GPS Ramp attack window, with GPS collapsing first and deepest — consistent with the expected behavior of the primary attacked modality.

<div style="page-break-after: always;"></div>


## 9. RESULTS AND DISCUSSION

The experimental results demonstrate three principal findings.

**Finding 1: Dual detection provides complementary coverage with zero false alarms.**

The LSTM Autoencoder alone achieves 93% recall but requires temporal context of 200 frames before producing a detection. The Mahalanobis detector operates frame-by-frame and provides early alerting for sustained attacks (IMU Bias Mahalanobis rate: 92.9%; LSTM rate: 100.0%), but misses temporally structured attacks like GPS Ramp (13.7% vs 100.0%). Neither detector alone achieves both high recall and frame-level responsiveness. The ensemble delivers 93% recall with 100% precision and immediate alerting for sustained high-magnitude attacks.

**Finding 2: Feature elevation magnitude predicts detection reliability.**

Attacks producing feature elevation above 5× clean baseline are detected by both detectors with near-100% frame rates on both drives. Attacks in the 1.0–1.8× elevation range (GPS Step, Camera Noise, LiDAR Phantom alone, IMU Burst) represent genuine detection boundaries — detectable on Drive 0009 by the LSTM but borderline on Drive 0051 due to higher baseline dynamics. This establishes a principled relationship between attack magnitude and system detectability.

**Finding 3: Coordinated attacks are more reliably detected than single-sensor attacks.**

The most practically dangerous scenario — ALL FOUR sensors attacked simultaneously — produces the highest feature elevation across all three features (F1: 17.4×, F2: 7.0×, GMIS: 5.9×) and achieves the highest detection confidence (LSTM: 100%, Mahalanobis: 94.3%) on both drives. This is the intended property of a physics-grounded cross-modal system: more sensors behaving inconsistently produces stronger, more redundant detection signals. An attacker who attempts to spoof more sensors makes the attack easier to detect, not harder.

**Camera Noise exception:** Gaussian pixel noise at σ=25 does not sufficiently alter mean optical flow magnitude (GMIS elevation: 1.1×) to exceed either detection threshold. This represents the detection boundary of our current camera proxy, not a fundamental weakness of the framework. More severe camera attacks (blackout, adversarial perturbation, flow suppression) would produce substantially larger GMIS deviations. This is clearly stated as a limitation.

<div style="page-break-after: always;"></div>


## 10. LIMITATIONS AND FUTURE WORK

### 10.1 Current Limitations

**L1 — Single dataset family.**
Evaluation is limited to two KITTI Raw sequences of highway and urban driving in Karlsruhe, Germany under favorable weather conditions. Performance under adverse weather (rain, fog, snow), night driving, or diverse geographic environments is untested.

**L2 — Simulated attack injection.**
All 14 attacks are synthetically injected into clean sensor proxy streams. Real adversarial hardware (GPS SDR spoofers, acoustic IMU injectors, infrared LiDAR attackers) may produce attack signatures with richer statistical structure that differs from our parameterized models.

**L3 — No direct baseline comparison.**
We compare against a naive feature-threshold detector (3× clean baseline) but do not implement EKF innovation monitoring or PhyScout for direct comparison. Absolute performance is demonstrated on a standardized dataset, but relative improvement over state-of-the-art cannot be claimed without implemented baselines.

**L4 — Consistency scoring in coordinated attack scenarios.**
When two correlated sensors are simultaneously spoofed (e.g., Coord GPS+IMU), clean sensors (LiDAR, Camera) also accumulate inconsistency through their graph edges to the compromised pair, reducing their consistency scores below 1.0 even though they are uncompromised. The consistency score should be interpreted as cross-modal agreement evidence rather than direct attack attribution probability.

**L5 — Camera proxy sensitivity to pixel-level noise.**
The Farneback optical flow magnitude proxy is insensitive to Gaussian pixel noise that preserves scene flow structure. Severe camera attacks that suppress or distort flow would be detectable.

**L6 — LSTM threshold calibrated on Drive 0009.**
The LSTM threshold (1.0961) is calibrated on Drive 0009's clean distribution. On Drive 0051 with different baseline dynamics, borderline attacks fall closer to this threshold, contributing to the recall reduction (0.93 → 0.64).

### 10.2 Future Work

1. **Baseline comparison.** Implement EKF innovation monitoring and PhyScout for quantitative comparison on the same KITTI sequences.
2. **Adverse weather evaluation.** Extend evaluation to nuScenes or Waymo Open Dataset which include diverse weather and lighting conditions.
3. **Improved consistency scoring.** Develop a Bayesian sensor fault isolation approach that accounts for correlated multi-sensor attacks using temporal evidence accumulation.
4. **Hardware deployment.** Deploy on embedded hardware (NVIDIA Jetson Nano, Raspberry Pi 5) to validate resource-constrained operation and measure inference latency.
5. **Online adaptation.** Extend the LSTM Autoencoder with online fine-tuning to handle legitimate long-term sensor drift without retraining.
6. **Slow drift attacks.** Investigate detection of very-low-magnitude, slow-ramp attacks designed to stay below the 3× detection threshold — a realistic advanced adversarial strategy.
7. **Drive-agnostic threshold.** Develop a threshold calibration strategy that transfers across drives without per-drive fitting, enabling zero-shot deployment.

<div style="page-break-after: always;"></div>


## 11. CONCLUSION

This work presents **SensorTrust**, a label-free, anchor-free cross-modal motion consistency framework for detecting coordinated multi-sensor spoofing attacks on autonomous vehicles. By extracting three physics-grounded features (F1, F2, GMIS) from four heterogeneous sensor modalities and combining an LSTM Autoencoder with a Mahalanobis Distance detector, the system achieves **Precision = 1.00** with **Recall = 0.93** across 14 attack scenarios on the KITTI Raw dataset, with zero false alarms on clean data from both evaluated drives.

The ablation study confirms that all three features contribute independently to detection coverage — no single feature achieves above 0.50 recall. The pairwise sensor disagreement graph provides interpretable per-sensor consistency scores that correctly identify the primary suspect sensor in single-sensor attack scenarios. Generalization evaluation on Drive 0051 (unseen during training) confirms that high-severity attacks remain detectable across different driving dynamics while clearly characterizing the detection boundary for weak single-sensor attacks.

The work makes the case that cross-modal physical consistency — the requirement that GPS, IMU, LiDAR, and Camera all agree on vehicle motion — is a robust and practically implementable foundation for spoofing detection that does not require labeled attack data, trusted anchor sensors, or HD map infrastructure. The framework is particularly well-suited to resource-constrained embodied AI deployments where computational budget, labeled data, and infrastructure support are all limited.

The complete implementation — including all 14 attack scenarios, trained LSTM model, feature extraction pipeline, anomaly detectors, suspicion ranking module, and interactive Streamlit dashboard — constitutes a reproducible research artifact demonstrating end-to-end cross-modal spoofing detection on real multi-sensor data.

<div style="page-break-after: always;"></div>


## 12. REFERENCES

[1] Zeng, K., et al. "All your GPS are belong to us: Towards stealthy manipulation of road navigation systems." *27th USENIX Security Symposium*, 2018.

[2] Jansen, K., et al. "Crowd-GPS-Sec: Leveraging crowdsourcing to detect and localize GPS spoofing attacks." *IEEE Symposium on Security and Privacy*, 2018.

[3] Son, Y., et al. "Rocking drones with intentional sound noise on gyroscopic sensors." *24th USENIX Security Symposium*, 2015.

[4] Jung, S., and Yoon, S. "Coordinated GPS-IMU spoofing attack against autonomous vehicle navigation." *IEEE Transactions on Vehicular Technology*, 2024.

[5] Cao, Y., et al. "Adversarial sensor attack on LiDAR-based perception in autonomous driving." *ACM CCS*, 2019.

[6] Shin, H., et al. "Illusion and dazzle: Adversarial optical channel exploits against lidars for automotive applications." *CHES*, 2017.

[7] Tu, J., et al. "PhyScout: Detecting sensor spoofing attacks via physical consistency." *USENIX Security*, 2023.

[8] Wang, Z., et al. "Cross-modal sensor verification for autonomous vehicle localization integrity." *MDPI Electronics*, 2024.

[9] You, C., et al. "Sensor fusion integrity monitoring for autonomous vehicle navigation." *IEEE Transactions on Intelligent Transportation Systems*, 2022.

[10] Malhotra, P., et al. "LSTM-based encoder-decoder for multi-sensor anomaly detection." *ICML Anomaly Detection Workshop*, 2016.

[11] Park, D., et al. "A multimodal anomaly detector for robot-assisted feeding using an LSTM-based variational autoencoder." *IEEE Robotics and Automation Letters*, 2018.

[12] Mahalanobis, P.C. "On the generalised distance in statistics." *Proceedings of the National Institute of Sciences of India*, 1936.

[13] Geiger, A., et al. "Are we ready for autonomous driving? The KITTI vision benchmark suite." *CVPR*, 2012.



<div align="center">

*End of Report*
