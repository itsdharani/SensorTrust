import json
import numpy as np
from src.graph.disagreement_graph import build_disagreement_graph
from src.graph.disagreement_baseline import DisagreementBaseline
from src.graph.trust_score import compute_trust_scores, compute_node_inconsistency
from src.graph.ranking import rank_sensors

with open('results/zsignals.json') as f:
    zsignals = json.load(f)

# Fit baseline fresh on Clean scenario's z-signals
clean = zsignals['Clean']
clean_graph = build_disagreement_graph(
    np.array(clean['gps']), np.array(clean['imu']),
    np.array(clean['lidar']), np.array(clean['camera'])
)
baseline = DisagreementBaseline()
baseline.fit(clean_graph)

# Sanity check the fix actually loaded
print("rect_floor present?", 'rect_floor' in baseline.stats['gps_imu'])

trust_payload = {}
disagreement_payload = {}

for name, z in zsignals.items():
    graph = build_disagreement_graph(
        np.array(z['gps']), np.array(z['imu']),
        np.array(z['lidar']), np.array(z['camera'])
    )
    calibrated = baseline.normalize(graph)
    inconsistency = compute_node_inconsistency(calibrated)
    trust = compute_trust_scores(inconsistency)
    ranking = rank_sensors(inconsistency)
    trust_payload[name] = {'trust': trust, 'ranking': [[s, float(sc)] for s, sc in ranking]}
    disagreement_payload[name] = {k: float(np.nanmean(v)) for k, v in calibrated.items()}

with open('results/trust.json', 'w') as f:
    json.dump(trust_payload, f)
with open('results/disagreement.json', 'w') as f:
    json.dump(disagreement_payload, f)

print("✅ Rewrote trust.json and disagreement.json from cached zsignals.json — no re-extraction needed")