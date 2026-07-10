import json

with open('results/trust.json') as f:
    trust = json.load(f)

print(f"{'Scenario':<20}{'GPS':>8}{'IMU':>8}{'LiDAR':>8}{'Camera':>8}   Suspicion Order")
print("-" * 100)
for name, v in trust.items():
    t = v['trust']
    order = [s for s, _ in v['ranking']]
    print(f"{name:<20}{t['gps']:>8.3f}{t['imu']:>8.3f}{t['lidar']:>8.3f}{t['camera']:>8.3f}   {' > '.join(order)}")