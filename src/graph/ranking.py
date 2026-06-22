import numpy as np

def rank_sensors(inconsistency):

    ranking = sorted(
        [
            (sensor, np.nanmean(score))
            for sensor, score in inconsistency.items()
        ],
        key=lambda x: x[1],
        reverse=True
    )

    return ranking