from collections import defaultdict

import numpy as np


class MetricsAccumulator:
    def __init__(self) -> None:
        self.accumulator = defaultdict(lambda: [])

    def update_metric(self, metric_name, metric_value):
        self.accumulator[metric_name].append(metric_value)

    def print_average_metric(self):
        for k, v in self.accumulator.items():
            average_v = np.array(v).mean()
            print(f"{k} - {average_v:.2f}")

        self.__init__()
