"""A balanced calibration/test protocol. Only calibration runs inform the prediction."""
from dataclasses import dataclass, field
import math
from model import WIDE, Route

# Each four-run block contains two runs of each route, with both start orders.
ORDER = ('shortcut', 'wide', 'wide', 'shortcut', 'wide', 'shortcut', 'shortcut', 'wide')


@dataclass
class Run:
    route: str
    error_ms: float | None
    elapsed: float
    clipped: bool


@dataclass
class Experiment:
    shortcut: Route
    calibration: list[Run] = field(default_factory=list)
    trials: list[Run] = field(default_factory=list)
    prediction: float | None = None   # wide mean minus shortcut mean

    @property
    def stage(self):
        if len(self.calibration) < len(ORDER):
            return 'calibration'
        if len(self.trials) < len(ORDER):
            return 'test'
        return 'complete'

    @property
    def next_route(self):
        if self.stage == 'complete':
            return None
        if self.stage == 'calibration':
            return ORDER[len(self.calibration)]
        return ORDER[::-1][len(self.trials)]

    def record(self, run):
        if run.route != self.next_route or self.stage == 'complete':
            raise ValueError('Run does not match the next scheduled route.')
        if self.stage == 'calibration':
            self.calibration.append(run)
            if len(self.calibration) == len(ORDER):
                means = {}
                for name, route in [('wide', WIDE), ('shortcut', self.shortcut)]:
                    errors = [math.inf if r.error_ms is None else r.error_ms
                              for r in self.calibration if r.route == name]
                    means[name] = float(route.times(errors).mean())
                self.prediction = means['wide'] - means['shortcut']
        else:
            self.trials.append(run)

    def measured_gain(self):
        if self.stage != 'complete':
            return None
        def mean(name):
            times = [r.elapsed for r in self.trials if r.route == name]
            return sum(times) / len(times)
        return mean('wide') - mean('shortcut')
