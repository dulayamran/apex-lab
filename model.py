"""Track geometry, timing physics and inverse design. Distances are in game units."""
from dataclasses import dataclass
from math import cos, dist, hypot, radians, sin, tan
from statistics import NormalDist
import numpy as np

SPEED = 90.0
TURN_X, FINISH_Y = 360.0, 245.0
KART_RADIUS = 3.2
CLIP_DELAY = 2.0
CUE_LEAD = 55.0
LATE_LIMIT_MS = 500.0
SIGMAS = np.array([0., 50., 100., 150., 200.])
DEFAULT_TARGET = np.array([0.50, 0.42, 0., -0.40, -0.67])
SIM_SEED = 417
SIM_RUNS = 6000
Point = tuple[float, float]


@dataclass(frozen=True)
class Route:
    cut: float           # distance before the outside corner at which the diagonal starts
    angle: float         # degrees up from the approach straight
    width: float         # full size of gate opening, measured across the diagonal

    @property
    def theta(self):
        return radians(self.angle)

    @property
    def launch(self):
        return (TURN_X - self.cut, 0.0)

    @property
    def exit(self):
        return (TURN_X, self.cut * tan(self.theta))

    @property
    def gate(self):
        return ((self.launch[0] + TURN_X) / 2, self.exit[1] / 2)

    @property
    def normal(self):
        return (sin(self.theta), -cos(self.theta))

    @property
    def cue_time(self):
        return (self.launch[0] - CUE_LEAD) / SPEED

    @property
    def clearance(self):
        return self.width / 2 - KART_RADIUS

    def displacement(self, errors_ms):
        # Timing error projected perpendicular to the diagonal. Once a kart hits
        # a barrier it follows the recovery line, limiting the displayed offset.
        errors = np.asarray(errors_ms)
        effective = np.where(errors > LATE_LIMIT_MS, np.inf, errors)
        lateral = effective * (SPEED * sin(self.theta) / 1000)
        return np.clip(lateral, -self.clearance - 10, self.clearance + 10)

    def times(self, errors_ms):
        """Vectorised version of the same polyline and gate pause used in play."""
        offset = self.displacement(errors_ms)
        straight = self.launch[0] + FINISH_Y - self.exit[1]
        half_diagonal = self.cut / (2 * cos(self.theta))
        length = straight + 2 * np.hypot(half_diagonal, offset)
        return length / SPEED + CLIP_DELAY * (np.abs(offset) > self.clearance)

    def flight(self, error_ms):
        # A missing input deliberately touches the late barrier.
        offset = float(self.displacement(float('inf') if error_ms is None else error_ms))
        nx, ny = self.normal
        gx, gy = self.gate
        crossing = (gx + nx * offset, gy + ny * offset)
        points = [(0., 0.), self.launch, crossing, self.exit, (TURN_X, FINISH_Y)]
        return Flight(points, abs(offset) > self.clearance)


@dataclass
class Flight:
    points: list[Point]
    clipped: bool

    @property
    def total(self):
        drive = sum(dist(a, b) / SPEED for a, b in zip(self.points, self.points[1:]))
        return drive + (CLIP_DELAY if self.clipped else 0)

    def position(self, elapsed):
        """Return world position, current heading vector, and recovery status."""
        for i, (a, b) in enumerate(zip(self.points, self.points[1:])):
            delta = (b[0] - a[0], b[1] - a[1])
            duration = hypot(*delta) / SPEED
            if elapsed <= duration:
                fraction = max(0.0, elapsed / duration)
                return (a[0] + delta[0] * fraction, a[1] + delta[1] * fraction), delta, False
            elapsed -= duration
            if i == 1 and self.clipped:
                if elapsed < CLIP_DELAY:
                    return b, delta, True
                elapsed -= CLIP_DELAY
        return self.points[-1], (0., 1.), False


WIDE = Route(38., 45., 63.)


def candidates():
    return (Route(cut, angle, width)
            for cut in range(78, 163, 7)
            for angle in range(33, 55, 3)
            for width in range(9, 32, 2))


def design_bank():
    """Precompute mean advantages at the five target points using normal quantiles."""
    normal = NormalDist()
    z = np.array([normal.inv_cdf((i + 0.5) / 512) for i in range(512)])
    errors = SIGMAS[:, None] * z
    wide_times = WIDE.times(errors)
    routes = list(candidates())
    gains = np.array([(wide_times - route.times(errors)).mean(axis=1) for route in routes])
    return routes, gains


def pick_design(bank, targets):
    routes, gains = bank
    loss = np.mean((gains - np.asarray(targets)) ** 2, axis=1)
    return routes[int(np.argmin(loss))]


def validate(route, targets, seed=SIM_SEED):
    """Fresh Monte Carlo stream, paired across routes to reduce comparison noise."""
    z = np.random.default_rng(seed).standard_normal(SIM_RUNS)
    devs = np.arange(0., 201., 5.)
    errors = devs[:, None] * z
    wide_times, cut_times = WIDE.times(errors), route.times(errors)
    gains = (wide_times - cut_times).mean(axis=1)
    residuals = np.interp(SIGMAS, devs, gains) - targets
    rmse = float(np.sqrt(np.mean(residuals**2)))
    return devs, gains, rmse, wide_times.mean(axis=1), cut_times.mean(axis=1)


def first_crossover(devs, gains):
    for i, (a, b) in enumerate(zip(gains, gains[1:])):
        if a > 0 and b <= 0:
            return float(devs[i] + a / (a - b) * (devs[i + 1] - devs[i]))
    return None


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fit and validate an Apex Lab corner.')
    parser.add_argument('--targets', nargs=5, type=float, default=DEFAULT_TARGET,
                        help='Desired gains in seconds at 0, 50, 100, 150 and 200 ms.')
    args = parser.parse_args()
    targets = np.asarray(args.targets)
    if not np.isfinite(targets).all():
        parser.error('Each target must be finite.')
    bank = design_bank()
    best = pick_design(bank, targets)
    devs, gains, rmse, wide, shortcut = validate(best, targets)
    print(f'{len(bank[0])} layouts compared. Selected {best}')
    print('Jitter ms    Target s     Gain s     Wide s    Shortcut s')
    for i, ms in enumerate(SIGMAS):
        j = int(np.searchsorted(devs, ms))
        print(f'{ms:9.0f} {targets[i]:11.3f} {gains[j]:10.3f} {wide[j]:10.3f} {shortcut[j]:13.3f}')
    cross = first_crossover(devs, gains)
    print(f'Validation RMSE {rmse:.3f} seconds')
    print('No positive-to-negative crossover in range.' if cross is None
          else f'First crossover near {cross:.1f} ms')
