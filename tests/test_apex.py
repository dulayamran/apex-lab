import csv
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

import numpy as np
import pygame as pg

from app import BUILD, CHART, H, MAP, PLAY, SESSION, VIEW, W, Game, draw
from model import (CLIP_DELAY, DEFAULT_TARGET, SPEED, WIDE, Route, design_bank,
                   first_crossover, pick_design, validate)


class TrackTests(unittest.TestCase):
    def test_shared_physics(self):
        for route in [WIDE, Route(78, 33, 11), Route(162, 54, 29)]:
            for error in [-1200, -130, -50, 0, 50, 130, 1200, None]:
                flight = route.flight(error)
                self.assertAlmostEqual(float(route.times(np.inf if error is None else error)),
                                       flight.total)
                self.assertEqual(flight.position(flight.total + .01)[0], flight.points[-1])
                if flight.clipped:
                    t = sum(np.linalg.norm(np.subtract(a,b)) for a,b in
                            zip(flight.points[:2], flight.points[1:3])) / SPEED
                    pos, _, stopped = flight.position(t + CLIP_DELAY/2)
                    self.assertTrue(stopped)
                    self.assertEqual(pos, flight.points[2])
        self.assertFalse(WIDE.flight(0).clipped)
        self.assertTrue(WIDE.flight(None).clipped)

    def test_width_changes_clip_outcome(self):
        self.assertTrue(Route(106, 48, 11).flight(90).clipped)
        self.assertFalse(Route(106, 48, 31).flight(90).clipped)

    def test_inverse_design_and_crossover(self):
        bank = design_bank()
        best = pick_design(bank, DEFAULT_TARGET)
        xs, gain, rmse, wide, cut = validate(best, DEFAULT_TARGET)
        self.assertLess(rmse, .10)
        self.assertGreater(gain[0], 0)
        self.assertLess(gain[-1], 0)
        self.assertTrue(75 < first_crossover(xs, gain) < 120)
        np.testing.assert_allclose(gain, wide - cut)
        self.assertNotEqual(best, pick_design(bank, [1, .8, .6, .4, .2]))
        self.assertIsNone(first_crossover([0, 50, 100], [1, .9, .8]))


class PlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pg.display.init()
        pg.font.init()

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.game = Game(Path(self.folder.name))
        self.surface = pg.Surface((W, H))

    def run_heat(self, error):
        g = self.game
        g.click(PLAY.center, 100.)
        start = g.start_time
        if error is not None:
            g.click(MAP.center, start + g.route.cue_time + error/1000)
        g.update(start + g.route.cue_time + .51)
        draw(self.surface, g, start + g.route.cue_time + .51)
        g.update(start + g.flight.total + .01)
        self.assertIsNone(g.start_time)

    def test_complete_experiment_and_log(self):
        g = self.game
        g.click(SESSION.center, 0.)
        # Calibrate with accurate taps. Withhold a deliberately poor second block.
        for _ in range(8):
            self.run_heat(5)
        frozen = g.experiment.prediction
        self.assertGreater(frozen, 0)
        for _ in range(8):
            self.run_heat(None if g.route_name == 'shortcut' else 0)
        self.assertEqual(g.experiment.prediction, frozen)
        self.assertLess(g.experiment.measured_gain(), 0)
        self.assertFalse(g.guided)
        draw(self.surface, g, 180.)
        with g.log_path.open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 16)
        self.assertEqual({row['run'] for row in rows[:8]}, {str(n) for n in range(1,9)})
        self.assertTrue(all(row['stage'] == 'test' for row in rows[8:]))
        self.assertTrue(all(float(row['predicted_gain_s']) == frozen for row in rows[8:]))
        self.assertEqual(sum(row['error_ms'] == '' for row in rows), 4)

    def test_change_target_and_retry(self):
        g = self.game
        old = g.cut
        g.click(CHART.topleft, 0)
        g.drag = None
        self.assertTrue(g.dirty)
        g.click(BUILD.center, 0)
        self.assertFalse(g.dirty)
        self.assertNotEqual(g.cut, old)
        g.click(VIEW.center, 0)
        self.assertEqual(g.chart_mode, 'times')
        draw(self.surface, g, 0)
        g.click(SESSION.center, 0)
        g.action(10)
        g.cancel_heat()
        self.assertEqual(len(g.experiment.calibration), 0)
        g.session_action()
        self.assertTrue(g.guided)
        g.session_action()
        self.assertFalse(g.guided)


if __name__ == '__main__':
    unittest.main()
