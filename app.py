"""Apex Lab desktop game. Run with python app.py after installing requirements.txt."""
import csv
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pygame as pg

from experiment import Experiment, Run
from model import (CUE_LEAD, DEFAULT_TARGET, FINISH_Y, LATE_LIMIT_MS, SIGMAS, SIM_RUNS, SPEED,
                   TURN_X, WIDE, design_bank, first_crossover, pick_design, validate)

W, H = 1176, 762
PAPER, WHITE = '#F4F3EE', '#FDFCF8'
INK, SUB, LINE = '#172C29', '#4F625C', '#CBCFC5'
DARK, GRID, ASPHALT = '#153531', '#20413A', '#334A46'
LIME, MINT, BLUE, ORANGE = '#D7F869', '#B0E7CF', '#91C3EC', '#FF9A68'
ACCENT = '#226850'
MAP = pg.Rect(28, 98, 670, 361)
CHART = pg.Rect(790, 222, 321, 160)
CUT_KEY, WIDE_KEY = pg.Rect(49, 590, 137, 37), pg.Rect(193, 590, 106, 37)
PLAY = pg.Rect(371, 584, 302, 47)
BUILD = pg.Rect(740, 455, 382, 41)
SESSION = pg.Rect(740, 678, 382, 39)
VIEW = pg.Rect(1032, 113, 90, 28)
YMIN, YMAX = -2.0, 1.25
FONT = {}


def font(size, kind='plain'):
    key = size, kind
    if key not in FONT:
        names = ('Cascadia Mono,Consolas,DejaVu Sans Mono' if kind == 'mono'
                 else 'Barlow,Segoe UI,DejaVu Sans')
        FONT[key] = pg.font.SysFont(names, size, bold=kind == 'bold')
    return FONT[key]


def label(surf, content, x, y, size=16, colour=INK, kind='plain', align='left'):
    image = font(size, kind).render(str(content), True, colour)
    rect = image.get_rect()
    setattr(rect, {'left':'topleft', 'right':'topright', 'center':'midtop'}[align], (x, y))
    surf.blit(image, rect)


def panel(surf, rect, fill=WHITE):
    pg.draw.rect(surf, fill, rect, border_radius=12)
    pg.draw.rect(surf, LINE, rect, 1, border_radius=12)


def button(surf, rect, words, *, active=False, primary=False, enabled=True):
    fill = (LIME if primary else DARK if active else WHITE) if enabled else '#E7E8E0'
    pg.draw.rect(surf, fill, rect, border_radius=7)
    pg.draw.rect(surf, DARK if enabled else LINE, rect, 1, border_radius=7)
    colour = SUB if not enabled else WHITE if active and not primary else INK
    label(surf, words, rect.centerx, rect.centery - 10, 15, colour, 'bold', 'center')


def project(point):
    return (round(103 + point[0] * 1.10), round(415 - point[1] * 1.10))


def chart_xy(ms, gain):
    x = CHART.left + ms * CHART.width / 200
    y = CHART.bottom - (np.clip(gain, YMIN, YMAX) - YMIN) * CHART.height / (YMAX - YMIN)
    return round(x), round(y)


def signed_gain(value):
    return f'{value:+.2f} s'


def winner(gain):
    if abs(gain) < .001:
        return 'Tie to 0.001 s'
    return ('Shortcut' if gain > 0 else 'Wide') + f' by {abs(gain):.2f} s'


def draw_course(surf, game, now):
    pg.draw.rect(surf, DARK, MAP, border_radius=12)
    for x in range(MAP.left + 22, MAP.right - 6, 23):
        for y in range(MAP.top + 22, MAP.bottom - 6, 23):
            pg.draw.circle(surf, GRID, (x, y), 1)
    # The thicker strokes show the two paved routes. Gate width controls collision.
    for route in (WIDE, game.cut):
        pts = list(map(project, route.flight(0).points))
        pg.draw.lines(surf, '#52635A', False, pts, 27)
        for p in pts[1:-1]:
            pg.draw.circle(surf, '#52635A', p, 13)
    for route in (WIDE, game.cut):
        pts = list(map(project, route.flight(0).points))
        pg.draw.lines(surf, ASPHALT, False, pts, 23)
        for p in pts[1:-1]:
            pg.draw.circle(surf, ASPHALT, p, 11)
    for route in (WIDE, game.cut):
        gx, gy = route.gate
        nx, ny = route.normal
        edges = [project((gx + k*nx*route.width/2, gy + k*ny*route.width/2))
                 for k in (-1, 1)]
        pg.draw.polygon(surf, ASPHALT, [project(route.launch), edges[0],
                                        project(route.exit), edges[1]])
    for route, colour in ((WIDE, BLUE), (game.cut, LIME)):
        a, middle, b = map(project, (route.launch, route.gate, route.exit))
        pg.draw.lines(surf, colour, False, [a, middle, b], 2)
        nx, ny = route.normal
        gx, gy = route.gate
        for sign in (-1, 1):
            def endpoint(d):
                return project((gx + sign * nx * d, gy + sign * ny * d))
            tip = endpoint(route.width / 2)
            pg.draw.line(surf, '#142A27', tip, endpoint(route.width / 2 + 9), 7)
            pg.draw.line(surf, colour, tip, endpoint(route.width / 2 + 9), 3)
            pg.draw.circle(surf, colour, tip, 3)
    # Mark the finish across the vertical straight.
    fx, fy = project((TURN_X, FINISH_Y))
    for row in range(2):
        for col in range(6):
            pg.draw.rect(surf, WHITE if (row + col) % 2 == 0 else INK,
                         (fx - 12 + 4 * col, fy - 4 + 4 * row, 4, 4))
    label(surf, 'FINISH', fx + 21, fy - 8, 12, MINT, 'mono')
    sx, sy = project((0, 0))
    label(surf, 'START', sx, sy + 15, 12, MINT, 'mono')
    route = game.route
    cue = project((route.launch[0] - CUE_LEAD, 0))
    pg.draw.line(surf, WHITE, (cue[0], cue[1]-14), (cue[0], cue[1]+14), 3)
    label(surf, 'TAP', cue[0], cue[1]+16, 12, WHITE, 'mono', 'center')
    label(surf, 'COURSE PLAN', 48, 116, 13, MINT, 'mono')
    for i, (name, colour) in enumerate([('Shortcut', LIME), ('Wide', BLUE)]):
        x = 50 + i * 111
        pg.draw.line(surf, colour, (x, 158), (x+19, 158), 3)
        label(surf, name, x+26, 149, 15, WHITE)
    label(surf, f'{game.cut.angle:.0f}° approach', 49, 194, 16, WHITE, 'bold')
    label(surf, f'{game.cut.width:.0f} unit opening', 49, 219, 14, MINT)
    if game.start_time is None:
        pos, heading, recovering = (0., 0.), (1., 0.), False
        if game.last and game.show_finish:
            pos, heading = (TURN_X, FINISH_Y), (0., 1.)
    else:
        pos, heading, recovering = game.flight.position(max(0., now - game.start_time))
    px, py = project(pos)
    angle = math.atan2(-heading[1], heading[0])
    forward = (math.cos(angle), math.sin(angle))
    side = (-forward[1], forward[0])
    def body_point(f, s):
        return (px + f * forward[0] + s * side[0], py + f * forward[1] + s * side[1])

    for axle in (-4, 3):
        pg.draw.line(surf, INK, body_point(axle, -4), body_point(axle, 4), 3)
    points = [body_point(f, s) for f, s in [(7,0), (3,-3), (-6,-3), (-6,3), (3,3)]]
    pg.draw.polygon(surf, ORANGE if recovering else WHITE, points)
    pg.draw.circle(surf, DARK, body_point(0, 0), 2)
    if recovering:
        label(surf, 'BARRIER +2.00 s', 670, 116, 13, ORANGE, 'mono', 'right')


class Game:
    def __init__(self, data_dir=None):
        self.bank = design_bank()
        self.target = DEFAULT_TARGET.copy()
        self.fitted_target = self.target.copy()
        self.drag = None
        self.chart_mode = 'gain'
        self.route_name = 'shortcut'
        self.start_time = None
        self.last = None
        self.show_finish = False
        self.experiment = None
        self.armed_end = False
        self.message = ''
        self.session_id = uuid4().hex[:12]
        self.design_id = 0
        self.experiment_id = 0
        self.data_dir = data_dir if data_dir is not None else Path.home() / 'ApexLab'
        self.save_error = False
        self.log_path = self.data_dir / f'runs-{self.session_id}.csv'
        self.fit()

    @property
    def route(self):
        return self.cut if self.route_name == 'shortcut' else WIDE

    @property
    def guided(self):
        return self.experiment is not None and self.experiment.stage != 'complete'

    @property
    def editing(self):
        return self.start_time is None and not self.guided

    @property
    def dirty(self):
        return not np.array_equal(self.target, self.fitted_target)

    def fit(self):
        self.cut = pick_design(self.bank, self.target)
        (self.devs, self.gains, self.rmse, self.wide_times,
         self.cut_times) = validate(self.cut, self.target)
        self.cross = first_crossover(self.devs, self.gains)
        self.fitted_target = self.target.copy()
        self.design_id += 1
        self.last = None
        self.show_finish = False
        self.experiment = None
        self.message = ''

    def select(self, name):
        if self.editing:
            self.route_name = name
            self.show_finish = False

    def action(self, now):
        self.armed_end = False
        if self.start_time is None:
            self.show_finish = False
            self.message = ''
            self.error = None
            self.latched = False
            self.start_time = now + random.uniform(.45, .9)
            self.flight = self.route.flight(0.)
            return
        elapsed = now - self.start_time
        if elapsed >= 0 and not self.latched:
            if elapsed <= self.route.cue_time + LATE_LIMIT_MS/1000:
                self.error = (elapsed - self.route.cue_time) * 1000
                self.flight = self.route.flight(self.error)
                self.latched = True

    def cancel_heat(self):
        if self.start_time is not None:
            self.start_time = None
            self.message = 'Run cancelled. Ready to retry.'

    def session_action(self):
        if self.start_time is not None or self.dirty:
            return
        if self.guided:
            if not self.armed_end:
                self.armed_end = True
                return
            self.experiment = None
        else:
            self.experiment = Experiment(self.cut)
            self.experiment_id += 1
            self.route_name = self.experiment.next_route
        self.armed_end = False
        self.show_finish = False
        self.message = ''

    def save(self, record, stage, index):
        prediction = self.experiment.prediction if self.experiment else None
        row = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), design=self.design_id,
                   experiment=self.experiment_id if stage != 'practice' else '', stage=stage,
                   run=index, route=record.route, cut=self.cut.cut, angle=self.cut.angle,
                   gate_width=self.cut.width, error_ms=record.error_ms, clipped=record.clipped,
                   time_s=record.elapsed, predicted_gain_s=prediction if stage == 'test' else None)
        row.update({f'target_{int(s)}ms_s': v for s, v in zip(SIGMAS, self.fitted_target)})
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            new_file = not self.log_path.exists()
            with self.log_path.open('a', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if new_file:
                    writer.writeheader()
                writer.writerow(row)
        except OSError:
            self.save_error = True

    def update(self, now):
        if self.start_time is None:
            return
        elapsed = now - self.start_time
        if not self.latched and elapsed > self.route.cue_time + LATE_LIMIT_MS/1000:
            self.flight = self.route.flight(None)
            self.latched = True
        if elapsed < self.flight.total:
            return
        self.last = Run(self.route_name, self.error, self.flight.total, self.flight.clipped)
        stage, index = 'practice', ''
        if self.guided:
            exp = self.experiment
            stage = exp.stage
            index = len(exp.calibration if stage == 'calibration' else exp.trials) + 1
            exp.record(self.last)
            if exp.next_route:
                self.route_name = exp.next_route
        self.save(self.last, stage, index)
        self.start_time = None
        self.show_finish = True

    def change_target(self, pos):
        i = self.drag
        value = YMIN + (CHART.bottom - pos[1]) * (YMAX - YMIN) / CHART.height
        self.target[i] = round(float(np.clip(value, YMIN, YMAX)), 2)

    def click(self, pos, now):
        if PLAY.collidepoint(pos) or (self.start_time is not None and MAP.collidepoint(pos)):
            self.action(now)
        elif VIEW.collidepoint(pos):
            self.chart_mode = 'times' if self.chart_mode == 'gain' else 'gain'
            self.drag = None
        elif SESSION.collidepoint(pos):
            self.session_action()
        elif CUT_KEY.collidepoint(pos):
            self.select('shortcut')
        elif WIDE_KEY.collidepoint(pos):
            self.select('wide')
        elif self.editing and BUILD.collidepoint(pos) and self.dirty:
            self.fit()
        elif self.editing and self.chart_mode == 'gain' and CHART.inflate(16, 16).collidepoint(pos):
            self.drag = int(np.argmin([abs(chart_xy(s, 0)[0] - pos[0]) for s in SIGMAS]))
            self.change_target(pos)


def draw_chart(surf, game):
    panel(surf, (718, 98, 430, 411))
    label(surf, 'Risk curve', 740, 111, 25, INK, 'bold')
    button(surf, VIEW, 'Times' if game.chart_mode == 'gain' else 'Gain')
    label(surf, 'Drag the five target points, then generate a corner.'
          if game.chart_mode == 'gain' else 'Mean run time at each level of timing variability.',
          741, 150, 14, SUB)
    if game.chart_mode == 'gain':
        label(surf, 'Wide minus shortcut (s)', 741, 182, 13, SUB)
        for x, colour, name in [(962, ACCENT, 'Model'), (1043, '#A45126', 'Target')]:
            pg.draw.line(surf, colour, (x,193), (x+16,193), 3)
            label(surf, name, x+21, 183, 13, SUB)
        limits, ticks = (YMIN, YMAX), [-2, -1, 0, 1]
    else:
        label(surf, 'Run time (s)', 741, 182, 13, SUB)
        for x, colour, name in [(931, ACCENT, 'Shortcut'), (1035, '#356D9A', 'Wide')]:
            pg.draw.line(surf, colour, (x,193), (x+16,193), 3)
            label(surf, name, x+21, 183, 13, SUB)
        low = math.floor(min(game.wide_times.min(), game.cut_times.min()) * 2) / 2
        high = math.ceil(max(game.wide_times.max(), game.cut_times.max()) * 2) / 2
        limits = low, max(low + .5, high)
        ticks = np.arange(low, limits[1] + .1, .5)

    def xy(ms, val):
        lo, hi = limits
        return (int(CHART.left + ms / 200 * CHART.width),
                int(CHART.bottom - (np.clip(val, lo, hi)-lo)/(hi-lo)*CHART.height))

    for v in ticks:
        _, y = xy(0, v)
        pg.draw.line(surf, '#AAB8AA' if v == 0 else '#DADFD4', (CHART.left,y), (CHART.right,y))
        label(surf, f'{v:g}', CHART.left-10, y-9, 13, SUB, 'mono', 'right')
    for v in SIGMAS:
        x, _ = xy(v, 0)
        pg.draw.line(surf, '#E2E6DC', (x, CHART.top), (x, CHART.bottom))
        label(surf, f'{v:.0f}', x, CHART.bottom+4, 12, SUB, 'mono', 'center')

    if game.chart_mode == 'gain':
        target_points = [chart_xy(s, h) for s, h in zip(SIGMAS, game.target)]
        pg.draw.lines(surf, '#BE754A', False, target_points, 2)
        pg.draw.lines(surf, ACCENT, False, [xy(s,h) for s,h in zip(game.devs, game.gains)], 3)
        for x, y in target_points:
            pg.draw.circle(surf, WHITE, (x,y), 6)
            pg.draw.circle(surf, '#A45126', (x,y), 6, 2)
    else:
        for curve, colour in [(game.cut_times, ACCENT), (game.wide_times, '#356D9A')]:
            pg.draw.lines(surf, colour, False, [xy(s,h) for s,h in zip(game.devs, curve)], 3)
    label(surf, 'Timing variability (standard deviation, ms)', CHART.centerx,
          CHART.bottom+25, 13, SUB, align='center')
    crossover = f'{game.cross:.0f} ms' if game.cross is not None else 'none in range'
    label(surf, f'Crossover  {crossover}', 741, 430, 14, INK, 'bold')
    label(surf, f'Fit error  {game.rmse:.3f} s', 1121, 430, 14, SUB, align='right')
    button(surf, BUILD, 'Generate corner' if game.dirty else 'Target fitted',
           primary=True, enabled=game.editing and game.dirty)


def draw_controls(surf, game, now):
    panel(surf, (28, 475, 670, 256))
    label(surf, 'Drive the corner', 50, 489, 25, INK, 'bold')
    elapsed = now - game.start_time if game.start_time is not None else None
    label(surf, 'Press Space or click when the kart reaches the white line.',
          50, 528, 16, SUB)
    label(surf, 'ROUTE', 50, 565, 12, SUB, 'mono')
    button(surf, CUT_KEY, '1   Shortcut', active=game.route_name == 'shortcut',
           enabled=game.editing or game.route_name == 'shortcut')
    button(surf, WIDE_KEY, '2   Wide', active=game.route_name == 'wide',
           enabled=game.editing or game.route_name == 'wide')
    if elapsed is None:
        text = 'Start run   [Space]'
    elif elapsed < 0:
        text = 'Get ready…'
    else:
        text = 'Drift   [Space]' if not game.latched else 'Drift recorded' if game.error is not None else 'Cue missed'
    button(surf, PLAY, text, primary=True, enabled=elapsed is None or (elapsed >= 0 and not game.latched))
    pg.draw.line(surf, LINE, (50,645), (673,645))
    if elapsed is not None:
        label(surf, 'RUN TIME', 50, 659, 12, SUB, 'mono')
        label(surf, f'{max(0.,elapsed):.2f} s', 50, 678, 23, INK, 'bold')
        if game.latched:
            feedback = 'No input' if game.error is None else f'{abs(game.error):.0f} ms ' + ('early' if game.error < 0 else 'late')
        else:
            feedback = 'Watch the white line'
        label(surf, feedback, 673, 678, 19, SUB, align='right')
    elif game.last and not game.message:
        r = game.last
        label(surf, f'LAST RUN  /  {r.route.upper()}', 50, 659, 12, SUB, 'mono')
        label(surf, f'{r.elapsed:.3f} s', 50, 678, 23, INK, 'bold')
        e = 'No input' if r.error_ms is None else f'{abs(r.error_ms):.0f} ms ' + ('early' if r.error_ms < 0 else 'late')
        label(surf, e + ('   /   Barrier +2 s' if r.clipped else '   /   Clean gate'),
              673, 682, 16, SUB, align='right')
    else:
        label(surf, game.message or 'A clean gate rewards the shorter line.',
              50, 664, 17, INK, 'bold')
        label(surf, 'A barrier hit adds two seconds.', 50, 691, 14, SUB)


def draw_session(surf, game):
    panel(surf, (718, 523, 430, 208))
    label(surf, 'Driver experiment', 740, 534, 24, INK, 'bold')
    exp = game.experiment
    if exp is None:
        lines = ['Calibrate on eight runs across both routes.',
                 'Test the prediction on eight fresh runs.']
        tag, action = '16 runs  /  same corner', 'Start experiment'
    elif exp.stage == 'complete':
        tag = 'Complete  /  16 of 16 runs'
        lines = ['Predicted   ' + winner(exp.prediction),
                 'Measured   ' + winner(exp.measured_gain())]
        action = 'Repeat experiment'
    elif exp.stage == 'calibration':
        tag = f'Calibration  /  {len(exp.calibration)} of 8 complete'
        lines = ['Timing errors from each route fit your prediction.',
                 'Next route   ' + exp.next_route.title()]
        action = 'End experiment'
    else:
        tag = f'Fresh test  /  {len(exp.trials)} of 8 complete'
        lines = ['Prediction   ' + winner(exp.prediction),
                 'Next route   ' + exp.next_route.title()]
        action = 'End experiment'
    label(surf, tag, 741, 574, 15, ACCENT, 'bold')
    for i, line in enumerate(lines):
        label(surf, line, 741, 605 + 22*i, 15, SUB)
    button(surf, SESSION, 'Click again to end experiment' if game.armed_end else action,
           enabled=game.start_time is None and not game.dirty)


def draw(surf, game, now):
    surf.fill(PAPER)
    # A hairpin rendered as three racing lines.
    for i in range(3):
        pg.draw.lines(surf, ACCENT, False,
                      [(29, 66-i*8), (48+i*2, 41-i*4), (60+i*4, 41-i*4)], 4)
    label(surf, 'Apex Lab', 83, 22, 41, INK, 'bold')
    label(surf, 'Inverse track designer', 315, 48, 18, SUB)
    label(surf, f'{len(game.bank[0]):,} layouts   /   {SIM_RUNS:,} validation runs per jitter level',
          1148, 50, 13, SUB, 'mono', 'right')
    pg.draw.line(surf, LINE, (28,83), (1148,83))
    draw_course(surf, game, now)
    draw_controls(surf, game, now)
    draw_chart(surf, game)
    draw_session(surf, game)
    label(surf, 'Esc cancels a run', 29, 740, 12, SUB, 'mono')
    log_message = 'Could not write CSV' if game.save_error else 'Completed runs save to ~/ApexLab/'
    label(surf, log_message, 1147, 740, 12, SUB, 'mono', 'right')


def main():
    pg.display.init()
    pg.font.init()
    desktop = pg.display.get_desktop_sizes()[0]
    scale = min(1., desktop[0]*.96/W, desktop[1]*.90/H)
    screen = pg.display.set_mode((int(W*scale), int(H*scale)), pg.RESIZABLE)
    pg.display.set_caption('Apex Lab  /  Inverse track designer')
    canvas = pg.Surface((W,H))
    game = Game()
    clock = pg.time.Clock()
    running = True
    while running:
        now = time.perf_counter()
        width, height = screen.get_size()
        scale = min(width/W, height/H)
        size = (max(1,int(W*scale)), max(1,int(H*scale)))
        margin = ((width - size[0])//2, (height - size[1])//2)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.KEYDOWN and not getattr(event, 'repeat', False):
                if event.key == pg.K_SPACE:
                    game.action(now)
                elif event.key in (pg.K_1, pg.K_2):
                    game.select('shortcut' if event.key == pg.K_1 else 'wide')
                elif event.key == pg.K_ESCAPE:
                    game.cancel_heat()
                    game.armed_end = False
            elif event.type == pg.WINDOWFOCUSLOST:
                game.cancel_heat()
                game.drag = None
            elif event.type == pg.MOUSEBUTTONUP and event.button == 1:
                game.drag = None
            elif event.type in (pg.MOUSEBUTTONDOWN, pg.MOUSEMOTION):
                pos = tuple((val-offset)/scale for val,offset in zip(event.pos, margin))
                if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    game.click(pos, now)
                elif event.type == pg.MOUSEMOTION and game.drag is not None:
                    game.change_target(pos)
        game.update(now)
        draw(canvas, game, now)
        screen.fill(PAPER)
        screen.blit(pg.transform.smoothscale(canvas, size), margin)
        pg.display.flip()
        clock.tick(90)
    pg.quit()


if __name__ == '__main__':
    main()
