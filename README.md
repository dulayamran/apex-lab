# Apex Lab

Draw the time advantage you want a racing shortcut to have at five levels of driver precision. Apex Lab searches 1,248 corner layouts, measures the closest fit with a fresh simulation, and lets you drive both routes.

## Run on Windows

Use Python 3.10 or newer. Extract the ZIP, open the folder containing `app.py`, and run these commands in its terminal.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

In PyCharm, you can also select the project's interpreter, install the packages in `requirements.txt` into that interpreter, and run `app.py`. On macOS or Linux, use `python3 -m venv .venv`, then `.venv/bin/python -m pip install -r requirements.txt` and `.venv/bin/python app.py`.

## Try a run

1. Press **Space** to start. Watch the kart approach the white **TAP** line. Press **Space** again as the centre of the kart crosses the line. Clicking the drift button or the course also works during a run.
2. Use **1** for the shortcut and **2** for the wide route. Both use the same movement and collision rules. Each route has its own cue on the approach straight. **Esc** cancels the current run. Leaving the window also cancels a run. Cancelled runs do not enter the experiment.
3. Drag the five points on the gain chart. Positive values mean the shortcut should be faster. Click **Generate corner** to search for the closest layout. The **Times** button switches to the mean duration of each route. Fit error is root mean squared error in seconds, measured at your five target points. A curve that is difficult to achieve will show a larger error.
4. Select **Start experiment** for eight calibration runs followed by eight test runs. Each block includes four runs per route with counterbalanced ordering. The game selects each route for you. Timing errors from the calibration block predict the mean time gap. That prediction is frozen before the first test run. At the end, compare it with the measured gap. Four observations per route make this a small pilot. New experiments record separate IDs so you can repeat it.

Finished runs are appended to `ApexLab/runs-<session>.csv` inside your home folder. Each row includes the geometry, target, experiment stage, route, signed timing error, collision and total time. A blank timing error records a missed cue. Test rows contain the frozen prediction in seconds, with a positive value favouring the shortcut.

## How the model works

The course is a single top-down corner. Each route joins two perpendicular straights with a diagonal. A route has three parameters: the distance cut off the first straight, the approach angle and the gate opening. The wide route stays fixed. The shortcut varies across 13 distances, 8 angles and 12 openings.

A kart travels at 90 game units per second. Your tap sets its line through the gate. A signed timing error of `e` seconds produces a lateral offset of `90 × e × sin(angle)` at the gate. The kart travels from the start of the diagonal through that offset point, then rejoins the exit. A centre offset greater than half the opening minus the kart radius clips the barrier and adds a two-second stop. For collisions, the model limits travel beyond the barrier to 10 units beyond the allowed centre offset. Missing the cue by more than 500 ms on the late side uses the late recovery line. An on-time tap is the shortest possible path between the diagonal's endpoints. Both the animated kart and simulated driver use this geometry.

For design, `model.py` evaluates 512 equally weighted normal quantiles at each timing standard deviation (0, 50, 100, 150 and 200 ms). It selects the layout with the lowest mean squared error across those five targets. Validation uses 6,000 draws from a separate seeded random-number generator at each of 41 precision levels. Each draw is shared by the two routes so the comparison pairs equal timing errors. The graph reports the first positive-to-negative crossing by interpolation. All simulated errors are centred on zero. The personal experiment instead uses your observed, signed errors for each route, including missed cues. Frame rate, screen refresh and keyboard latency all affect those measurements.

## Reproduce the default result

From the same Python environment, you can run the model and tests without opening the game.

```text
python model.py
python model.py --targets 0.7 0.6 0.2 -0.2 -0.5
python -m unittest discover -s tests -v
```

With seed 417, the default target selects a 48-degree shortcut, a 21-unit opening and a 106-unit cut. With perfect timing the shortcut saves about **0.478 s**. Its estimated advantage crosses zero at **94 ms** of timing standard deviation. These figures describe the simulator. The driver experiment measures your own performance.

## Source map

| File | Contents |
| --- | --- |
| `app.py` | The Pygame interface, race controls, chart editing and CSV recording |
| `model.py` | Track geometry, shared timing rules, search and simulation |
| `experiment.py` | Assignment of routes, prediction from calibration runs and the independent test block |
| `tests/test_apex.py` | Regression checks for geometry, track selection, race controls, prediction and logging |
