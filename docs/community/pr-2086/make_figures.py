# The two pictures of the community post on sunnypilot PR #2086 (Speed Limit Resolver, Map Data limit ahead).
# 1. sim.png: a simple simulation of the resolver's map logic on master and with the PR, the car braking at most as the
#    planner does (A_CRUISE_MIN).
# 2. replay.png: the counts measured by replaying 5 logged drives (31 drops of the map's limit, Map Data only).
# Run: uv run --no-project --with matplotlib python docs/community/pr-2086/make_figures.py
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).parent
SURFACE, INK, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE = '#2a78d6', '#eb6834'
LIMIT_ADAPT_ACC = -1.0  # m/s², as in sunnypilot's resolver
A_CRUISE_MIN = -0.5  # m/s², the planner's strongest braking for a lower cruise speed
KPH = 1 / 3.6

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE})


def resolver(version: str, v_ego: float, current: float, ahead: float, distance: float) -> float:
    """The map limit the resolver gives (m/s) at distance (m) before the change from current to ahead."""
    if distance <= 0:
        return ahead
    if version == 'master':
        # time.monotonic() minus a Unix time: the distance driven is about -v_ego * 1.79e9 m, so the distance to
        # the limit ahead never shrinks below the braking distance.
        distance += v_ego * 1.79e9
        v_start = v_ego
    else:
        v_start = max(v_ego, current)
    if 0 < ahead < v_start:
        adapt_time = (ahead - v_start) / LIMIT_ADAPT_ACC
        adapt_distance = v_start * adapt_time + 0.5 * LIMIT_ADAPT_ACC * adapt_time ** 2
        if distance <= adapt_distance:
            return ahead
    return current


def drive(version: str, brake: float = -A_CRUISE_MIN):
    """A car at 80 km/h approaching an 80 -> 50 change of the map's limit; it follows the resolver's limit, braking at
    brake m/s². The resolver is read once a second (mapd sends the limit ahead at 1 Hz)."""
    current, ahead = 80 * KPH, 50 * KPH
    x, v, dt, t = -340.0, current, 0.02, 0.0
    xs, speeds, limits = [], [], []
    limit = current
    while x < 340:
        if t % 1.0 < dt or x >= 0 > x - v * dt:
            limit = resolver(version, v, current, ahead, -x)
        if v > limit:
            v = max(limit, v - brake * dt)
        xs.append(x)
        speeds.append(v / KPH)
        limits.append(limit / KPH)
        x += v * dt
        t += dt
    return xs, speeds, limits


def sim():
    rows = (('master', 'sunnypilot master: the limit ahead is never used'),
            ('pr', 'PR #2086: switches to 50 before the change and holds'))
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True, sharey=True)
    for ax, (version, title) in zip(axes, rows):
        xs, speeds, limits = drive(version)
        ax.plot([-340, 0, 0, 340], [80, 80, 50, 50], color=MUTED, lw=1.2, ls=(0, (4, 3)), label="map's limit at the car")
        ax.plot(xs, limits, color=ORANGE, lw=2, label="resolver's limit", drawstyle='steps-post')
        if version == 'pr':
            ideal_x, ideal_v, _ = drive(version, brake=-LIMIT_ADAPT_ACC)
            ax.plot(ideal_x, ideal_v, color=BLUE, lw=1.2, ls=(0, (1, 2)), label='at 1 m/s², what the resolver plans for')
        ax.plot(xs, speeds, color=BLUE, lw=2, label='car speed, braking at most 0.5 m/s² (planner)')
        ax.fill_between(xs, 50, speeds, where=[x > 0 and s > 50.5 for x, s in zip(xs, speeds)], color=BLUE, alpha=0.12,
                        lw=0)
        at_change = next(s for x, s in zip(xs, speeds) if x >= 0)
        reached = next(x for x, s in zip(xs, speeds) if s <= 50.05)
        ax.text(6, at_change + 1.5, f'{at_change:.0f} km/h at the change', color=INK, fontsize=10)
        ax.axvline(reached, color=GRID, lw=0.8)
        ax.text(reached + 4, 43, f'50 after {round(reached, -1):.0f} m', color=INK, fontsize=10)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_title(title, loc='left', fontsize=11.5, color=INK, pad=6)
        ax.set_ylim(40, 88)
        ax.set_yticks([50, 60, 70, 80])
        ax.grid(axis='y', color=GRID, lw=0.8)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.set_ylabel('km/h')
    axes[-1].set_xlabel("metres to where the map's limit changes from 80 to 50")
    handles, labels = [], []
    for ax in axes[::-1]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc='upper left', ncol=2, frameon=False, bbox_to_anchor=(0.07, 0.995), fontsize=10,
               labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / 'sim.png', dpi=160)


def replay():
    versions = ['sunnypilot master', 'Clock fix only', 'Both fixes (PR #2086)']
    early, back = [0, 9, 27], [0, 6, 1]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    y = range(len(versions))
    h = 0.34
    ax.barh([i - h / 2 - 0.02 for i in y], early, height=h, color=BLUE, label='switched early to the lower limit')
    ax.barh([i + h / 2 + 0.02 for i in y], back, height=h, color=ORANGE, label='went back up before the change')
    for i in y:
        ax.text(early[i] + 0.4, i - h / 2 - 0.02, str(early[i]), va='center', color=INK, fontsize=10.5)
        ax.text(back[i] + 0.4, i + h / 2 + 0.02, str(back[i]), va='center', color=INK, fontsize=10.5)
    ax.set_yticks(list(y), versions, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 31)
    ax.set_xticks([0, 10, 20, 31])
    ax.set_xlabel("drops of the map's limit, of 31 (5 logged drives, policy Map Data only)")
    ax.grid(axis='x', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.legend(loc='upper right', frameon=False, fontsize=10, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / 'replay.png', dpi=160)


if __name__ == '__main__':
    sim()
    replay()
