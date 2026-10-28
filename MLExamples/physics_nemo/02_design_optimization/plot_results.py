"""Plot saved optimization outputs without loading the model or using a GPU."""
import argparse
import csv
import json
from pathlib import Path


def plot_results(run_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    run_dir = Path(run_dir)
    result = json.loads((run_dir / 'result.json').read_text(encoding='utf-8'))
    with (run_dir / 'grid.csv').open() as stream:
        rows = [{key: float(value) for key, value in row.items()}
                for row in csv.DictReader(stream)]
    cb = np.array([row['crash_box'] for row in rows])
    beam = np.array([row['beam'] for row in rows])
    peak = np.array([row['adjusted_peak_mm'] for row in rows])
    mass = np.array([row['mass_proxy'] for row in rows])
    limit = result['arguments']['limit_mm']
    factor = result['arguments']['safety_factor']
    candidate = result['candidate']
    feasible = peak <= limit
    cb_axis, beam_axis = np.unique(cb), np.unique(beam)
    z = np.full((len(beam_axis), len(cb_axis)), np.nan)
    for row in rows:
        i = np.searchsorted(beam_axis, row['beam'])
        j = np.searchsorted(cb_axis, row['crash_box'])
        z[i, j] = row['adjusted_peak_mm']
    if np.isnan(z).any():
        raise ValueError('Expected a complete rectangular design grid')

    with plt.rc_context({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': 'white'}):
        fig, ax = plt.subplots(figsize=(9, 7), layout='constrained')
        surface = ax.pcolormesh(cb_axis, beam_axis, z, shading='nearest', cmap='viridis_r')
        fig.colorbar(surface, ax=ax, label='Margin-adjusted predicted peak (mm)')
        # A contour interpolates the coarse grid; it is not another model evaluation.
        if z.min() < limit < z.max():
            contour = ax.contour(cb_axis, beam_axis, z, levels=[limit], colors='white', linewidths=2)
            ax.clabel(contour, fmt={limit: f'{limit:g} mm limit'}, fontsize=11)
        ax.scatter(cb[feasible], beam[feasible], marker='o', s=45, facecolors='none',
                   edgecolors='white', linewidths=1.4, label='Grid: predicted feasible')
        ax.scatter(cb[~feasible], beam[~feasible], marker='x', s=45, color='#f28e2b',
                   label='Grid: exceeds limit')
        ax.scatter([1], [1], marker='D', s=90, color='#56b4e9', edgecolors='black', label='Nominal design')
        ax.scatter([candidate['crash_box']], [candidate['beam']], marker='*', s=230,
                   color='#e15759', edgecolors='black', label='SLSQP candidate', zorder=5)
        ax.set(xlabel='Crash-box thickness / nominal', ylabel='Beam thickness / nominal',
               title='Where does the surrogate satisfy the displacement limit?',
               xlim=(.66, 1.34), ylim=(.66, 1.34))
        ax.set_aspect('equal')
        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc='outside lower center', ncol=2, frameon=False)
        legend.legend_handles[0].set_edgecolor('#555555')
        fig.savefig(run_dir / 'design_space.png', dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 6), layout='constrained')
        ax.axhspan(0, limit, color='#59a14f', alpha=.08)
        ax.axhline(limit, color='#555555', linestyle='--', label=f'Limit: {limit:g} mm')
        ax.scatter(mass[feasible], peak[feasible], s=55, color='#207f69', label='Grid: predicted feasible')
        ax.scatter(mass[~feasible], peak[~feasible], s=55, marker='x', color='#d47518',
                   label='Grid: exceeds limit')
        nominal_peak = factor * result['nominal_peak_mm']
        ax.scatter([1], [nominal_peak], s=100, marker='D', color='#56b4e9', edgecolors='black', label='Nominal design')
        ax.scatter([candidate['mass_proxy']], [candidate['adjusted_peak_mm']], marker='*', s=250,
                   color='#e15759', edgecolors='black', label='SLSQP candidate', zorder=5)
        ax.annotate(f"Candidate mass proxy: {candidate['mass_proxy']:.4f}\n"
                    f"Predicted peak: {candidate['adjusted_peak_mm']:.2f} mm",
                    (candidate['mass_proxy'], candidate['adjusted_peak_mm']),
                    xytext=(.48, .92), textcoords='axes fraction', va='top',
                    arrowprops={'arrowstyle': '->', 'color': '#444444'},
                    bbox={'facecolor': 'white', 'edgecolor': '#cccccc', 'alpha': .95})
        values = np.r_[peak, nominal_peak, candidate['adjusted_peak_mm'], limit]
        padding = max(float(np.ptp(values))*.2, 5.)
        ax.set(xlabel='Mass proxy (nominal = 1)', ylabel='Margin-adjusted predicted peak (mm)',
               title='Mass versus predicted displacement',
               ylim=(max(0, values.min()-padding), values.max()+padding))
        ax.grid(alpha=.2)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncol=2, frameon=False)
        fig.savefig(run_dir / 'mass_displacement.png', dpi=160)
        plt.close(fig)
    return ['design_space.png', 'mass_displacement.png']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path, help='Directory containing grid.csv and result.json')
    args = parser.parse_args()
    for name in plot_results(args.run_dir):
        print(args.run_dir / name)
