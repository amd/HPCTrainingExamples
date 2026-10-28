"""Render inverse-fit artifacts from saved numerical results, without a GPU."""
import argparse
import csv
import json
from pathlib import Path


def plot_results(run_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import numpy as np

    run_dir = Path(run_dir)
    run = json.loads((run_dir/'run.json').read_text())
    with (run_dir/'grid.csv').open() as stream:
        grid = list(csv.DictReader(stream))
    with (run_dir/'optimization_history.csv').open() as stream:
        history = list(csv.DictReader(stream))
    with np.load(run_dir/'snapshot.npz') as data:
        coords, observed, fitted = data['coords_mm'], data['observed_mm'], data['fitted_mm']
    # Display all observations on undeformed x-y coordinates with equal physical
    # aspect. Fitting uses all 3 displacement components, not this 2D projection.
    obs_mag, fit_mag = np.linalg.norm(observed,axis=1),np.linalg.norm(fitted,axis=1)
    residual = np.linalg.norm(fitted-observed,axis=1)
    with plt.rc_context({'font.size':10, 'figure.facecolor':'white'}):
        fig, axes = plt.subplots(1,3,figsize=(14,6),layout='constrained')
        common_max = max(float(obs_mag.max()),float(fit_mag.max()))
        for ax,values,title,vmax in zip(axes,[obs_mag,fit_mag,residual],
                    ['Observed FEA displacement','Fitted surrogate displacement','Vector residual magnitude'],
                    [common_max,common_max,float(residual.max())]):
            points=ax.scatter(coords[:,0],coords[:,1],c=values,s=2,cmap='viridis',vmin=0,vmax=max(vmax,1e-9),rasterized=True)
            ax.set(title=title,xlabel='Initial x (mm)',ylabel='Initial y (mm)')
            ax.set_aspect('equal')
            fig.colorbar(points,ax=ax,label='mm',shrink=.7)
        fig.suptitle(f"One snapshot: frame {run['observation']['frame_after_initial']} | fit vector RMSE {run['fitted_vector_rmse_mm']:.3f} mm")
        fig.savefig(run_dir/'displacement_fit.png',dpi=160)
        plt.close(fig)

        cb=np.unique([float(row['crash_box']) for row in grid])
        beam=np.unique([float(row['beam']) for row in grid])
        z=np.empty((len(beam),len(cb)))
        for row in grid:
            z[np.searchsorted(beam,float(row['beam'])),np.searchsorted(cb,float(row['crash_box']))]=float(row['vector_rmse_mm'])
        fig,ax=plt.subplots(figsize=(8,7),layout='constrained')
        positive=np.maximum(z,1e-6)
        colors=ax.pcolormesh(cb,beam,positive,shading='nearest',cmap='viridis',
                            norm=LogNorm(vmin=positive.min(),vmax=max(positive.max(),positive.min()*1.001)))
        fig.colorbar(colors,ax=ax,label='Vector RMSE (mm; log color scale)')
        for start in sorted(set(int(row['start']) for row in history)):
            rows=[row for row in history if int(row['start'])==start]
            ax.plot([float(row['crash_box']) for row in rows],[float(row['beam']) for row in rows],
                    '-o',markersize=3,linewidth=1.3,label=f'Start {start}')
        truth=run['observation']['true_design']
        estimate=run['inferred_design']
        ax.scatter([truth['crash_box']],[truth['beam']],marker='D',s=110,c='white',edgecolors='black',label='True design (revealed after fit)',zorder=5)
        ax.scatter([estimate['crash_box']],[estimate['beam']],marker='*',s=240,c='#e15759',edgecolors='black',label='Selected fit',zorder=6)
        ax.set(xlabel='Crash-box thickness / nominal',ylabel='Beam thickness / nominal',
               title='Which configurations explain the observation?',xlim=(.66,1.34),ylim=(.66,1.34))
        ax.set_aspect('equal')
        fig.legend(*ax.get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
        fig.savefig(run_dir/'loss_landscape.png',dpi=160)
        plt.close(fig)

        fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
        for start in sorted(set(int(row['start']) for row in history)):
            rows=[row for row in history if int(row['start'])==start]
            axes[0].plot([int(row['iteration']) for row in rows],[float(row['vector_rmse_mm']) for row in rows],'-o',markersize=3,label=f'Start {start}')
        axes[0].set(xlabel='Accepted optimizer iteration (0 = start)',ylabel='Vector RMSE (mm)',title='Autograd-guided fitting')
        axes[0].legend(frameon=False)
        axes[0].grid(alpha=.2)
        x=np.arange(2)
        axes[1].bar(x-.18,[truth['crash_box'],truth['beam']],width=.36,color='#4e79a7',label='True design')
        axes[1].bar(x+.18,[estimate['crash_box'],estimate['beam']],width=.36,color='#e15759',label='Inferred design')
        axes[1].set(xticks=x,xticklabels=['Crash box','Beam'],ylabel='Thickness / nominal',ylim=(0,1.45),title='Parameter recovery')
        axes[1].legend(frameon=False)
        fig.savefig(run_dir/'convergence.png',dpi=160)
        plt.close(fig)
    return ['displacement_fit.png','loss_landscape.png','convergence.png']


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir',type=Path)
    args=parser.parse_args()
    for name in plot_results(args.run_dir):
        print(args.run_dir/name)
