from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from quantum_scattering.core import gaussian_x, free_propagate_fft, ensure_dir

OUT=Path(__file__).resolve().parent/'outputs'/'part1'


def components(ax,x,psi):
    ax.plot(x,psi.real,color='red',label='Real Part of psi0')
    ax.plot(x,psi.imag,color='blue',linestyle='--',label='Imaginary Part of psi0')
    ax.plot(x,np.abs(psi)**2,color='black',label='Square modulus of psi0')
    ax.set_xlabel('Position'); ax.set_ylabel('Amplitude')


def initial_figures():
    x=np.linspace(-10,10,4000)
    fig,axs=plt.subplots(1,3,figsize=(15,5))
    for ax,a in zip(axs,[0.1,1.0,5.0]):
        components(ax,x,gaussian_x(x,a,1.0)); ax.set_title(f'a={a:.1f} p=1.0'); ax.set_ylim(-1,3); ax.set_xlim(-10,10); ax.legend()
    fig.suptitle('initial gaussians - width is changing',fontsize=22,fontweight='bold'); fig.tight_layout(); fig.savefig(OUT/'part1_initial_widths.png',dpi=180); plt.close(fig)

    fig,axs=plt.subplots(1,3,figsize=(15,5))
    for ax,p in zip(axs,[0.1,1.0,5.0]):
        components(ax,x,gaussian_x(x,1.0,p)); ax.set_title(f'a=1.0 p={p:.1f}'); ax.set_ylim(-1,3); ax.set_xlim(-10,10); ax.legend()
    fig.suptitle('initial gaussians - momentum is changing',fontsize=22,fontweight='bold'); fig.tight_layout(); fig.savefig(OUT/'part1_initial_momenta.png',dpi=180); plt.close(fig)


def snapshots():
    x=np.linspace(-30,100,2**15,endpoint=False); dx=x[1]-x[0]
    psi0=gaussian_x(x,1.0,5.0)
    fig,axs=plt.subplots(3,1,figsize=(12,6),sharex=True)
    for ax,t in zip(axs,[0.0,4.0,8.0]):
        psi=free_propagate_fft(psi0,dx,t)
        ax.plot(x,psi.real,color='red'); ax.plot(x,psi.imag,color='blue',linestyle='--'); ax.plot(x,np.abs(psi)**2,color='black')
        ax.set_title(f'a=1.0,p=5.0,t={t:.1f}'); ax.set_ylabel('Amplitude'); ax.set_xlim(-10,80); ax.set_ylim(-1,1)
    axs[-1].set_xlabel('Position'); fig.suptitle('free wavepacket propagation - snapshots',fontsize=22,fontweight='bold'); fig.tight_layout(); fig.savefig(OUT/'part1_free_snapshots.png',dpi=180); plt.close(fig)


def animation(quick=False):
    x=np.linspace(-50,300,2**14,endpoint=False); dx=x[1]-x[0]; psi0=gaussian_x(x,1.0,4.0)
    times=np.arange(0,45.0001,0.6 if quick else 0.2)
    fig,ax=plt.subplots(figsize=(10,5));
    lr,=ax.plot([],[],color='red',label='Real Part of psit'); li,=ax.plot([],[],color='blue',linestyle='--',label='Imaginary Part of psit'); lp,=ax.plot([],[],color='black',label='Square modulus of psit')
    ax.set_xlim(-50,300); ax.set_ylim(-1,1); ax.set_xlabel('Position'); ax.set_ylabel('Amplitude'); ax.legend(loc='upper right'); title=ax.set_title('')
    def update(t):
        psi=free_propagate_fft(psi0,dx,float(t)); lr.set_data(x,psi.real); li.set_data(x,psi.imag); lp.set_data(x,np.abs(psi)**2); title.set_text(f'Free particle evolution,a=1,p=4,t={t:.1f}'); return lr,li,lp,title
    FuncAnimation(fig,update,frames=times,blit=False).save(OUT/'part1_free_wavepacket.gif',writer=PillowWriter(fps=18)); plt.close(fig)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--quick',action='store_true'); a=ap.parse_args(); ensure_dir(OUT); initial_figures(); snapshots(); animation(a.quick)
if __name__=='__main__': main()
