from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from quantum_scattering.core import ensure_dir
from quantum_scattering.scattering import appendix_v_scan, appendix_v_state, model_potential
OUT=Path(__file__).resolve().parent/'outputs'/'part3'; DX=0.0025


def prob(E): return float(np.abs(appendix_v_scan(np.array([E]),dx=DX).T[0])**2)
def peak(lo,hi):
    r=minimize_scalar(lambda e:-prob(float(e)),bounds=(lo,hi),method='bounded',options={'xatol':1e-13,'maxiter':300}); return float(r.x),float(-r.fun)


def main_profile():
    e1,p1=peak(.60,.64); e2,p2=peak(1.28,1.38)
    E=np.unique(np.r_[np.linspace(.002,3,2200),np.linspace(e1-.004,e1+.004,1600),np.linspace(e2-.10,e2+.10,1500),e1,e2])
    s=appendix_v_scan(E,dx=DX); TT=np.abs(s.T)**2; RR=np.abs(s.R)**2
    with (OUT/'part3_transmission_data.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['E','T_real','T_imag','R_real','R_imag','absT2','absR2','unitarity_error']); w.writerows(zip(E,s.T.real,s.T.imag,s.R.real,s.R.imag,TT,RR,s.probability_error))
    fig,ax=plt.subplots(figsize=(9,5)); ax.plot(E,TT,color='black'); ax.scatter([e1,e2],[p1,p2],color='red',zorder=3); ax.text(e1+.03,.99,f'E = {e1:.4f}',color='red'); ax.text(e2+.03,.99,f'E = {e2:.4f}',color='red'); ax.set_xlim(0,3); ax.set_ylim(-.05,1.1); ax.set_xlabel('Energy'); ax.set_ylabel('Transmission Probability'); ax.grid(linestyle='--',alpha=.4); fig.tight_layout(); fig.savefig(OUT/'part3_transmission_profile.png',dpi=180); plt.close(fig)
    for name,e,lo,hi in [('first',e1,.6200,.6220),('second',e2,1.20,1.55)]:
        Ez=np.linspace(lo,hi,2400); z=appendix_v_scan(Ez,dx=DX); P=np.abs(z.T)**2
        fig,ax=plt.subplots(figsize=(9,5)); ax.plot(Ez,P,color='black'); ax.scatter([e],[prob(e)],color='red',zorder=3); ax.text(e+(hi-lo)*.08,.99,f'E = {e:.4f}',color='red',fontsize=13); ax.set_xlabel('Energy'); ax.set_ylabel(r'$|T|^2$'); ax.set_ylim(-.05,1.1); ax.grid(linestyle='--',alpha=.4); ax.set_title(f'Transmission zoom near Peak{1 if name=="first" else 2}, E={e:.6f}'); fig.tight_layout(); fig.savefig(OUT/f'part3_{name}_transmission_peak.png',dpi=180); plt.close(fig)
    (OUT/'part3_peak_summary.txt').write_text(f'dx={DX}\nE1={e1:.10f}, |T|^2={p1:.12f}\nE2={e2:.10f}, |T|^2={p2:.12f}\nmax unitarity error={s.probability_error.max():.3e}\n',encoding='utf-8')
    return e1,e2


def wf(energies,filename,title):
    x=np.linspace(-50,50,5000); V=model_potential(x); fig,axs=plt.subplots(3,1,figsize=(12,8),sharex=True)
    for ax,E in zip(axs,energies):
        psi=appendix_v_state(E,x,dx=DX).psi_modified
        ax.plot(x,psi.real,color='#d9534f',label='Re[ψ(x)]'); ax.plot(x,psi.imag,color='#5b9bd5',label='Im[ψ(x)]'); ax.plot(x,V,color='gray',label='V(x)'); ax.set_title(f'E = {E:.8f}'); ax.set_ylabel('ψ(x)'); ax.legend(loc='upper right'); ax.grid(alpha=.2)
    axs[-1].set_xlabel('x'); fig.suptitle(title,fontsize=20,fontweight='bold'); fig.tight_layout(); fig.savefig(OUT/filename,dpi=180); plt.close(fig)


def main():
    ensure_dir(OUT); e1,e2=main_profile()
    wf([0.4,e1,0.75],'part3_continuum_states_first_peak.png','continuum wavefunctions (around the first peak)')
    wf([1.0,e2,1.6],'part3_continuum_states_second_peak.png','continuum wavefunctions (around the second peak)')
if __name__=='__main__': main()
