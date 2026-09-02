from dataclasses import dataclass
from pathlib import Path
import argparse,csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from quantum_scattering.core import gaussian_p,ensure_dir
from quantum_scattering.scattering import appendix_v_scan,left_incident_state_matrix,model_potential
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'outputs'/'part4'; MASS=1.; HBAR=1.
PART3_PEAK_SUMMARY=ROOT/'outputs'/'part3'/'part3_peak_summary.txt'

@dataclass(frozen=True)
class Scenario:
    name:str; Emin:float; E0:float; Emax:float; delta_p:float; collision_time:float; snapshots:tuple; xlim:tuple; ylim:tuple
    @property
    def p0(self): return np.sqrt(2*self.E0)
    @property
    def pmin(self): return np.sqrt(2*self.Emin)
    @property
    def pmax(self): return np.sqrt(2*self.Emax)
    @property
    def alpha(self):
        # Formal lecture definition Delta p = hbar sqrt(alpha).
        return (self.delta_p/HBAR)**2
    @property
    def x0(self): return -(self.p0/MASS)*self.collision_time

def saved_part3_second_peak():
    try:
        lines=PART3_PEAK_SUMMARY.read_text(encoding='utf-8').splitlines()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f'Part-3 peak summary not found: {PART3_PEAK_SUMMARY}. Run Part 3 first.') from exc
    for line in lines:
        if line.startswith('E2='):
            return float(line.split('=',1)[1].split(',',1)[0])
    raise ValueError(f'E2 was not found in Part-3 peak summary: {PART3_PEAK_SUMMARY}')

SCENARIOS={
'blocked':Scenario('blocked',0.8,1.0,1.2,0.049497,25.0,(-20.,10.,25.,50.),(-200.,200.),(-1.,2.)),
'partially_blocked':Scenario('partially_blocked',1.248814,1.328814,1.408814,0.015556,30.0,(-20.,10.,35.,80.),(-200.,200.),(-1.,2.)),
'pass':Scenario('pass',2.5,2.7,2.9,0.026870,5.0,(-20.,5.,20.,50.),(-200.,200.),(-1.,2.)),
}

def second_resonance_scenario():
    E0=saved_part3_second_peak(); half_width=.001
    return Scenario('second_resonance',E0-half_width,E0,E0+half_width,0.000177,5500.0,(-20.,2000.,4500.,10000.),(-20000.,20000.),(-.10,.10))

def selected_scenarios(name):
    if name=='all': return [*SCENARIOS.values(),second_resonance_scenario()]
    return [second_resonance_scenario()] if name=='second_resonance' else [SCENARIOS[name]]

def weights(p):
    w=np.empty_like(p); w[1:-1]=.5*(p[2:]-p[:-2]); w[0]=.5*(p[1]-p[0]); w[-1]=.5*(p[-1]-p[-2]); return w

def setup(sc,x,n_p):
    p=np.linspace(max(1e-8,sc.p0-6.5*sc.delta_p),sc.p0+6.5*sc.delta_p,n_p); states,T,R=left_incident_state_matrix(p,x,dx=.01)
    coeff=gaussian_p(p,sc.alpha,sc.p0,x0=sc.x0); E=p**2/2; return p,E,states,T,R,weights(p)*coeff

def evolve(E,states,base,times): return (np.exp(-1j*np.asarray(times)[:,None]*E[None,:])*base[None,:])@states

def tprob(E): return float(np.abs(appendix_v_scan(np.array([E]),dx=.01).T[0])**2)

def plotwave(ax,x,psi,t,ylim):
    ax.plot(x,psi.real,color='red',label=r'Re $\Psi$'); ax.plot(x,psi.imag,color='blue',linestyle='--',label=r'Im $\Psi$'); ax.plot(x,np.abs(psi)**2,color='black',label=r'$|\Psi|^2$'); ax.plot(x,model_potential(x),color='0.65',label=r'$V(x)$'); ax.axhline(0,color='0.4',lw=.5); ax.set_ylim(*ylim); ax.set_title(f'$t={t:.1f}$'); ax.set_ylabel('amplitude'); ax.grid(alpha=.2)

def summary(sc,n_p):
    x=np.linspace(sc.xlim[0],sc.xlim[1],1800 if sc.name=='second_resonance' else 1500); p,E,states,T,R,base=setup(sc,x,n_p); psis=evolve(E,states,base,sc.snapshots)
    Eplot=np.linspace(1.30,1.36,1000) if sc.name=='second_resonance' else np.linspace(max(1e-5,sc.Emin-.2*(sc.Emax-sc.Emin)),sc.Emax+.2*(sc.Emax-sc.Emin),900)
    P=np.abs(appendix_v_scan(Eplot,dx=.01).T)**2
    fig=plt.figure(figsize=(14,9),constrained_layout=True); gs=fig.add_gridspec(4,2,width_ratios=[1,1.2]); aT=fig.add_subplot(gs[:2,0]); aP=fig.add_subplot(gs[2:,0]); axs=[fig.add_subplot(gs[i,1]) for i in range(4)]
    aT.plot(Eplot,P,color='black');
    for val,col,lab in [(sc.Emin,'blue',rf'$E_{{min}}={sc.Emin:.6f}$'),(sc.E0,'red',rf'$E_0={sc.E0:.6f}$'),(sc.Emax,'green',rf'$E_{{max}}={sc.Emax:.6f}$')]: aT.axvline(val,color=col,linestyle='--',label=lab)
    aT.set(xlabel='Energy',ylabel=r'$|T(E)|^2$',title='Transmission Spectrum $T(E)$'); aT.set_ylim(-.02,1.05); aT.grid(alpha=.2); aT.legend(fontsize=8)
    phi=gaussian_p(p,sc.alpha,sc.p0); pp=np.abs(phi)**2; aP.plot(p,pp,color='black',label=r'$|\phi(p)|^2$')
    for val,Ee,col,lab in [(sc.pmin,sc.Emin,'blue',r'$p_{min}$'),(sc.p0,sc.E0,'red',r'$p_0$'),(sc.pmax,sc.Emax,'green',r'$p_{max}$')]:
        yy=float(np.interp(val,p,pp)); aP.scatter([val],[yy],color=col,s=35,label=rf'{lab}, E = {Ee:.6f}, $|T|^2$ = {tprob(Ee):.4f}')
    aP.set(xlabel='Momentum',ylabel=r'$|\phi(p)|^2$',title=rf'$\alpha={sc.alpha:.8g},\ p_0={sc.p0:.4f},\ \Delta p=\hbar\sqrt{{\alpha}}={sc.delta_p:.6f}$'); aP.grid(alpha=.2); aP.legend(fontsize=7)
    for ax,psi,t in zip(axs,psis,sc.snapshots): plotwave(ax,x,psi,t,sc.ylim); ax.set_xlim(*sc.xlim)
    axs[0].legend(loc='upper right',fontsize=7); axs[-1].set_xlabel('x'); fig.savefig(OUT/f'part4_{sc.name}.png',dpi=170); plt.close(fig)
    with (OUT/f'part4_{sc.name}_parameters.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['quantity','value']);
        for k,v in [('Emin',sc.Emin),('E0',sc.E0),('Emax',sc.Emax),('pmin',sc.pmin),('p0',sc.p0),('pmax',sc.pmax),('alpha',sc.alpha),('Delta_p',sc.delta_p),('x0',sc.x0)]: w.writerow([k,v])

def animation(sc,n_p,quick=False,local=False):
    if local:
        x=np.linspace(-60,60,1800); ylim=(-.1,.1); times=np.linspace(2600,7000,65 if quick else 110); suffix='_local'
    else:
        x=np.linspace(*sc.xlim,1200 if sc.name!='second_resonance' else 1600); ylim=sc.ylim; suffix=''
        ranges={'blocked':(-25,70),'partially_blocked':(-30,105),'pass':(-25,65),'second_resonance':(-100,11000)}; lo,hi=ranges[sc.name]; times=np.linspace(lo,hi,65 if quick else 110)
    p,E,states,T,R,base=setup(sc,x,n_p); allpsi=evolve(E,states,base,times)
    fig,ax=plt.subplots(figsize=(10,4.7)); lr,=ax.plot([],[],color='red',label=r'Re $\Psi$'); li,=ax.plot([],[],color='blue',ls='--',label=r'Im $\Psi$'); lp,=ax.plot([],[],color='black',label=r'$|\Psi|^2$'); lv,=ax.plot(x,model_potential(x),color='0.65',label=r'$V(x)$'); ax.set_xlim(x[0],x[-1]); ax.set_ylim(*ylim); ax.set_xlabel('x'); ax.set_ylabel('amplitude'); ax.legend(loc='upper right'); title=ax.set_title(''); tt=ax.text(.02,.88,'',transform=ax.transAxes,fontsize=11)
    def upd(i):
        y=allpsi[i]; lr.set_data(x,y.real); li.set_data(x,y.imag); lp.set_data(x,np.abs(y)**2); title.set_text(rf'$E_{{min}}={sc.Emin:.4f},\ E_0={sc.E0:.4f},\ E_{{max}}={sc.Emax:.4f}$'+'\n'+rf'$p_0={sc.p0:.4f},\ \alpha={sc.alpha:.3g},\ \Delta p={sc.delta_p:.4g}$'); tt.set_text(rf'$t={times[i]:.1f}$'); return lr,li,lp,lv,title,tt
    FuncAnimation(fig,upd,frames=len(times),blit=False).save(OUT/f'part4_{sc.name}{suffix}.gif',writer=PillowWriter(fps=16)); plt.close(fig)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--scenario',choices=['all',*SCENARIOS,'second_resonance'],default='all'); ap.add_argument('--quick',action='store_true'); a=ap.parse_args(); ensure_dir(OUT); items=selected_scenarios(a.scenario); n_p=180 if a.quick else 300
    for sc in items: summary(sc,n_p); animation(sc,n_p,a.quick); animation(sc,n_p,a.quick,local=True) if sc.name=='second_resonance' else None
if __name__=='__main__': main()
