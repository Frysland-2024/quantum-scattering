from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from quantum_scattering.core import gaussian_p, regularized_delta, ensure_dir
OUT=Path(__file__).resolve().parent/'outputs'/'part2'


def delta_figure():
    p=np.linspace(-2,2,40001)
    fig,axs=plt.subplots(1,3,figsize=(15,5))
    for ax,L in zip(axs,[10.0,100.0,1000.0]):
        ax.plot(p,regularized_delta(p,L),color='black',label=rf'$\delta_L(p)=\dfrac{{\sin(pL)}}{{\pi p}}$\n$L={L:.0e}$')
        ax.set_xlim(-2,2); ax.set_xlabel('Momentum'); ax.set_ylabel('Amplitude'); ax.legend(loc='upper right')
    fig.suptitle('regularized delta function',fontsize=22,fontweight='bold'); fig.tight_layout(); fig.savefig(OUT/'part2_regularized_delta.png',dpi=180); plt.close(fig)


def test_table():
    alpha,p0=1.0,5.0; p_reference=np.linspace(-10.0,10.0,5000); phi0=float(np.real(gaussian_p(np.array([0.0]),alpha,p0)[0]))
    rows=[]
    for L in [10.0,100.0,1000.0]:
        if L == 1000.0:
            target_dp=(np.pi/L)/32.0
            n_grid=int(np.ceil(20.0/target_dp))+1
            if n_grid % 2 == 0: n_grid+=1
            p=np.linspace(-10.0,10.0,n_grid)
        else:
            p=p_reference
        phi=np.real(gaussian_p(p,alpha,p0))
        value=float(np.trapezoid(phi*regularized_delta(p,L),p)); rows.append((L,value,phi0,abs(value-phi0)))
    fig,ax=plt.subplots(figsize=(12,3)); ax.axis('off')
    cell=[[f'{r[1]:.8f}' for r in rows],[f'{phi0:.8f}']*3,[f'{r[3]:.8f}' for r in rows]]
    tab=ax.table(cellText=cell,rowLabels=[r'$\int_{-L}^{L}dp\,\phi(p)\,\delta_L(p)$',r'$\phi(p=0)$','Absolute Difference'],colLabels=[r'$10^1$',r'$10^2$',r'$10^3$'],loc='center',cellLoc='center'); tab.auto_set_font_size(False); tab.set_fontsize(13); tab.scale(1,1.7)
    ax.set_title('example of a test calculation',fontsize=22,fontweight='bold',loc='left'); fig.tight_layout(); fig.savefig(OUT/'part2_delta_test_table.png',dpi=180,bbox_inches='tight'); plt.close(fig)


def main(): ensure_dir(OUT); delta_figure(); test_table()
if __name__=='__main__': main()
