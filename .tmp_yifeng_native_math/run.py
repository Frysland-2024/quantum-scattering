from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt
from pathlib import Path

SRC = Path('Yifeng CHEN 999016959/Quantum_Scattering_Project_Report.docx')
OUT = SRC
doc = Document(SRC)

M='m:'

def me(tag,val=None):
    e=OxmlElement(M+tag)
    if val is not None:
        e.set(qn('m:val'),str(val))
    return e

def clear_para(p):
    for c in list(p._p):
        if c.tag != qn('w:pPr'):
            p._p.remove(c)

def set_display(p,builder,after=10):
    clear_para(p)
    p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before=Pt(0)
    p.paragraph_format.space_after=Pt(after/20)
    omp=me('oMathPara'); om=me('oMath'); builder(om); omp.append(om); p._p.append(omp)

def run(text,italic=True):
    r=me('r'); rp=me('rPr'); rp.append(me('sty','i' if italic else 'p')); r.append(rp)
    t=me('t'); t.text=text; r.append(t); return r

def t(parent,text,italic=True): parent.append(run(text,italic))
def tnor(parent,text):
    r=me('r'); rp=me('rPr'); rp.append(me('nor','1')); r.append(rp); tt=me('t'); tt.text=text; r.append(tt); parent.append(r)
def plain(text): return lambda p:t(p,text,False)
def var(text): return lambda p:t(p,text,True)

def sub(parent,base,subb):
    z=me('sSub'); e=me('e'); base(e); s=me('sub'); subb(s); z.extend([e,s]); parent.append(z)
def sup(parent,base,supp):
    z=me('sSup'); e=me('e'); base(e); s=me('sup'); supp(s); z.extend([e,s]); parent.append(z)
def subsup(parent,base,subb,supp):
    z=me('sSubSup'); e=me('e'); base(e); s1=me('sub'); subb(s1); s2=me('sup'); supp(s2); z.extend([e,s1,s2]); parent.append(z)
def frac(parent,num,den):
    z=me('f'); n=me('num'); num(n); d=me('den'); den(d); z.extend([n,d]); parent.append(z)
def rad(parent,expr):
    z=me('rad'); pr=me('radPr'); pr.append(me('degHide','1')); z.append(pr); z.append(me('deg')); e=me('e'); expr(e); z.append(e); parent.append(z)
def func(parent,name,arg):
    z=me('func'); z.append(me('funcPr')); fn=me('fName'); t(fn,name,False); e=me('e'); arg(e); z.extend([fn,e]); parent.append(z)
def delim(parent,beg,end,inner):
    z=me('d'); pr=me('dPr'); pr.append(me('begChr',beg)); pr.append(me('endChr',end)); z.append(pr); e=me('e'); inner(e); z.append(e); parent.append(z)
def nary(parent,symbol,expr,lo=None,hi=None):
    z=me('nary'); pr=me('naryPr'); pr.append(me('chr',symbol)); pr.append(me('limLoc','undOvr'))
    if lo is None: pr.append(me('subHide','1'))
    if hi is None: pr.append(me('supHide','1'))
    z.append(pr)
    sl=me('sub');
    if lo is not None: lo(sl)
    su=me('sup');
    if hi is not None: hi(su)
    e=me('e'); expr(e)
    z.extend([sl,su,e]); parent.append(z)
def acc(parent,base,chr_):
    z=me('acc'); pr=me('accPr'); pr.append(me('chr',chr_)); z.append(pr); e=me('e'); base(e); z.append(e); parent.append(z)
def bar(parent,base):
    z=me('bar'); pr=me('barPr'); pr.append(me('pos','top')); z.append(pr); e=me('e'); base(e); z.append(e); parent.append(z)
def absval(parent,inner): delim(parent,'|','|',inner)

def find(text):
    for p in doc.paragraphs:
        if p.text == text: return p
    raise RuntimeError(f'paragraph not found: {text}')

def x0(p): sub(p,var('x'),plain('0'))
def p0(p): sub(p,var('p'),plain('0'))
def up(p): sub(p,var('u'),var('p'))
def psiE(p): sub(p,var('ψ'),var('E'))
def psibarE(p): sub(p,lambda q:bar(q,var('ψ')),var('E'))
def phij(p): sub(p,var('φ'),var('j'))
def phii(p): sub(p,var('φ'),var('i'))
def psin(p): sub(p,var('ψ'),var('n'))
def Tij(p): sub(p,var('T'),lambda q:(t(q,'i',True),t(q,'j',True)))
def Vij(p): sub(p,var('V'),lambda q:(t(q,'i',True),t(q,'j',True)))
def Hij(p): sub(p,var('H'),lambda q:(t(q,'i',True),t(q,'j',True)))
def deltaij(p): sub(p,var('δ'),lambda q:(t(q,'i',True),t(q,'j',True)))
def Hhat(p): acc(p,var('H'),'̂')
def Htheta(p): sub(p,var('H'),var('θ'))
def En(p): sub(p,var('E'),var('n'))
def Gaman(p): sub(p,var('Γ'),var('n'))
def Eresn(p): sub(p,var('E'),lambda q:(t(q,'res',False),t(q,',',False),t(q,'n',True)))
def expf(parent,inside): func(parent,'exp',lambda e:delim(e,'(',')',inside))

# Part 1
set_display(find('ψ(0,x) = (2a/π)¹ᐟ⁴ exp[−a(x−x₀)²] exp[i p₀(x−x₀)]'), lambda o:(t(o,'ψ',True),t(o,'(',False),t(o,'0',False),t(o,',',False),t(o,'x',True),t(o,') = ',False),sup(o,lambda e:delim(e,'(',')',lambda q:(t(q,'2',False),t(q,'a',True),t(q,'/',False),t(q,'π',False))),lambda ss:frac(ss,lambda n:t(n,'1',False),lambda d:t(d,'4',False))),t(o,' ',False),expf(o,lambda q:(t(q,'−',False),t(q,'a',True),sup(q,lambda r:delim(r,'(',')',lambda z:(t(z,'x',True),t(z,'−',False),x0(z))),plain('2')))),t(o,' ',False),expf(o,lambda q:(t(q,'i',True),t(q,' ',False),p0(q),delim(q,'(',')',lambda r:(t(r,'x',True),t(r,'−',False),x0(r)))))))
set_display(find('−(1/2) d²uₚ(x)/dx² = E uₚ(x).'), lambda o:(t(o,'−',False),delim(o,'(',')',lambda e:(t(e,'1',False),t(e,'/2',False))),t(o,' ',False),sup(o,var('d'),plain('2')),up(o),t(o,'(',False),t(o,'x',True),t(o,')',False),t(o,'/',False),t(o,'d',True),sup(o,var('x'),plain('2')),t(o,' = ',False),t(o,'E',True),t(o,' ',False),up(o),t(o,'(',False),t(o,'x',True),t(o,').',False)))
set_display(find('uₚ(x) = 1/√(2π) exp(ipx).'), lambda o:(up(o),t(o,'(',False),t(o,'x',True),t(o,') = ',False),frac(o,lambda n:t(n,'1',False),lambda d:rad(d,lambda e:(t(e,'2',False),t(e,'π',False)))),t(o,' ',False),expf(o,lambda e:(t(e,'i',True),t(e,'p',True),t(e,'x',True))),t(o,'.',False)))
set_display(find('Φ(p) = 1/√(2π) ∫ψ(0,x) exp(−ipx) dx'), lambda o:(t(o,'Φ',True),t(o,'(',False),t(o,'p',True),t(o,') = ',False),frac(o,lambda n:t(n,'1',False),lambda d:rad(d,lambda e:(t(e,'2',False),t(e,'π',False)))),t(o,' ',False),nary(o,'∫',lambda e:(t(e,'ψ',True),t(e,'(',False),t(e,'0',False),t(e,',',False),t(e,'x',True),t(e,')',False),t(e,' ',False),expf(e,lambda q:(t(q,'−',False),t(q,'i',True),t(q,'p',True),t(q,'x',True))),t(e,' ',False),t(e,'d',True),t(e,'x',True)))))
set_display(find('ψ(t,x) = 1/√(2π) ∫Φ(p) exp(ipx) exp(−ip²t/2) dp'), lambda o:(t(o,'ψ',True),t(o,'(',False),t(o,'t',True),t(o,',',False),t(o,'x',True),t(o,') = ',False),frac(o,lambda n:t(n,'1',False),lambda d:rad(d,lambda e:(t(e,'2',False),t(e,'π',False)))),t(o,' ',False),nary(o,'∫',lambda e:(t(e,'Φ',True),t(e,'(',False),t(e,'p',True),t(e,')',False),t(e,' ',False),expf(e,lambda q:(t(q,'i',True),t(q,'p',True),t(q,'x',True))),t(e,' ',False),expf(e,lambda q:(t(q,'−',False),t(q,'i',True),sup(q,var('p'),plain('2')),t(q,'t',True),t(q,'/2',False))),t(e,' ',False),t(e,'d',True),t(e,'p',True)))))

# Part 3
set_display(find('V(x) = (x²/2 − 0.8) exp(−0.1x²), '), lambda o:(t(o,'V',True),t(o,'(',False),t(o,'x',True),t(o,') = ',False),delim(o,'(',')',lambda e:(frac(e,lambda n:sup(n,var('x'),plain('2')),lambda d:t(d,'2',False)),t(e,' − 0.8',False))),t(o,' ',False),expf(o,lambda e:(t(e,'−0.1',False),sup(e,var('x'),plain('2')))),t(o,',',False)))
set_display(find('ψ̄E(x→+∞) = exp(ikx).'), lambda o:(psibarE(o),t(o,'(',False),t(o,'x',True),t(o,' → +∞',False),t(o,') = ',False),expf(o,lambda e:(t(e,'i',True),t(e,'k',True),t(e,'x',True))),t(o,'.',False)))
set_display(find('ψ̄E(xₙ₋₁) = [2 + 2(Δx)²(V(xₙ)−E)] ψ̄E(xₙ) − ψ̄E(xₙ₊₁).'), lambda o:(psibarE(o),t(o,'(',False),sub(o,var('x'),lambda s:t(s,'n−1',False)),t(o,') = ',False),delim(o,'[',']',lambda e:(t(e,'2 + 2',False),sup(e,lambda q:delim(q,'(',')',lambda r:t(r,'Δx',False)),plain('2')),delim(e,'(',')',lambda q:(t(q,'V',True),t(q,'(',False),sub(q,var('x'),var('n')),t(q,') − ',False),t(q,'E',True))))),t(o,' ',False),psibarE(o),t(o,'(',False),sub(o,var('x'),var('n')),t(o,') − ',False),psibarE(o),t(o,'(',False),sub(o,var('x'),lambda s:t(s,'n+1',False)),t(o,').',False)))
set_display(find('ψ̄E(x→−∞) = A(E) exp(ikx) + B(E) exp(−ikx).'), lambda o:(psibarE(o),t(o,'(',False),t(o,'x',True),t(o,' → −∞',False),t(o,') = ',False),t(o,'A',True),t(o,'(',False),t(o,'E',True),t(o,') ',False),expf(o,lambda e:(t(e,'i',True),t(e,'k',True),t(e,'x',True))),t(o,' + ',False),t(o,'B',True),t(o,'(',False),t(o,'E',True),t(o,') ',False),expf(o,lambda e:(t(e,'−',False),t(e,'i',True),t(e,'k',True),t(e,'x',True))),t(o,'.',False)))
set_display(find('ψE(x) = ψ̄E(x)/A(E),'), lambda o:(psiE(o),t(o,'(',False),t(o,'x',True),t(o,') = ',False),frac(o,lambda n:(psibarE(n),t(n,'(',False),t(n,'x',True),t(n,')',False)),lambda d:(t(d,'A',True),t(d,'(',False),t(d,'E',True),t(d,')',False))),t(o,',',False)))
set_display(find('T(E) = 1/A(E),     R(E) = B(E)/A(E).'), lambda o:(t(o,'T',True),t(o,'(',False),t(o,'E',True),t(o,') = ',False),frac(o,lambda n:t(n,'1',False),lambda d:(t(d,'A',True),t(d,'(',False),t(d,'E',True),t(d,')',False))),t(o,',      ',False),t(o,'R',True),t(o,'(',False),t(o,'E',True),t(o,') = ',False),frac(o,lambda n:(t(n,'B',True),t(n,'(',False),t(n,'E',True),t(n,')',False)),lambda d:(t(d,'A',True),t(d,'(',False),t(d,'E',True),t(d,')',False))),t(o,'.',False)))

# Part 4
set_display(find('Ψ(t,x) = ∫ Φ(p) ψp(x) exp(−i p²t/2) dp'), lambda o:(t(o,'Ψ',True),t(o,'(',False),t(o,'t',True),t(o,',',False),t(o,'x',True),t(o,') = ',False),nary(o,'∫',lambda e:(t(e,'Φ',True),t(e,'(',False),t(e,'p',True),t(e,') ',False),sub(e,var('ψ'),var('p')),t(e,'(',False),t(e,'x',True),t(e,') ',False),expf(e,lambda q:(t(q,'−',False),t(q,'i',True),sup(q,var('p'),plain('2')),t(q,'t',True),t(q,'/2',False))),t(e,' ',False),t(e,'d',True),t(e,'p',True)))))
set_display(find('Φ(p) = 1/√(2π) ∫ Ψ(0,x) exp(−ipx) dx'), lambda o:(t(o,'Φ',True),t(o,'(',False),t(o,'p',True),t(o,') = ',False),frac(o,lambda n:t(n,'1',False),lambda d:rad(d,lambda e:(t(e,'2',False),t(e,'π',False)))),t(o,' ',False),nary(o,'∫',lambda e:(t(e,'Ψ',True),t(e,'(',False),t(e,'0',False),t(e,',',False),t(e,'x',True),t(e,') ',False),expf(e,lambda q:(t(q,'−',False),t(q,'i',True),t(q,'p',True),t(q,'x',True))),t(e,' ',False),t(e,'d',True),t(e,'x',True)))))

# Part 5
set_display(find('φj(x) = √(2/L) sin[jπ(x + L/2)/L],    j = 1, 2, …, J.'), lambda o:(phij(o),t(o,'(',False),t(o,'x',True),t(o,') = ',False),rad(o,lambda e:frac(e,lambda n:t(n,'2',False),lambda d:t(d,'L',True))),t(o,' ',False),t(o,'sin',False),delim(o,'[',']',lambda q:(t(q,'j',True),t(q,'π',False),delim(q,'(',')',lambda r:(t(r,'x',True),t(r,' + ',False),t(r,'L',True),t(r,'/2',False))),t(q,'/',False),t(q,'L',True))),t(o,',      ',False),t(o,'j',True),t(o,' = 1, 2, …, ',False),t(o,'J',True),t(o,'.',False)))
set_display(find('ψn(x) = ∑j=1J cj(n) φj(x).'), lambda o:(psin(o),t(o,'(',False),t(o,'x',True),t(o,') = ',False),nary(o,'∑',lambda e:(subsup(e,var('c'),var('j'),lambda q:(t(q,'(',False),t(q,'n',True),t(q,')',False))),t(e,' ',False),phij(e),t(e,'(',False),t(e,'x',True),t(e,')',False)),lambda lo:(t(lo,'j',True),t(lo,' = 1',False)),lambda hi:t(hi,'J',True)),t(o,'.',False)))
set_display(find('Ĥ = −1/2 d²/dx² + V(x).'), lambda o:(acc(o,var('H'),'̂'),t(o,' = −',False),frac(o,lambda n:t(n,'1',False),lambda d:t(d,'2',False)),t(o,' ',False),frac(o,lambda n:sup(n,var('d'),plain('2')),lambda d:(t(d,'d',True),sup(d,var('x'),plain('2')))),t(o,' + ',False),t(o,'V',True),t(o,'(',False),t(o,'x',True),t(o,').',False)))
set_display(find('Hij = ⟨φi|Ĥ|φj⟩ = Tij + Vij.'), lambda o:(Hij(o),t(o,' = ',False),t(o,'⟨',False),phii(o),t(o,'|',False),acc(o,var('H'),'̂'),t(o,'|',False),phij(o),t(o,'⟩ = ',False),Tij(o),t(o,' + ',False),Vij(o),t(o,'.',False)))
set_display(find('Tij = j2π2 δij/(2L2),'), lambda o:(Tij(o),t(o,' = ',False),frac(o,lambda n:(sup(n,var('j'),plain('2')),sup(n,plain('π'),plain('2')),t(n,' ',False),deltaij(n)),lambda d:(t(d,'2',False),sup(d,var('L'),plain('2')))),t(o,',',False)))
set_display(find('Vij = ∫−L/2L/2 φi(x)V(x)φj(x) dx.'), lambda o:(Vij(o),t(o,' = ',False),nary(o,'∫',lambda e:(phii(e),t(e,'(',False),t(e,'x',True),t(e,')',False),t(e,'V',True),t(e,'(',False),t(e,'x',True),t(e,')',False),phij(e),t(e,'(',False),t(e,'x',True),t(e,') ',False),t(e,'d',True),t(e,'x',True)),lambda lo:frac(lo,lambda n:(t(n,'−',False),t(n,'L',True)),lambda d:t(d,'2',False)),lambda hi:frac(hi,lambda n:t(n,'L',True),lambda d:t(d,'2',False))),t(o,'.',False)))
set_display(find('Hc(n) = Enc(n).'), lambda o:(t(o,'H',True),t(o,' ',False),sup(o,var('c'),lambda q:(t(q,'(',False),t(q,'n',True),t(q,')',False))),t(o,' = ',False),En(o),t(o,' ',False),sup(o,var('c'),lambda q:(t(q,'(',False),t(q,'n',True),t(q,')',False))),t(o,'.',False)))

# Part 6
set_display(find('Hθ = exp(−2iθ)T + V[x exp(iθ)] = −(1/2) exp(−2iθ) d²/dx² + V[x exp(iθ)].'), lambda o:(sub(o,var('H'),var('θ')),t(o,' = ',False),expf(o,lambda e:(t(e,'−2',False),t(e,'i',True),t(e,'θ',True))),t(o,' ',False),t(o,'T',True),t(o,' + ',False),t(o,'V',True),delim(o,'[',']',lambda e:(t(e,'x',True),t(e,' ',False),expf(e,lambda q:(t(q,'i',True),t(q,'θ',True))))),t(o,' = −',False),frac(o,lambda n:t(n,'1',False),lambda d:t(d,'2',False)),t(o,' ',False),expf(o,lambda e:(t(e,'−2',False),t(e,'i',True),t(e,'θ',True))),t(o,' ',False),frac(o,lambda n:sup(n,var('d'),plain('2')),lambda d:(t(d,'d',True),sup(d,var('x'),plain('2')))),t(o,' + ',False),t(o,'V',True),delim(o,'[',']',lambda e:(t(e,'x',True),t(e,' ',False),expf(e,lambda q:(t(q,'i',True),t(q,'θ',True))))),t(o,'.',False)))
set_display(find('Hθ c⁽ⁿ⁾ = Eₙ(θ)c⁽ⁿ⁾.'), lambda o:(sub(o,var('H'),var('θ')),t(o,' ',False),sup(o,var('c'),lambda q:(t(q,'(',False),t(q,'n',True),t(q,')',False))),t(o,' = ',False),En(o),t(o,'(',False),t(o,'θ',True),t(o,')',False),sup(o,var('c'),lambda q:(t(q,'(',False),t(q,'n',True),t(q,')',False))),t(o,'.',False)))
set_display(find('Eₙ = Eres,n − iΓₙ/2,     Γₙ = −2 Im(Eₙ).'), lambda o:(En(o),t(o,' = ',False),sub(o,var('E'),lambda q:(t(q,'res',False),t(q,',',False),t(q,'n',True))),t(o,' − ',False),t(o,'i',True),Gaman(o),t(o,'/2',False),t(o,',      ',False),Gaman(o),t(o,' = −2 ',False),tnor(o,'I'),tnor(o,'m'),t(o,'(',False),En(o),t(o,').',False)))
set_display(find('|T(E)|² ≈ (Γₙ/2)² / [(E−Eres,n)² + (Γₙ/2)²].'), lambda o:(sup(o,lambda e:absval(e,lambda q:(t(q,'T',True),t(q,'(',False),t(q,'E',True),t(q,')',False))),plain('2')),t(o,' ≈ ',False),sup(o,lambda e:delim(e,'(',')',lambda q:(Gaman(q),t(q,'/2',False))),plain('2')),t(o,' / ',False),delim(o,'[',']',lambda q:(sup(q,lambda r:delim(r,'(',')',lambda s:(t(s,'E',True),t(s,' − ',False),sub(s,var('E'),lambda u:(t(u,'res',False),t(u,',',False),t(u,'n',True))))),plain('2')),t(q,' + ',False),sup(q,lambda r:delim(r,'(',')',lambda s:(Gaman(s),t(s,'/2',False))),plain('2')))),t(o,'.',False)))

head=find('1.3 Free propagation')
prev=head._p.getprevious()
if prev is not None and prev.xpath('.//w:br[@w:type="page"]'):
    prev.getparent().remove(prev)

doc.save(OUT)
print(OUT)
