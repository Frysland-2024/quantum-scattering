from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from pathlib import Path

src = Path('Yifeng CHEN 999016959/Quantum_Scattering_Project_Report.docx')
doc = Document(src)
paras = doc.paragraphs
idx = next(i for i,p in enumerate(paras) if p.text == 'The Dirac delta function is defined by its action on a smooth test function f(p):')
p32,p33,p34,p35,p36,p37 = paras[idx:idx+6]
assert p37._p.xpath('.//w:drawing'), 'Figure 4 image paragraph not found'
style = p32.style

def clear(p):
    for c in list(p._p):
        if c.tag != qn('w:pPr'):
            p._p.remove(c)

def body(p,text,after=1.5):
    clear(p); p.style=style; p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before=Pt(0); p.paragraph_format.space_after=Pt(after)
    r=p.add_run(text); r.font.name='Times New Roman'; r.font.size=Pt(10.5)

def insert_before(anchor,text=None,center=False,after=1.5):
    x=OxmlElement('w:p'); anchor._p.addprevious(x); q=Paragraph(x,anchor._parent); q.style=style
    q.alignment=WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.JUSTIFY
    q.paragraph_format.space_before=Pt(0); q.paragraph_format.space_after=Pt(after)
    if text is not None:
        r=q.add_run(text); r.font.name='Times New Roman'; r.font.size=Pt(10.5)
    return q

M='m:'
def me(tag,val=None):
    e=OxmlElement(M+tag)
    if val is not None: e.set(qn('m:val'),val)
    return e

def mr(text,italic=True):
    r=me('r'); rp=me('rPr'); rp.append(me('sty','i' if italic else 'p')); r.append(rp)
    t=me('t'); t.text=text; r.append(t); return r

def txt(parent,text,italic=True): parent.append(mr(text,italic))
def sub(parent,b,s,bi=True,si=True):
    z=me('sSub'); e=me('e'); txt(e,b,bi); ss=me('sub'); txt(ss,s,si); z.extend([e,ss]); parent.append(z)
def frac(parent,nb,db):
    f=me('f'); n=me('num'); d=me('den'); nb(n); db(d); f.extend([n,d]); parent.append(f)
def integ(parent,lo,hi,eb):
    n=me('nary'); pr=me('naryPr'); pr.append(me('chr','∫')); pr.append(me('limLoc','undOvr')); n.append(pr)
    a=me('sub'); txt(a,lo,False); b=me('sup'); txt(b,hi,False); e=me('e'); eb(e); n.extend([a,b,e]); parent.append(n)
def math(p,builder,after=1.5):
    clear(p); p.style=style; p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before=Pt(0); p.paragraph_format.space_after=Pt(after)
    op=me('oMathPara'); o=me('oMath'); builder(o); op.append(o); p._p.append(op)

body(p32,'The Dirac delta function is defined through its action on a smooth test function f(p):')
def e1(o):
    def e(x):
        for a,i in [('f',1),('(',0),('p',1),(')',0),('δ',1),('(',0),('p',1),(')',0),('d',1),('p',1)]: txt(x,a,bool(i))
    integ(o,'−∞','+∞',e)
    for a,i in [(' = ',0),('f',1),('(',0),('0',0),(')',0)]: txt(o,a,bool(i))
math(p33,e1)

body(p34,'More generally,',after=1)
def e2(o):
    def e(x):
        for a,i in [('f',1),('(',0),('p',1),(')',0),('δ',1),('(',0),('p',1),(' − ',0)]: txt(x,a,bool(i))
        sub(x,'p','0',True,False)
        for a,i in [(')',0),('d',1),('p',1)]: txt(x,a,bool(i))
    integ(o,'−∞','+∞',e)
    for a,i in [(' = ',0),('f',1),('(',0)]: txt(o,a,bool(i))
    sub(o,'p','0',True,False); txt(o,')',False)
math(p35,e2)

body(p36,'A Fourier representation of the same distribution is',after=1)
def e3(o):
    for a,i in [('δ',1),('(',0),('p',1),(')',0),(' = ',0)]: txt(o,a,bool(i))
    frac(o,lambda n:txt(n,'1',False),lambda d:(txt(d,'2',False),txt(d,'π',False))); txt(o,' ',False)
    def e(x):
        f=me('func'); f.append(me('funcPr')); fn=me('fName'); txt(fn,'exp',False); a=me('e')
        for t,i in [('(',0),('i',1),('p',1),('x',1),(')',0)]: txt(a,t,bool(i))
        f.extend([fn,a]); x.append(f); txt(x,'d',True); txt(x,'x',True)
    integ(o,'−∞','+∞',e)
q=insert_before(p37,center=True,after=1); math(q,e3,after=1)

insert_before(p37,'For the numerical calculation, the infinite x-range is truncated to [−L, L]. This gives the regularized delta function',after=1)
def e4(o):
    sub(o,'δ','L');
    for a,i in [('(',0),('p',1),(')',0),(' = ',0)]: txt(o,a,bool(i))
    frac(o,lambda n:txt(n,'1',False),lambda d:(txt(d,'2',False),txt(d,'π',False))); txt(o,' ',False)
    def e(x):
        f=me('func'); f.append(me('funcPr')); fn=me('fName'); txt(fn,'exp',False); a=me('e')
        for t,i in [('(',0),('i',1),('p',1),('x',1),(')',0)]: txt(a,t,bool(i))
        f.extend([fn,a]); x.append(f); txt(x,'d',True); txt(x,'x',True)
    integ(o,'−L','L',e); txt(o,' = ',False)
    def n(x):
        f=me('func'); f.append(me('funcPr')); fn=me('fName'); txt(fn,'sin',False); a=me('e')
        for t,i in [('(',0),('L',1),('p',1),(')',0)]: txt(a,t,bool(i))
        f.extend([fn,a]); x.append(f)
    frac(o,n,lambda d:(txt(d,'π',False),txt(d,'p',True)))
q=insert_before(p37,center=True,after=1); math(q,e4,after=1)

insert_before(p37,'As L becomes very large, the regularized function approaches the Dirac delta distribution, so that',after=1)
def e5(o):
    def e(x):
        for a,i in [('f',1),('(',0),('p',1),(')',0)]: txt(x,a,bool(i))
        sub(x,'δ','L')
        for a,i in [('(',0),('p',1),(')',0),('d',1),('p',1)]: txt(x,a,bool(i))
    integ(o,'−∞','+∞',e)
    for a,i in [(' → ',0),('f',1),('(',0),('0',0),(')',0)]: txt(o,a,bool(i))
q=insert_before(p37,center=True,after=2); math(q,e5,after=2)

doc.save(src)
print('restored Part 2')
