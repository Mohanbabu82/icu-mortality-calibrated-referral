"""Monotonicity stress test for DWAC and DWAC-M (Appendix A.3).
Extracts the calibrator classes verbatim from the analysis notebook and fits them to 100 simulated
calibration sets. Expected: DWAC non-monotone in most sets; DWAC-M never. Runtime ~5-8 min on CPU.
Usage: python tests/test_dwac_monotonicity.py [path/to/notebook.ipynb]
"""
import json, numpy as np, re, sys
import os
NB=sys.argv[1] if len(sys.argv)>1 else os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','wids_calibration_study.ipynb')
nb=json.load(open(NB))
src='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
# pull the metric + calibrator definitions exactly as in the user's notebook
a=src.index('EPS = 1e-6'); b=src.index('CALIBRATORS = {')
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score
g={'np':np,'stats':stats,'IsotonicRegression':IsotonicRegression,'average_precision_score':average_precision_score}
exec(src[a:b],g)
DWAC,DWACM,mv=g['DWAC'],g['DWACMonotone'],None
def viol(f,lo=-12,hi=12,n=8000):
    s=1/(1+np.exp(-np.linspace(lo,hi,n))); o=f.transform(s); return int((np.diff(o)<-1e-12).sum())
r=np.random.default_rng(0); res=[]
for t in range(100):
    n=int(r.integers(300,4000)); prev=r.uniform(0.02,0.4)
    z=r.normal(np.log(prev/(1-prev)),r.uniform(0.8,2.5),n)
    y=(r.random(n)<1/(1+np.exp(-(r.uniform(0.5,1.5)*z+r.normal(0,0.5))))).astype(int)
    p=1/(1+np.exp(-z))
    d=DWAC().fit(p,y); m=DWACM().fit(p,y)
    gd=np.linspace(-6,6,4000); od=d.transform(1/(1+np.exp(-gd))); om=m.transform(1/(1+np.exp(-gd)))
    lv=np.log(p/(1-p)); gi=np.linspace(lv.min(),lv.max(),4000); si=1/(1+np.exp(-gi))
    inr=float(np.max(np.abs(d.transform(si)-m.transform(si))))
    res.append((viol(d),viol(m), float(np.max(np.abs(od-om))), inr, n))
res=np.array(res)
import json
q=lambda a:[float(np.percentile(a,x)) for x in (0,25,50,75,100)]
json.dump({'dwac_steps_q':q(res[:,0]),'dwacm_steps_q':q(res[:,1]),'maxdiff_q':q(res[:,2]),'frac_maxdiff_gt_0.01':float((res[:,2]>0.01).mean()),'frac_gt_0.05':float((res[:,2]>0.05).mean()),'inrange_maxdiff_q':q(res[:,3]),'n_q':q(res[:,4]),'corr_steps_n':float(np.corrcoef(res[:,0],res[:,4])[0,1])},open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'simulation_distributions.json'),'w'),indent=1)
print('trials',len(res))
print('DWAC non-monotone in', int((res[:,0]>0).sum()),'/',len(res),' max decreasing steps',int(res[:,0].max()))
print('DWAC-M non-monotone in', int((res[:,1]>0).sum()),'/',len(res))
print('max |DWAC-M - DWAC| on [-6,6] logits: median %.2e, max %.2e'%(np.median(res[:,2]),res[:,2].max()))
# ranks preserved -> AUROC identical
y=(r.random(5000)<0.1).astype(int); p=np.clip(r.beta(1,8,5000)+0.3*y*r.random(5000),0,1)
m=DWACM().fit(p[:2500],y[:2500]); q=m.transform(p[2500:])
order=np.argsort(p[2500:],kind='stable'); print('rank reversals on held-out:',int((np.diff(q[order])<-1e-12).sum()))
print('outside-grid behaviour (constant clamp):', m.transform(np.array([1e-9,1-1e-9])))

assert int((res[:,1]>0).sum())==0, 'DWAC-M produced a monotonicity violation'
print('PASS: DWAC-M monotone in all trials')
