import json, numpy as np, pandas as pd
from pathlib import Path
V2 = Path(__file__).resolve().parents[1] / 'analysis' / 'results' / 'v2'
pred=pd.read_parquet(f'{V2}/phase6/minute_pu/test_predictions_2025.parquet')
s=json.load(open(f'{V2}/phase6/minute_dataset_summary.json')); n_hours=s['years']['2025']['population_edge_5min_units']/12
fit=json.load(open(f'{V2}/phase6/minute_pu/metrics.json')); prior=fit['priors_per_unit']['injury_from_police_register']; lab=fit['fits']['pn_reliable__b30']['prior']
k=pred.kind.to_numpy(); m=k=='marginal'; c=k=='official_case'
h=lambda p:1-(1-np.clip(p,0,1))**12
def newton(z,y,w):
    X=np.c_[np.ones_like(z),z]; b=np.array([np.log(y.sum()/(w*(1-y)).sum())-np.average(z,weights=w),1.])
    for _ in range(50):
        p=1/(1+np.exp(-X@b)); g=X.T@(w*(y-p)); H=(X*(w*p*(1-p))[:,None]).T@X
        st=np.linalg.solve(H,g); b+=st if np.isfinite(st).all() else 0
        if np.abs(st).max()<1e-10: break
    cov=np.linalg.inv(H); return b, np.sqrt(np.diag(cov))
res={}
for name,p in {'pn_reliable':pred.pn_reliable__b30.to_numpy(),'pn_rescaled':np.clip(pred.pn_reliable__b30.to_numpy()*prior/lab,0,1),
 'nnpu_reliable':pred.nnpu_reliable__b30.to_numpy(),'nnpu_prop':pred.nnpu_reliable_propensity__b30.to_numpy(),'pn_official':pred.pn_official_hour__b30.to_numpy(),
 'pn_union':pred.pn_union__b30.to_numpy(),'mswspu':pred.mswspu_credibility__b30.to_numpy()}.items():
    hu,hc=np.clip(h(p[m]),1e-12,1-1e-12),np.clip(h(p[c]),1e-12,1-1e-12); z=np.log(np.r_[hu,hc]/(1-np.r_[hu,hc])); y=np.r_[np.zeros(m.sum()),np.ones(c.sum())]
    w=np.r_[np.full(m.sum(),(n_hours-c.sum())/m.sum()),np.ones(c.sum())]
    b,se=newton(z,y,w)
    res[name]={'ratio':float(hu.mean()*n_hours/c.sum()),'intercept':float(b[0]),'slope':float(b[1]),'slope_se':float(se[1])}
    print(name,{a:round(v,3) for a,v in res[name].items()})
json.dump(res,open(f'{V2}/phase10/calibration_slopes.json','w'),indent=1)
