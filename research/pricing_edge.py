"""
DIAGNOSTIC DE PRICING (in-frame, 100% pricing-model).

Question : sur les binaires 5m Polymarket, existe-t-il une zone de moneyness /
un regime ou le PRIX DU MODELE (BS avec sigma = vol realisee trailing depuis les
klines 1s) predit la resolution MIEUX que le PRIX DU MARCHE ?

- Si oui (log-loss modele < log-loss marche) quelque part -> c'est la que le
  pricing model a de l'edge, on y concentre les ameliorations (skew, sigma, ATM).
- Si non, meme a l'ATM -> le pricing n'ajoute rien sur ces donnees.

On compare, bucke par |z| = |ln(S/K)| / (sigma*sqrt(tau)) (moneyness standardisee) :
   log-loss(p_model)  vs  log-loss(p_mkt=mid)  vs  outcome (0/1).
On mesure aussi si (p_model - p_mkt) predit (outcome - p_mkt)  -> edge conditionnel.
"""
import pandas as pd, numpy as np, glob, os, sys, time
from scipy.stats import norm

PM   = "/home/ryan/GitProject/stats_btc_poly/data/pmdata/btc_5m"
K1S  = "/home/ryan/GitProject/stats_btc_poly/data/btc_klines_1s"
YEAR_SEC = 365.25*24*3600

# ---------- Spot 1s ----------
_spot_ts = None; _spot_px = None
def load_spot(days):
    global _spot_ts, _spot_px
    ts_list, px_list = [], []
    for d in sorted(days):
        f = f"{K1S}/BTCUSDT-1s-{d}.csv"
        if not os.path.exists(f): continue
        a = pd.read_csv(f, header=None, usecols=[0,4]).values
        ts_list.append((a[:,0]/1e6).astype(np.int64))   # µs -> s
        px_list.append(a[:,1].astype(float))
    _spot_ts = np.concatenate(ts_list); _spot_px = np.concatenate(px_list)
    o = np.argsort(_spot_ts); _spot_ts=_spot_ts[o]; _spot_px=_spot_px[o]
    print(f"Spot 1s charge : {len(_spot_ts):,} secondes ({len(days)} jours)")

def spot_at(t):
    i = np.searchsorted(_spot_ts, t, side='right')-1
    return _spot_px[np.clip(i,0,len(_spot_px)-1)]

def realized_vol(t, window=180, step=5):
    """vol annualisee des rendements pas-de-'step's sur les 'window's precedentes."""
    i1 = np.searchsorted(_spot_ts, t, side='right')-1
    i0 = np.searchsorted(_spot_ts, t-window, side='right')-1
    if i1-i0 < 6: return None
    px = _spot_px[i0:i1+1:step]
    if len(px) < 4 or (px<=0).any(): return None
    r = np.diff(np.log(px))
    sd = np.std(r, ddof=1)
    if not np.isfinite(sd) or sd<=0: return None
    return sd*np.sqrt(YEAR_SEC/step)   # annualise (returns tous les 'step's)

def bs_up(S,K,sigma,tau):
    if tau<=0 or sigma<=0:
        return 1.0 if S>K else 0.0
    d2 = (np.log(S/K) - 0.5*sigma*sigma*tau)/(sigma*np.sqrt(tau))
    return float(norm.cdf(d2))

# ---------- une fenetre ----------
def process(path):
    fn = os.path.basename(path)
    start = int(fn.split('-')[-1].split('.')[0]); end = start+300
    df = pd.read_parquet(path, columns=['event_type','timestamp','best_bid',
                                        'best_ask','winning_outcome'])
    wo = df['winning_outcome'].dropna()
    if wo.empty: return None
    payout = 1.0 if wo.iloc[0]=='yes' else 0.0
    K = spot_at(start)
    if not np.isfinite(K) or K<=0: return None

    tob = df.loc[df['event_type']=='price_change',['timestamp','best_bid','best_ask']].dropna()
    tob = tob[(tob['best_bid']>0)&(tob['best_ask']>0)]
    tob = tob[tob['best_ask']>=tob['best_bid']].sort_values('timestamp')
    if len(tob)<10: return None
    tob_ts = (tob['timestamp'].values.astype('datetime64[s]').astype(np.int64))
    mid = ((tob['best_bid']+tob['best_ask'])/2.0).values

    rows=[]
    for t in range(start+30, end-15, 5):          # decisions toutes les 5s
        j = np.searchsorted(tob_ts, t, side='right')-1
        if j<0: continue
        p_mkt = mid[j]
        if not (0.01 < p_mkt < 0.99): continue
        S = spot_at(t); tau=(end-t)/YEAR_SEC
        sig = realized_vol(t)
        if sig is None: continue
        p_mod = bs_up(S,K,sig,tau)
        z = abs(np.log(S/K))/(sig*np.sqrt(tau)+1e-12)
        rows.append((z, tau, sig, p_mod, p_mkt, payout))
    return rows

def logloss(p,y):
    p=np.clip(p,1e-4,1-1e-4)
    return -(y*np.log(p)+(1-y)*np.log(1-p))

def main(nfiles=None):
    files = sorted(glob.glob(f"{PM}/*.parquet"))
    if nfiles: files=files[:nfiles]
    # jours 1s necessaires
    days=set()
    for f in files:
        start=int(os.path.basename(f).split('-')[-1].split('.')[0])
        for dd in (start-300, start, start+300):
            days.add(pd.to_datetime(dd,unit='s').strftime('%Y-%m-%d'))
    load_spot(days)

    allrows=[]; t0=time.time()
    for i,f in enumerate(files):
        try: r=process(f)
        except Exception as e: r=None
        if r: allrows.extend(r)
        if (i+1)%150==0: print(f"  ...{i+1}/{len(files)} ({time.time()-t0:.0f}s)")
    A=np.array(allrows)
    print(f"\n{len(A):,} points de decision ; {len(files)} marches.\n")
    z,tau,sig,pmod,pmkt,y = A.T

    # sanity : correlation prix marche <-> outcome (mapping token)
    print(f"corr(p_mkt, outcome) = {np.corrcoef(pmkt,y)[0,1]:+.3f}  (doit etre > 0)")
    print(f"corr(p_mod, outcome) = {np.corrcoef(pmod,y)[0,1]:+.3f}")
    print(f"sigma realisee mediane = {np.median(sig):.2f}  ({np.percentile(sig,10):.2f}-{np.percentile(sig,90):.2f})\n")

    print("="*78)
    print(" LOG-LOSS  modele(sigma realise) vs marche(mid),  par |z| standardise")
    print(" (z = |ln(S/K)|/(sigma*sqrt(tau)) ; petit z = ATM = la ou sigma compte)")
    print("="*78)
    print(f" {'bucket |z|':>14} | {'n':>7} | {'LL_marche':>10} | {'LL_modele':>10} | {'gain modele':>11}")
    print("-"*78)
    edges = [0,0.25,0.5,0.75,1.0,1.5,2.5,1e9]
    for a,b in zip(edges[:-1],edges[1:]):
        m=(z>=a)&(z<b)
        if m.sum()<50: continue
        llm=logloss(pmkt[m],y[m]).mean(); llo=logloss(pmod[m],y[m]).mean()
        lbl=f"[{a:.2f},{b:.2f})" if b<1e8 else f">={a:.2f}"
        star=" <==" if llo<llm else ""
        print(f" {lbl:>14} | {m.sum():>7} | {llm:>10.4f} | {llo:>10.4f} | {llm-llo:>+11.4f}{star}")
    llm=logloss(pmkt,y).mean(); llo=logloss(pmod,y).mean()
    print("-"*78)
    print(f" {'GLOBAL':>14} | {len(y):>7} | {llm:>10.4f} | {llo:>10.4f} | {llm-llo:>+11.4f}")
    print("="*78)

    # edge conditionnel : (p_mod-p_mkt) predit-il (outcome-p_mkt) a l'ATM ?
    print("\nEDGE CONDITIONNEL : le modele corrige-t-il le marche dans la bonne direction ?")
    print(" regression outcome ~ p_mkt + (p_mod - p_mkt), coef sur le terme correctif :")
    for lbl,m in [("ATM |z|<0.75",(z<0.75)),("hors ATM |z|>=0.75",(z>=0.75)),("tous",np.ones(len(z),bool))]:
        if m.sum()<100: continue
        d=(pmod[m]-pmkt[m])
        X=np.column_stack([np.ones(m.sum()),pmkt[m],d])
        beta,*_=np.linalg.lstsq(X,y[m],rcond=None)
        # correlation partielle simple du terme correctif avec residu
        yr=y[m]-pmkt[m]
        c=np.corrcoef(d,yr)[0,1]
        print(f"  {lbl:>20}: coef(correctif)={beta[2]:+.3f}  corr(p_mod-p_mkt, outcome-p_mkt)={c:+.3f}  n={m.sum()}")
    print("\nLecture : coef/corr > 0 et significatif a l'ATM = le modele apporte une")
    print("info que le prix n'a pas -> edge de pricing exploitable la. ~0 = redondant.")

if __name__=="__main__":
    main(int(sys.argv[1]) if len(sys.argv)>1 else None)
