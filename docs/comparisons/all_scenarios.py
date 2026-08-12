import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon

def W(pos, p, q):      return wasserstein_distance(pos, pos, p, q)
def JSD_nats(p, q):    return jensenshannon(np.array(p,float), np.array(q,float), base=np.e)**2
def JSD_bits(p, q):    return jensenshannon(np.array(p,float), np.array(q,float), base=2)**2
def PSI(p, q, eps=1e-9):
    p=np.clip(np.array(p,float),eps,None); q=np.clip(np.array(q,float),eps,None)
    return np.sum((p-q)*np.log(p/q))
def close(a, b, tol=5e-3): return abs(a-b) < tol

checks = []
def verify(label, got, expected):
    ok = close(got, expected)
    checks.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:42s} got={got:.4f}  doc={expected:.4f}")

# ---------- Example 1: continuous (scores 1-5) ----------
print("Example 1 — continuous (quiz scores 1..5)")
pos = np.array([1,2,3,4,5], float)
P = np.array([0.4,0.3,0.2,0.1,0.0]); Q = np.array([0.0,0.1,0.2,0.3,0.4])
verify("W1 = 2.0 grades",              W(pos,P,Q),            2.0)
verify("mean P = 2.0",                 np.sum(pos*P),         2.0)
verify("mean Q = 4.0",                 np.sum(pos*Q),         4.0)
M = 0.5*(P+Q)
verify("M is uniform 0.2",             M.mean(),              0.2)
assert np.allclose(M, 0.2), "M not uniform!"
kl = lambda a,b: np.sum(a[a>0]*np.log(a[a>0]/b[a>0]))
verify("KL(P||M) = 0.330",             kl(P,M),               0.330)
verify("KL(Q||M) = 0.330",             kl(Q,M),               0.330)
verify("JSD = 0.330 nats",             JSD_nats(P,Q),         0.330)
verify("JSD = 0.475 bits",             JSD_bits(P,Q),         0.475)

# ---------- Example 1b: latency (spacing 10) + standardization ----------
print("Example 1b — latency (ms, spacing 10) + standardization")
ms = np.array([10,20,30,40,50], float)
verify("W1 = 20 ms",                   W(ms,P,Q),             20.0)
mean_ref = np.sum(ms*P); sd_ref = np.sqrt(np.sum(ms**2*P)-mean_ref**2)
verify("mean_ref = 20 ms",             mean_ref,              20.0)
verify("sd_ref = 10 ms",               sd_ref,                10.0)
verify("W1 / sd_ref = 2.0 SDs",        W(ms,P,Q)/sd_ref,      2.0)
verify("W1 / mean_ref = 1.0",          W(ms,P,Q)/mean_ref,    1.0)
sec = ms/1000.0
sd_sec = np.sqrt(np.sum(sec**2*P)-(np.sum(sec*P))**2)
verify("seconds W1 = 0.02 s",          W(sec,P,Q),            0.02)
verify("seconds W1/sd = 2.0 (invariant)", W(sec,P,Q)/sd_sec,  2.0)
verify("JSD ms  = 0.330 nats",         JSD_nats(P,Q),         0.330)  # spacing-blind

# ---------- Example 2: binary (churn) ----------
print("Example 2 — binary (churn)")
b = np.array([0,1], float)
Pb = np.array([0.8,0.2]); Qb = np.array([0.5,0.5])
verify("W1 = 0.3 (= |dp|)",            W(b,Pb,Qb),            0.3)
verify("|dp| = 0.3",                   abs(0.5-0.2),          0.3)
Mb = 0.5*(Pb+Qb)
verify("M = [0.65,0.35] (first)",      Mb[0],                 0.65)
verify("JSD = 0.051 nats",             JSD_nats(Pb,Qb),       0.051)
verify("JSD = 0.073 bits",             JSD_bits(Pb,Qb),       0.073)
verify("PSI = 0.416",                  PSI(Pb,Qb),            0.416)

# ---------- Example 3: ordinal (satisfaction) ----------
print("Example 3 — ordinal (satisfaction 1..5)")
o = np.array([1,2,3,4,5], float)
Po = np.array([0,0,1,0,0.]); Qa = np.array([0,0,0,1,0.]); Qb2 = np.array([0,0,0,0,1.])
verify("W1(P,Qa) = 1",                 W(o,Po,Qa),            1.0)
verify("W1(P,Qb) = 2",                 W(o,Po,Qb2),           2.0)
verify("JSD(P,Qa) = 1 bit",            JSD_bits(Po,Qa),       1.0)
verify("JSD(P,Qb) = 1 bit",            JSD_bits(Po,Qb2),      1.0)
verify("JSD(P,Qa) = ln2 nats",         JSD_nats(Po,Qa),       np.log(2))
verify("PSI(P,Qa) = PSI(P,Qb) (equal)", PSI(Po,Qa)-PSI(Po,Qb2), 0.0)
assert close(JSD_bits(Po,Qa), JSD_bits(Po,Qb2)), "JSD should be equal"
assert not close(W(o,Po,Qa), W(o,Po,Qb2)), "Wasserstein should differ"
print("  [PASS] punchline: JSD equal for A & B, Wasserstein differs (1 vs 2)")

print(f"\n{sum(checks)}/{len(checks)} numeric checks passed; all assertions held."
      if all(checks) else "SOME CHECKS FAILED")