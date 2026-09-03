import numpy as np
from scipy.optimize import brentq

G=9.8
VC=3.0
R_SMOKE=10.0
T_SMOKE=20.0
RCYL=7.0
CX,CY=0.0,200.0
Z0,Z1=0.0,10.0

# Q1 fixed parameters
M0=np.array([20000.0,0.0,2000.0])
VM=300.0
F0=np.array([17800.0,0.0,1800.0])
VF=120.0
theta=np.pi
td=1.5
tau=3.6

vM=-VM*M0/np.linalg.norm(M0)
vF=np.array([VF*np.cos(theta),VF*np.sin(theta),0.0])
te=td+tau
Me=M0+vM*te
E=F0+vF*te+np.array([0.0,0.0,-0.5*G*tau**2])
vC=np.array([0.0,0.0,-VC])

def state_at_s(s):
    return Me+vM*s, E+vC*s

def candidate_points(s,nphi=360,nz=41):
    M,_=state_at_s(s)

    # upper/lower rims
    phi=np.linspace(0,2*np.pi,nphi,endpoint=False)
    x=CX+RCYL*np.cos(phi)
    y=CY+RCYL*np.sin(phi)
    top=np.column_stack([x,y,np.full(nphi,Z1)])
    bottom=np.column_stack([x,y,np.full(nphi,Z0)])

    # two tangent generators
    dx,dy=M[0]-CX,M[1]-CY
    Dh=np.hypot(dx,dy)
    alpha=np.arctan2(dy,dx)
    delta=np.arccos(np.clip(RCYL/Dh,-1.0,1.0))
    phis=[alpha+delta,alpha-delta]
    z=np.linspace(Z0,Z1,nz)

    generators=[]
    for ph in phis:
        generators.append(np.column_stack([
            np.full(nz,CX+RCYL*np.cos(ph)),
            np.full(nz,CY+RCYL*np.sin(ph)),
            z
        ]))

    return np.vstack([top,bottom,*generators])

def violation(s,nphi=360,nz=41):
    P=candidate_points(s,nphi,nz)
    M,C=state_at_s(s)
    A=C-M
    B=P-M
    B2=np.einsum("ij,ij->i",B,B)
    cr=np.cross(A[None,:],B)
    d=np.sqrt(np.einsum("ij,ij->i",cr,cr)/B2)
    lam=(B@A)/B2

    # <=0 iff all silhouette candidates satisfy d<=R and 0<=lambda<=1
    return max(d.max()-R_SMOKE,-lam.min(),lam.max()-1.0)

def solve(nphi=360,nz=41,nscan=400):
    xs=np.linspace(0,T_SMOKE,nscan+1)
    ys=np.array([violation(x,nphi,nz) for x in xs])
    roots=[]
    for i in range(nscan):
        if ys[i]*ys[i+1]<0:
            roots.append(brentq(lambda s: violation(s,nphi,nz),
                                xs[i],xs[i+1],xtol=1e-12,rtol=1e-12))
    cuts=sorted(set([0.0,T_SMOKE]+[round(float(r),12) for r in roots]))
    intervals=[]
    for a,b in zip(cuts[:-1],cuts[1:]):
        if violation((a+b)/2,nphi,nz)<=0:
            intervals.append((a,b))
    duration=sum(b-a for a,b in intervals)
    return duration,[(te+a,te+b) for a,b in intervals]

if __name__=="__main__":
    duration,intervals=solve(nphi=720,nz=41,nscan=400)
    print("intervals =",intervals)
    print("duration =",duration)
