#!/usr/bin/env python3
"""Bayesian optimization with Gaussian process surrogate."""
import random, math, sys

class GaussianProcess:
    def __init__(self, length_scale=1.0):
        self.ls=length_scale; self.X=[]; self.y=[]
    def _kernel(self, x1, x2):
        return math.exp(-sum((a-b)**2 for a,b in zip(x1,x2))/(2*self.ls**2))
    def fit(self, X, y): self.X=X; self.y=y
    def predict(self, x):
        if not self.X: return 0.0, 1.0
        k=[self._kernel(x,xi) for xi in self.X]
        K=[[self._kernel(xi,xj)+1e-6*(i==j) for j,xj in enumerate(self.X)] for i,xi in enumerate(self.X)]
        # Simple approximate: weighted mean + uncertainty
        sk=sum(k)+1e-10
        mu=sum(ki*yi for ki,yi in zip(k,self.y))/sk
        sigma=max(0.01, 1.0-sum(ki**2 for ki in k)/sk)
        return mu, sigma

def bayesian_optimize(fn, bounds, n_init=5, n_iter=20):
    dims=len(bounds)
    X=[[random.uniform(lo,hi) for lo,hi in bounds] for _ in range(n_init)]
    y=[fn(x) for x in X]
    gp=GaussianProcess()
    for i in range(n_iter):
        gp.fit(X,y)
        best_acq=-1e10; best_x=None
        for _ in range(200):
            cand=[random.uniform(lo,hi) for lo,hi in bounds]
            mu,sigma=gp.predict(cand)
            acq=-mu+1.96*math.sqrt(sigma)  # UCB
            if acq>best_acq: best_acq=acq; best_x=cand
        val=fn(best_x); X.append(best_x); y.append(val)
    best_i=min(range(len(y)),key=lambda i:y[i])
    return X[best_i], y[best_i]

if __name__ == "__main__":
    random.seed(42)
    fn=lambda x: (x[0]-3)**2+(x[1]+2)**2+math.sin(x[0]*3)*5
    best,val=bayesian_optimize(fn,[(-5,10),(-5,10)],n_iter=30)
    print(f"Best: ({best[0]:.3f}, {best[1]:.3f}), val={val:.3f}")
