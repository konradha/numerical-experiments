import torch
import numpy as np
import scipy.special as sps

def lgwt01(N):
    x, w = sps.roots_legendre(N)
    # [-1,1] to [0,1]
    x = (x + 1) / 2
    w /= 2
    x = torch.tensor(x, dtype=torch.float32)
    w = torch.tensor(w, dtype=torch.float32) 
    return x, w

def trova(x, xs):
    indr = list(range(len(x)))
    inds = []
    xx = x.clone()
    for xi in xs:
        ind = torch.argmin(torch.abs(xx - xi))
        inds.append(indr[ind])
        del indr[ind]
        xx = torch.cat([xx[:ind], xx[ind+1:]])
    return torch.tensor(inds), torch.tensor(indr)

def ghbvmks2(k, s):
    r = k - s
    x, beta = lgwt01(k)
    x, ind = torch.sort(x)
    beta = beta[ind]

    from scipy.special import lpmv
    from scipy import special
    def legendre_matlab(n, x):
        result = np.zeros((n+1,) + x.shape)
        for m in range(n+1):
            result[m] = lpmv(m, n, x)
        return result
 
    if r > 0:
        inds, indr = trova(x, torch.arange(1, s+1) / (s+1))
    else:
        inds = torch.arange(s)
        indr = torch.tensor([])
    
    W = torch.zeros(k, s+1)
    for i in range(s+1): 
        W[:, i] = torch.tensor(special.eval_legendre(i, 2 * x.cpu().numpy() - 1))     
    W1 = torch.zeros(k, s)

    W1[:, 0] = x 
    W1[:, 1:s] = (W[:, 2:(s+1)] - W[:, 0:(s-1)]) / 2

    

    A = torch.zeros(k, k+1)
    B = A.clone()

    A[:s, 0] = -torch.ones(s)    
    for i in range(-1, s-1):
        A[i+1, i+3] = 1



    B[:s, 1:] = W1[inds, :s] @ W[:, :s].T @ torch.diag(beta) 

        
    #print("LHS", A[s:(k+1), torch.cat([torch.tensor([0]), inds + 1])])
    #print("t1", - W1[indr, :(s+1)])
    #print("tsolve", torch.linalg.solve(W1[inds, :s], A[:s, torch.cat([torch.tensor([0]), inds + 1])]))
    #raise Exception 
    
    A[s:k, torch.cat([torch.tensor([0]), inds + 1])] =\
            - W1[indr, :s] @ torch.linalg.solve(W1[inds, :s], A[:s, torch.cat([torch.tensor([0]), inds+1])])

    #print("LHS", A[s:k, torch.cat([torch.tensor([0]), inds + 1])])
    #print("m", - W1[indr, :s])
    #print("inv", torch.linalg.solve(W1[inds, :s], A[:s, torch.cat([torch.tensor([0]), inds+1])]))
    #raise Exception
    
    A[s:k, 0] -= 1
    A[s:k, indr+1] = torch.eye(r)

    
    return x, beta, A, B, inds, indr

# Example usage
k, s = 5, 3
x, beta, A, B, inds, indr = ghbvmks2(k, s)
print("A:", A)
print("B:", B)
#print("x:", x)
#print("beta:", beta)
#print("A shape:", A.shape)
#print("B shape:", B.shape)
#print("inds:", inds)
#print("indr:", indr)
#
