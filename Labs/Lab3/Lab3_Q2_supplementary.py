
def dIdT(l, T):
    return np.pi*(c**3)*(h**2) / (k*(T**2)*(l**6)*(-1+np.cosh( c * h /(k*T*l) )))

def dEdT(t, l1, l2, N):
    pos, w = gaussxwab(N, l1, l2)
    L, T = np.meshgrid(pos, t)
    return np.sum(w*dIdT(L, T), axis=1)

def dE_totaldT(T):
    return 8*(k**4)*(np.pi**5)*(T**3)/(15*(c**2)*(h**3))

def d_eta(T, l1, l2, N):
    """d_eta = dE/Et - E/Et^2 * dEt
        dE = \int_l1^l2 dI/dt 
    """
    E_tot = E_total(T)
    return dEdT(T, l1, l2, N)/E_tot - dE_totaldT(T)*E(T, l1, l2, N)/(E_tot**2)
