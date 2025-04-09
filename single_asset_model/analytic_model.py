import numpy as np
from scipy.integrate import quad

class analytic_solution:
    def __init__(self, ra, r, lambd, mu, sigma):
        self.a = (1-ra)/ra * sigma**2
        self.b = 2 / ra * (ra * r - r - lambd)
        self.c = (lambd + r) ** 2 / (ra * sigma**2)
        self.g = -lambd*mu*(lambd + r) / (ra * sigma**2)
        self.nu = (self.b ** 2 - 4 * self.a * self.c) ** 0.5
        self.d = lambd*mu/ra
        self.r = r
        self.ra = ra
        self.lambd = lambd
        self.sigma = sigma
        self.mu = mu

    def integrand(self, t):
        a,d,r,ra,sigma,mu,lambd = self.a, self.d, self.r, self.ra, self.sigma, self.mu, self.lambd
        term1 = (a / 2) * (self.B(t) ** 2)
        term2 = d * self.B(t)
        term3 = (sigma ** 2 / 2) * self.C(t)
        term4 = ((lambd*mu) ** 2) / (2 * ra * sigma ** 2)
        term5 = r
        return term1 + term2 + term3 + term4 + term5


    def A(self, tau):
        A, _ = quad(self.integrand, 0, tau)
        return A

    def B(self, tau):
        b,nu,g,r = self.b, self.nu, self.g, self.r
        return (-4*g*r * (1-np.exp(-nu*tau/2))**2 +2*g*nu*(1-np.exp(-nu*tau))) / (nu * (2*nu - (b+nu)*(1-np.exp(-nu*tau))))


    def C(self, tau):
        b,c,nu = self.b, self.c, self.nu
        return (2*c*(1-np.exp(-nu*tau)))/(2*nu-(b+nu)*(1-np.exp(-nu*tau)))

    def optimal_nocost(self, tau, X):
        ra,lambd,mu,r,sigma = self.ra, self.lambd, self.mu, self.r, self.sigma
        return (1/ra) * ((lambd*(mu-X)/X)-r)/((sigma/X)**2) + (1-ra)/ra * (self.C(tau)*X + self.B(tau))*X

    def J(self, W, X, tau):
        ra = self.ra
        return ((W * (np.exp(self.A(tau) + self.B(tau) * X + (self.C(tau) * (X**2)) / 2)))**(1 - ra) - 1)/(1 - ra)
