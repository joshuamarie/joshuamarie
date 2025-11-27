import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class Family(ABC):
    """Abstract base class for GLM families"""
    
    @abstractmethod
    def linkfun(self, mu):
        """Link function g(mu)"""
        pass
    
    @abstractmethod
    def linkinv(self, eta):
        """Inverse link function mu = g^-1(eta)"""
        pass
    
    @abstractmethod
    def variance(self, mu):
        """Variance function V(mu)"""
        pass
    
    @abstractmethod
    def mu_eta(self, eta):
        """Derivative of inverse link dmu/deta"""
        pass
    
    @property
    @abstractmethod
    def family_name(self):
        pass


class Gaussian(Family):
    """Gaussian family with identity link"""
    
    def linkfun(self, mu):
        return mu
    
    def linkinv(self, eta):
        return eta
    
    def variance(self, mu):
        return np.ones_like(mu)
    
    def mu_eta(self, eta):
        return np.ones_like(eta)
    
    @property
    def family_name(self):
        return "gaussian"


class Poisson(Family):
    """Poisson family with log link"""
    
    def linkfun(self, mu):
        return np.log(mu)
    
    def linkinv(self, eta):
        return np.exp(eta)
    
    def variance(self, mu):
        return mu
    
    def mu_eta(self, eta):
        return np.exp(eta)
    
    @property
    def family_name(self):
        return "poisson"


class Binomial(Family):
    """Binomial family with logit link"""
    
    def linkfun(self, mu):
        return np.log(mu / (1 - mu))
    
    def linkinv(self, eta):
        return 1 / (1 + np.exp(-eta))
    
    def variance(self, mu):
        return mu * (1 - mu)
    
    def mu_eta(self, eta):
        exp_eta = np.exp(eta)
        return exp_eta / ((1 + exp_eta) ** 2)
    
    @property
    def family_name(self):
        return "binomial"


def glm_custom(formula, data, family = None, max_iter = 100, tol = 1e-7, eps = 1e-8):
    if family is None:
        family = Gaussian()
    
    """Python lacks formula interface, parse them manually instead"""
    parts = formula.split('~')
    if len(parts) != 2:
        raise ValueError("Formula must be in the form 'y ~ x1 + x2'")
    
    y_name = parts[0].strip()
    x_names = [x.strip() for x in parts[1].split('+')]
    
    y = data[y_name].values.reshape(-1, 1)
    
    # Python's own design matrix
    X_list = [np.ones((len(data), 1))] 
    var_names = ['Intercept']
    
    for x_name in x_names:
        X_list.append(data[x_name].values.reshape(-1, 1))
        var_names.append(x_name)
    
    X = np.hstack(X_list)
    
    n, p = X.shape
    
    beta = np.zeros((p, 1))
    converged = False
    
    # IRLS
    for i in range(max_iter):
        # 1. Calculate linear predictor
        eta = X @ beta
        
        # 2. Calculate fitted values
        mu = family.linkinv(eta)
        
        # 3. Calculate weights
        V = family.variance(mu).flatten()
        gradient = family.mu_eta(eta).flatten()
        
        """Add small constant for numerical stability"""
        V = np.maximum(V, eps)
        gradient = np.maximum(np.abs(gradient), eps) * np.sign(gradient)
        
        w_vec = (gradient ** 2) / V
        
        # 4. Working response
        z = eta + (y - mu) / gradient.reshape(-1, 1)
        
        # 5. Update coefficients by solving weighted least squares
        W = np.diag(w_vec)
        
        try:
            beta_new = np.linalg.solve(X.T @ W @ X, X.T @ W @ z)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ z)
        
        # 6. Check convergence
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            converged = True
            break
        
        beta = beta_new
    
    if not converged:
        print(f"Warning: Algorithm did not converge in {max_iter} iterations")
    
    return {
        'coefficients': dict(zip(var_names, beta.flatten())),
        'converged': converged,
        'iterations': i + 1,
        'beta_vector': beta
    }
  
