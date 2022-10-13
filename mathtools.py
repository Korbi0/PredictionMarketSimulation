import sympy as sp
from sympy.abc import b, x, y
import numpy as np
from scipy.stats import norm, truncnorm
from matplotlib import pyplot as plt


lmsr_cost = sp.Symbol('b') * sp.log(sp.exp(-sp.Symbol('x')/sp.Symbol('b')) + sp.exp(-sp.Symbol('y')/sp.Symbol('b')))

lmsr_cost_function = sp.lambdify((b, x, y), lmsr_cost, "numpy")


def truncated_distribution(mean, std, lower_bound, upper_bound):
  clip_a, mean, std, clip_b = lower_bound, mean, std, upper_bound

  a, b = (clip_a - mean) / std, (clip_b - mean) / std

  dist = truncnorm(loc=mean, scale=std, a=a, b=b)
  
  return dist

def normal_distribution(mean, std, lower_bound, upper_bound):
  dist = norm(loc=mean, scale=std)
  
  return dist


def plot_distribution(dist, resolution):
  x = np.linspace(0, 1, resolution)
  y = dist.pdf(x)
  plt.plot(x, y)
  return x, y


def expected_income(exp_v, exp_a_unc, ep_unc_exp_v, ep_unc_a_unc, resolution1, distribution1="norm", distribution2="norm"):
  """
  Expected value
  expected aleatoric uncertainty
  epistemic uncertainty about the expected value
  epistemic uncertainty about the aleatoric uncertainty
  """

  if distribution1 == "norm":
    mu_a = norm(loc=exp_v, scale=ep_unc_exp_v)
    sigma_a = norm(exp_a_unc, ep_unc_a_unc)
  elif distribution1 == "truncnorm":
    mu_a = truncated_distribution(exp_v, ep_unc_exp_v, 0, 1)
    sigma_a = truncated_distribution(exp_a_unc, ep_unc_a_unc, 0, 100)

  

  x = 0
  for i in range(resolution1):
    mu = mu_a.rvs()
    sigma = sigma_a.rvs()
    
    if distribution2 == "norm":
      val = norm(loc=mu, scale=sigma).expect(lambda x: x, lb=0., ub=1.)
    elif distribution2 == "truncnorm":
      val = truncated_distribution(mu, sigma, 0, 1).expect(lambda x: x, lb=0., ub=1.)
    x += val

  result = x / resolution1
  
  return result

def get_expected_income(expected_value, aleatory_uncertainty):
  return norm(loc=expected_value, scale=aleatory_uncertainty).expect(lambda x: x, lb=0, ub=1.)

