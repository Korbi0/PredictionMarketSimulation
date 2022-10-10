import numpy as np
from scipy.stats import norm, truncnorm
from matplotlib import pyplot as plt

from classes import Market_Participant_binary, Market_Participant_continuous, Market

import random

random.seed(1)

def get_expected_income(expected_value, aleatory_uncertainty):
  return norm(loc=expected_value, scale=aleatory_uncertainty).expect(lambda x: x, lb=0, ub=1.)


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




def run_single_market(true_value, number_of_participants, number_of_iterations, config):

    market = Market(true_value, number_of_participants, number_of_iterations, config['mode'], config['exchange_fee'], config['pright'])


    p = market.run()

    if len(p) > 0:
        return np.mean(p), market.uncertainties
    else:
        return 0.5, market.uncertainties


def run_experiment(true_value, particpants_and_iterations, config):
    if 'inary' in config['mode']:
        true_value = round(true_value)
        output, uncertainties = run_single_market(true_value, particpants_and_iterations, particpants_and_iterations, config)
        res = abs(round(output) - true_value) < 0.1
        return res, uncertainties
    else:
        output, uncertainties = run_single_market(true_value, particpants_and_iterations, particpants_and_iterations, config)
        res = abs(output - true_value)
    return res, uncertainties

def run_experiments(n, participants_and_iterations, config):
    results = []
    uncertainties = []

    for i in range(n):
        true_value = random.random()
        output = run_experiment(true_value, participants_and_iterations, config)
        results.append(output[0])
        uncertainties.append(output[1])
    return results, uncertainties


def run_experiment_series(list_of_participant_numbers, config, n=10, true_value=None):
    results = []
    uncertainties = []
    if 'inary' in config['mode']:
        for p in list_of_participant_numbers:
            output = run_experiments(n, p, config)
            results.append(output[0].count(True) / n)
            uncertainties.append(output[1])
    else:
        for p in list_of_participant_numbers:
            output = run_experiments(n, p, config)
            results.append(np.median(output[0]))
            uncertainties.append(output[1])
    return list_of_participant_numbers, results, uncertainties
    