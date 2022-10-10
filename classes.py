import random
import numpy as np
from scipy.stats import norm, truncnorm

random.seed(1)

def lmsr_cost_function(b, x, y):
  """
  b: liquidity parameter
  x: amount owed if no
  y: amount owed if yes
  """
  return b * np.log(np.exp(x/b) + np.exp(y/b))


def truncated_distribution(mean, std, lower_bound, upper_bound):
  clip_a, mean, std, clip_b = lower_bound, mean, std, upper_bound

  a, b = (clip_a - mean) / std, (clip_b - mean) / std

  dist = truncnorm(loc=mean, scale=std, a=a, b=b)
  
  return dist

def normal_distribution(mean, std, lower_bound, upper_bound):
  dist = norm(loc=mean, scale=std)
  
  return dist

def get_participant(true_value, mode, credence_distribution=truncated_distribution, pright=.5):

  """
  pright comes into action only for participants in binary markets.
  pright is the probability of the belief being true
  """
  if mode == "binary":
      if not pright:
        belief = belief = np.random.randint(2)
      else:
        gets_it_right = np.random.uniform(0.0, 1.0) < pright
        if gets_it_right:
          belief = true_value
        else:
          belief = 1 - true_value
      uncertainty = np.random.uniform(0.1, 0.5)


      # Assumption that people who are correct are more confident in their belief
      if abs(belief - true_value) < 0.1:
        uncertainty = np.random.uniform(0.01, 0.3)

      participant = Market_Participant_binary(belief, uncertainty)
      return participant

  elif mode == "index":
      # Mode is index
      expected_result = np.random.uniform(0., 1.)



      epistemic_uc_about_outcome = abs(true_value - expected_result)

      participant = Market_Participant_continuous(expected_result, epistemic_uc_about_outcome, credence_distribution)

      return participant

  else:
    raise ValueError("mode must be binary or index")





class Market():
    def __init__(self, x, n_t, n_r, mode, exchange_fee=0., pright=0.5):
        self.x = x
        self.n_t = int(n_t)
        self.n_r = int(n_r)
        self.mode = mode
        self.exchange_fee = exchange_fee
        self.traders = []
        self.prices = []
        self.uncertainties = []
        self.pright = pright
        
        for i in range(self.n_t):
            self.traders.append(get_participant(self.x, self.mode, pright=self.pright))
        
        self.uncertainty = 1
        self.uncertainties.append(self.uncertainty)
    
    def run(self):
        for i in range(self.n_r):
            seller, buyer = random.sample(self.traders, 2)

            if seller.expected_income_of_share > buyer.expected_income_of_share:
                seller, buyer = buyer, seller
            
            proposed_price = (seller.expected_income_of_share + buyer.expected_income_of_share) / 2

            seller_price = proposed_price - self.exchange_fee
            buyer_price = proposed_price + self.exchange_fee
            if seller.decide_on_sell(seller_price, self.uncertainty) and buyer.decide_on_buy(buyer_price, self.uncertainty):
                self.prices.append(proposed_price)
                self.uncertainty = np.std(self.prices) + (1 / (len(self.prices) + 1))
                self.uncertainties.append(self.uncertainty)
        
        return self.prices
  

class Market_Participant_binary():
  """
  participant for the case where the question is binary (e.g. will Scholz be reelected)

  The participant holds a belief of either 1 (yes) or 0 (no)

  He has an uncertainty in [0, 0.5) which describes the probability he assigns to
  the possibility that his belief is wrong
  """
  def __init__(self, belief, uncertainty):
    # the participant believes either 1 or 0
    self.belief = belief
    # Number between 0 and 0.5
    self.uncertainty = uncertainty

    self.expected_income_of_share = 1 - self.uncertainty

    if not self.belief:
      self.expected_income_of_share = self.uncertainty
  
  def decide_on_buy(self, offered_price, market_prices=None, config=None):

    expected_value = -offered_price + self.expected_income_of_share

    return expected_value > 0.
  
  def decide_on_sell(self, price, market_prices=None, config=None):
    
    expected_value = price - self.expected_income_of_share

    return expected_value > 0.



class Market_Participant_continuous():
  """
  participant for the case where the question is continous
  (e.g. what is the vote share of SPD going to be)

  The participant's credence is represented by a Gaussian distribution, whose
  mean is the participant's credence and whose variance is the participant's
  uncertainty

  """
  def __init__(self, expected_result, uncertainty, credence_distribution):

    self.expected_result = expected_result

    self.uncertainty = uncertainty

    self.credence = credence_distribution(self.expected_result, self.uncertainty, 0., 1.,)

    self.expected_income_of_share = self.credence.expect(lambda x: x, lb=0., ub=1.)


  def decide_on_buy(self, offered_price, market_uncertainty=None):

    expected_value = self.expected_income_of_share - offered_price


    
    if not market_uncertainty:
      return expected_value > 0.
    else:
      # A trader who takes market certaity into account will be more likely to trade if his own uncertainty is lower than that of the market

      market_assessment = market_uncertainty > self.uncertainty

      return expected_value > 0. and market_assessment
  
  def decide_on_sell(self, price, market_uncertainty=None):
    
    expected_value = - self.expected_income_of_share + price

    if not market_uncertainty:
      return expected_value > 0.
    else:
      # A trader who takes market certaity into account will be more likely to trade if his own uncertainty is lower than that of the market
      market_assessment = market_uncertainty > self.uncertainty

      return expected_value > 0. and market_assessment


class Market_Maker_binary():
  """
  Market maker using a logarithmic market scoring rule
  """

  def __init__(self, liquidity, payout_vector=np.array([0., 0.]), cost_function=lmsr_cost_function, initial_cash=0.):
    """
    liquidity: the liquidity parameter for the scoring function
    payout_vector: The first component records the money the market maker needs to pay out if the outcome is "no", the second component if it is "yes"
    cost_function: the cost function to use
    initial_cash: the amount of money the market maker starts with
    """
    self.liquidity = liquidity
    self.payout_vector = payout_vector
    self.cost_function = cost_function
    self.current_cost = self.cost_function(self.liquidity, self.payout_vector[0], self.payout_vector[1])
    self.current_cash = initial_cash

  def work_with_alternative_inputs(self, direction, buysell):
    if direction == 'yes':
      direction = 1
    if direction == 'no':
      direction = 0
    if buysell == 'buy':
      buysell = 1
    if buysell == 'sell':
      buysell = -1
    return direction, buysell

  def quote_price(self, direction, buysell, amount):
    """
    direction: 0 for no, 1 for yes
    buysell: 1 for buy, -1 for sell
    amount: amount of shares to be bought or sold by the trader
    """
    direction, buysell = self.work_with_alternative_inputs(direction, buysell)

    if buysell not in [1, -1]:
      raise ValueError("buysell must be 1 or -1")

    new_cost = self.cost_function(self.liquidity, self.payout_vector[0] + direction * buysell * amount, self.payout_vector[1] + (1 - direction) * buysell * amount)
    costdiff = new_cost - self.current_cost
    return costdiff

    return costdiff

  def make_trade(self, direction, buysell, amount):
    """
    direction: 0 for no, 1 for yes
    buysell: 1 for buy, -1 for sell
    amount: amount of shares to be bought or sold by the trader
    """
    direction, buysell = self.work_with_alternative_inputs(direction, buysell)
    if buysell not in [1, -1]:
      raise ValueError("buysell must be 1 or -1")


    print(f"Making trade: The trader is {'buying' if buysell == 1 else 'selling'} a bet {'on' if direction else 'against'} the proposition, which would pay out ${amount}")

    print(f"Current cost: {self.current_cost}; current cash: {self.current_cash}; current payout: {self.payout_vector}")
    costdiff = self.quote_price(direction, buysell, amount)

    print(f"Cost difference: {costdiff}")

    self.current_cost = costdiff + self.current_cost
    self.current_cash = self.current_cash + costdiff

    self.payout_vector = self.payout_vector + np.array([direction * buysell * amount, (1 - direction) * buysell * amount])
    print(f"New cost: {self.current_cost}; new cash {self.current_cash}; new payout: {self.payout_vector}")


  def quote_vector_trade_price(self, trade_vector):
    """
    trade_vector: The change in assets the trade would cause in trader's portfolio
    e.g. a trade vector of [0.3, -1] makes a trade which give the trader assets that would
    pay out 0.3 if the proposition is false and -1 if it is true
    i.e. the trader is buying 0.3 bets on the proposition and selling 1 bet on the proposition
    """

    new_vector = self.payout_vector + trade_vector
    new_cost = self.cost_function(self.liquidity, new_vector[0], new_vector[1])
    costdiff = new_cost - self.current_cost
    return costdiff

  def make_vector_trade(self, trade_vector):
    """
    trade_vector: The change in assets the trade would cause in trader's portfolio
    e.g. a trade vector of [0.3, -1] makes a trade which give the trader assets that would
    pay out 0.3 if the proposition is false and -1 if it is true
    i.e. the trader is buying 0.3 bets on the proposition and selling 1 bet on the proposition
    """
    print(f"Making trade: Trader is buying {trade_vector}")

    print(f"Current cost: {self.current_cost}; current cash: {self.current_cash}; current payout: {self.payout_vector}")
    costdiff = self.quote_vector_trade_price(trade_vector)

    print(f"Cost difference: {costdiff}")

    self.current_cost = costdiff + self.current_cost
    self.current_cash = self.current_cash + costdiff

    self.payout_vector = self.payout_vector + trade_vector
    print(f"New cost: {self.current_cost}; new cash {self.current_cash}; new payout: {self.payout_vector}")

