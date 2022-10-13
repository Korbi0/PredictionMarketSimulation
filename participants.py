from multipledispatch import dispatch
import random
from agent import Agent

from sympy.abc import b, x, y
random.seed(1)

from mathtools import *

def get_participant(true_value,
                    mode,
                    take_market_uncertainty_into_account=False,
                    credence_distribution=truncated_distribution,
                    pright=.5):

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


      # Assumption that people who are correct are more confident
      if abs(belief - true_value) < 0.1:
        uncertainty = np.random.uniform(0.01, 0.3)

      participant = Market_Participant_binary(belief, uncertainty)
      return participant

  elif mode == "index":
      # Mode is index
      expected_result = np.random.uniform(0., 1.)

      epistemic_uc_about_outcome = abs(true_value - expected_result)

      participant = Market_Participant_continuous(expected_result,
        epistemic_uc_about_outcome,
        credence_distribution,
        take_market_uncertainty_into_account=take_market_uncertainty_into_account)

      return participant

  else:
    raise ValueError("mode must be binary or index")

class Market_Participant(Agent):
    def __init__(self,
                    expected_result,
                    uncertainty,
                    take_market_uncertainty_into_account,
                    credence_distribution=None):
        super().__init__()
        if credence_distribution:
            self.credence_distribution = credence_distribution
        self.take_market_uncertainty_into_account = take_market_uncertainty_into_account
        self.expected_result = expected_result
        self.uncertainty = uncertainty
        self.get_credence()
        self.get_expected_income_of_shares()
        self.validate()
    

    def get_credence(self):
        """
        Should set self.credence to the credence of the participant in the proposition
        """
        raise NotImplementedError("get_credence must be implemented")

    def get_expected_income_of_shares(self):
        """
        Should set self.expected_income_yes_share
        and self.expected_income_no_share to the expected income of a bet on 
        'yes' and of a bet on 'no' respectively
        """
        raise NotImplementedError("get_expected_income_of_shares must be implemented")

    def validate(self):
        """
        Validate that all necessary attributes have been set
        """
        try:
            assert(self.credence)
            assert(self.expected_income_yes_share)
            assert(self.expected_income_no_share)
        except:
            print('error')


    def decide_on_trade(self, offered_price, direction=1, buysell=1, market_uncertainty=None):
        """
        offered_price: price at which trade would take place
        buysell: 1 means 'buy', -1 means 'sell
        direction: 1 means 'yes', 0 means 'no'
        """
        direction, buysell = self.work_with_alternative_inputs(direction, buysell)

        assert(buysell in [1, -1])
        assert(direction in [1, 0])
        expected_income_of_share = (direction * self.expected_income_yes_share
                                    + (1 - direction) * self.expected_income_no_share)
        expected_value_of_trade = buysell * (expected_income_of_share - offered_price)

        if not self.take_market_uncertainty_into_account or not market_uncertainty:
            return expected_value_of_trade > 0.
        else:
            # A trader who takes market certaity into account will be
            # more likely to trade if his own uncertainty
            # is lower than that of the market
            market_assessment = market_uncertainty > self.uncertainty

            return expected_value_of_trade > 0. and market_assessment



class Market_Participant_binary(Market_Participant):
    """
    participant for the case where the question is binary
    (e.g. will Scholz be reelected)

    The participant holds a belief of either 1 (yes) or 0 (no)

    He has an uncertainty in [0, 0.5) which describes the probability he assigns to
    the possibility that his belief is wrong
    """
    def __init__(self, belief, uncertainty):
        super().__init__(belief, uncertainty, take_market_uncertainty_into_account=False)
        if not self.expected_result:
            (self.expected_income_yes_share, self.expected_income_no_share) = (
                self.expected_income_no_share, self.expected_income_yes_share
                )

    def get_credence(self):
        self.credence = 1 - self.uncertainty
    
    def get_expected_income_of_shares(self):
        self.expected_income_yes_share = self.credence
        self.expected_income_no_share = 1 - self.credence



class Market_Participant_continuous(Market_Participant):
    """
    participant for the case where the question is continous
    (e.g. what is the vote share of SPD going to be)

    The participant's credence is represented by a Gaussian distribution, whose
    mean is the participant's credence and whose variance is the participant's
    uncertainty

    """

    def __init__(self,
                    expected_result,
                    uncertainty,
                    credence_distribution,
                    take_market_uncertainty_into_account=False):
        super().__init__(expected_result,
                        uncertainty,
                        take_market_uncertainty_into_account,
                        credence_distribution=credence_distribution)
        self.take_market_uncertainty_into_account = take_market_uncertainty_into_account

    def get_credence(self):
       self.credence = self.credence_distribution(self.expected_result,
                                                    self.uncertainty, 0., 1.,)


    def get_expected_income_of_shares(self):
        lb = self.credence.ppf(0.001)
        ub = self.credence.ppf(0.999)
        self.expected_income_yes_share = self.credence.expect(lambda x: x, lb=lb, ub=ub)
        self.expected_income_no_share = self.credence.expect(lambda x: (1 - x), lb=lb, ub=ub)


class Market_Maker_binary(Agent):

  def __init__(self,
                liquidity=10,
                initial_portfolio=np.array([0., 0.]),
                cost=lmsr_cost,
                initial_cash=0.):
    """
    liquidity: the liquidity parameter for the scoring function
    payout_vector: The first component records the money the market maker
    needs to pay out if the outcome is "no", the second component if it is "yes"
    cost: the cost function to use as a sympy symbolic expression
    initial_cash: the amount of money the market maker starts with
    """
    super().__init__(initial_portfolio=initial_portfolio, initial_cash=initial_cash)
    self.liquidity = liquidity
    self.cost = cost
    self.dcostdx = self.cost.diff(x)
    self.dcostdy = self.cost.diff(y)
    self.dcostdx_func = sp.lambdify([b, x, y], self.dcostdx)
    self.dcostdy_func = sp.lambdify([b, x, y], self.dcostdy)
    self.cost_function = sp.lambdify([b, x, y], self.cost)
    self.current_cost = self.cost_function(self.liquidity,
                                            self.portfolio[0],
                                            self.portfolio[1])



  def get_credences(self):
    """
    Returns the credences of the market maker
    the first component represents the credence in 'no',
    the second component the credence in 'yes'
    """
    b = self.liquidity
    x = self.portfolio[0]
    y = self.portfolio[1]
    return np.array([self.dcostdx_func(b, x, y), self.dcostdy_func(b, x, y)])


  def quote_price(self, *args):
    """
    direction: 0 for no, 1 for yes
    buysell: 1 if self would buy, -1 if self would sell
    number_of_shares: number of shares to be bought or sold by self

    alternatively:
    pass a numpy array which describes, as a vector, the trade in terms of
    changes to the portfolio
    """
    
    new_portfolio = self.calculate_new_portfolio_after_trade(*args)
    new_cost = self.cost_function(self.liquidity, new_portfolio[0], new_portfolio[1])
    costdiff = new_cost - self.current_cost
    return costdiff

  @dispatch(int, int, float, float)
  def make_trade(self, direction, buysell, number_of_shares, price_per_share):
    """
    direction: 0 for no, 1 for yes
    buysell: 1 for buy, -1 for sell
    amount: amount of shares to be bought or sold by the trader
    """

    print(f"Making trade: The trader is {'buying' if buysell == 1 else 'selling'}\
     a bet {'on' if direction else 'against'} the proposition, which would pay\
     out ${number_of_shares}")

    print(f"Current cost: {self.current_cost}; current cash: \
        {self.cash}; current portfolio: {self.portfolio}")

    costdiff = self.quote_price(direction, buysell, number_of_shares)
    print(f"Cost difference: {costdiff}")

    super().make_trade(direction, buysell, number_of_shares, price_per_share)
    self.current_cost = self.cost_function(self.liquidity,
                                            - self.portfolio[0],
                                            - self.portfolio[1])


    print(f"New cost: {self.current_cost}; new cash {self.cash}; \
        new portfolio: {self.portfolio}")

  @dispatch(object, float)
  def make_trade(self, trade_vector, money):
    """
    trade_vector: The change in assets the trade would cause in self's portfolio
    e.g. a trade vector of [0.3, -1] makes a trade which give self assets that would
    pay out 0.3 if the proposition is false and -1 if it is true
    i.e. self is buying 0.3 bets on the proposition and selling 1 bet on the proposition

    money: The amount of money the trade costs
    """

    print(f"Making vector trade: {trade_vector} for ${money}")

    print(f"Current cost: {self.current_cost}; current cash: \
        {self.cash}; current portfolio: {self.portfolio}")

    costdiff = self.quote_price(object)
    print(f"Cost difference: {costdiff}")

    super().make_trade(object, money)
    self.current_cost = self.cost_function(self.liquidity,
                                            - self.portfolio[0],
                                            - self.portfolio[1])


    print(f"New cost: {self.current_cost}; new cash {self.cash}; \
        new portfolio: {self.portfolio}")
