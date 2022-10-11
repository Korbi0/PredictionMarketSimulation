import random
import numpy as np
from math import isnan

random.seed(1)



class Market_Generic():
  def __init__(self, traders):
    self.traders = traders # List of traders
    self.yes_prices = [] # List of prices at which bets on 'yes' have been traded
    self.no_prices = [] # List of prices at which bets on 'no' have been traded
    

  def run_market(self, trading_rounds):
    for i in range(trading_rounds):
      trade = self.trading_round()
      if trade is None:
        continue
      elif trade['direction'] == 'yes':
        self.yes_prices.append(trade['price'])
      elif trade['direction'] == 'no':
        self.no_prices.append(trade['price'])
      else:
        raise ValueError("direction must be yes or no")
    return self.yes_prices, self.no_prices

  def trading_round(self):
    """
    Should implement an event at which a trade may take place
    If a trade takes place, should return a dictionary of the form
    {'direction': 'no' or 'yes', 'price': price, 'buyer': trader1, 'seller': trader2}
    if no trade takes place, should return None
    """
    raise NotImplementedError("This method must be implemented in the child class")
    pass

class DoubleAuctionMarket(Market_Generic):
  def __init__(self, x, traders, mode, exchange_fee=0., pright=0.5):
      super().__init__(traders)
      self.x = x
      self.mode = mode
      self.exchange_fee = exchange_fee
      self.traders = traders
      assert(len(self.traders) > 0)
      self.yes_prices = []
      self.uncertainties = []
      self.pright = pright

      
      self.uncertainty = 1
      self.uncertainties.append(self.uncertainty)
    
  def trading_round(self):
      seller, buyer = random.sample(self.traders, 2)

      if seller.expected_income_of_share > buyer.expected_income_of_share:
          seller, buyer = buyer, seller
      
      proposed_price = (seller.expected_income_of_share + buyer.expected_income_of_share) / 2

      seller_price = proposed_price - self.exchange_fee
      buyer_price = proposed_price + self.exchange_fee
      if seller.decide_on_sell(seller_price, self.uncertainty) and buyer.decide_on_buy(buyer_price, self.uncertainty):
          uncertainty = np.std(self.yes_prices) + (1 / (len(self.yes_prices) + 1))
          if not isnan(uncertainty):
            self.uncertainty = uncertainty
            self.uncertainties.append(self.uncertainty)
          return {'direction': 'yes', 'price': proposed_price, 'buyer': buyer, 'seller': seller}
      else:
        return None
        
    

