import random
import numpy as np
from math import isnan
from participants import Market_Maker_binary
random.seed(1)



class Market_Generic():
  def __init__(self, traders, mode='binary', exchange_fee=0.0):
    self.traders = traders # List of traders
    self.mode = mode
    self.exchange_fee = exchange_fee
    assert(self.mode in ['binary', 'index'])
    self.yes_prices = [] # List of prices at which bets on 'yes' have been traded
    self.no_prices = [] # List of prices at which bets on 'no' have been traded
    

  def run_market(self, trading_rounds):
    """
    Runs the market for a given number of trading rounds and reports
    the market's credence after theses rounds"""
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
    return self.get_credence()

  def trading_round(self):
    """
    Should implement an event at which a trade may take place
    If a trade takes place, should return a dictionary of the form
    {'direction': 'no' or 'yes', 'price': price, 'buyer': trader1, 'seller': trader2}
    if no trade takes place, should return None
    """
    raise NotImplementedError("This method must be implemented in the child class")
  
  def get_credence(self):
    """
    Returns the market's credence
    """
    raise NotImplementedError("This method must be implemented in the child class")

class DoubleAuctionMarket(Market_Generic):
  """
  A market where a trade only takes place if there are two traders
  which can agree to a price
  """
  def __init__(self, traders, mode, exchange_fee=0.):
      super().__init__(traders, mode, exchange_fee)
      assert(len(self.traders) > 0)
      self.yes_prices = []
      self.uncertainties = []

      
      self.uncertainty = 1
      self.uncertainties.append(self.uncertainty)
    
  def trading_round(self):
      seller, buyer = random.sample(self.traders, 2)

      if seller.expected_income_yes_share > buyer.expected_income_yes_share:
          seller, buyer = buyer, seller
      
      proposed_price = ((seller.expected_income_yes_share
                        + buyer.expected_income_yes_share)
                        / 2)
      seller_price = proposed_price - self.exchange_fee
      buyer_price = proposed_price + self.exchange_fee
      if (seller.decide_on_trade(seller_price, buysell=-1, direction=1, 
            market_uncertainty=self.uncertainty)
          and
          buyer.decide_on_trade(buyer_price,
              buysell=1, direction=1, market_uncertainty=self.uncertainty)):
          seller.make_trade(1, -1, 1., seller_price)
          buyer.make_trade(1, 1, 1., buyer_price)
          uncertainty = np.std(self.yes_prices) + (1 / (len(self.yes_prices) + 1))
          if not isnan(uncertainty):
            self.uncertainty = uncertainty
            self.uncertainties.append(self.uncertainty)
          return {'direction': 'yes',
                  'price': proposed_price,
                  'buyer': buyer,
                  'seller': seller}
      else:
        return None
      
  def get_credence(self):
      if len(self.yes_prices) == 0 and len(self.no_prices) == 0:
          return 0.5
      else:
          return np.mean(self.yes_prices + [1 - x for x in self.no_prices])
        
    
class MarketMakerMarket(Market_Generic):
  """
  A market where all trades are made with a single market maker
  who is always willing to trade at the price he quotes
  """
  def __init__(self, traders, mode='binary', exchange_fee=0, liquidity=10):
    super().__init__(traders, mode, exchange_fee)
    assert(len(self.traders) > 0)
    self.yes_prices = []
    self.market_maker = Market_Maker_binary(liquidity=liquidity)

  def trading_round(self):
    # get the trader most wanting to trade
    trader, direction = self.find_trader_most_willing_to_trade()

    # If the trader buys, the market maker sells, hence the -1
    offer = self.market_maker.quote_price(direction, -1, 1.)

    accept = trader.decide_on_trade(offered_price=offer,
                                    direction=direction, buysell=1)

    if accept:
      self.market_maker.make_trade(direction, -1, 1., offer)
      trader.make_trade(direction, 1, 1., offer)

  def find_trader_most_willing_to_trade(self):
    """
    Returns the trader most willing to trade, and the direction
    where direction 0 means 'no' and direction 1 means 'yes'
    """
    price_to_buy_yes = self.market_maker.quote_price('yes', 'buy', 1.)
    price_to_buy_no = self.market_maker.quote_price('no', 'buy', 1.)
    def expected_profit(price, expected_income):
      return expected_income - price
    trader = max(self.traders,
      key=lambda x: max(
        expected_profit(price_to_buy_yes, x.expected_income_yes_share),
        expected_profit(price_to_buy_no, 1-x.expected_income_no_share))
        )
    if trader.expected_income_yes_share > 0.5:
      direction = 1
    else:
      direction = 0
    return trader, direction
  
  def get_credence(self):
     return self.market_maker.get_credences()[0]