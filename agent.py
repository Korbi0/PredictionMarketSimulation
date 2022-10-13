from multipledispatch import dispatch
from numpy import array

class Error(Exception):
    pass
class BankrupcyError(Error):
    """Raised when an agent wants to make
         a trade that costs more than he can spend"""

class Agent():
    def __init__(self,
            initial_cash=0.,
            initial_portfolio=array([0., 0.]),
            minimum_possible_cash=-float('inf')):
        """
        initial_cash: Amount of Cash the agent starts with
        initial_portfolio: First component records the money the agent receives
        if the outcome is 'no', second component records the money the agent
        receives if the outcome is 'yes'
        """
        self.cash = initial_cash
        self.minimum_possible_cash = minimum_possible_cash
        self.portfolio = initial_portfolio
    
    @dispatch(int, int, float, float)
    def make_trade(self, direction, buysell, number_of_shares, price_per_share):
        """
        direction: 0 for no, 1 for yes
        buysell: 1 for buy, -1 for sell
        number_of_shares: number of shares to be bought or sold
        price_per_share: price per share
        """
        direction, buysell = self.work_with_alternative_inputs(direction,
                                                                buysell)

        if buysell not in [-1, 1]:
            raise ValueError("buysell must be 1 or -1")
        
        money = buysell * (number_of_shares * price_per_share)

        if self.cash - money < self.minimum_possible_cash:
            raise BankrupcyError()

        self.cash = self.cash - money
        self.portfolio = self.calculate_new_portfolio_after_trade(direction,
                        buysell, number_of_shares)

    @dispatch(object, float)
    def make_trade(self, trade_vector, money):
        """
        trade_vector: The change in assets the trade would cause in self's portfolio
        e.g. a trade vector of [0.3, -1] makes a trade which give self assets that would
        pay out 0.3 if the proposition is false and -1 if it is true
        i.e. self is buying 0.3 bets on the proposition and selling 1 bet on the proposition

        money: The amount of money the trade costs
        """
        if self.cash - money < self.minimum_possible_cash:          
            raise BankrupcyError()
        
        self.cash = self.cash - money

        self.portfolio = self.calculate_new_portfolio_after_trade(trade_vector)

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
    
    @dispatch(object, object, float)
    def calculate_new_portfolio_after_trade(self,
                                            direction,
                                            buysell,
                                            number_of_shares):
        """
        direction: 0 for no, 1 for yes
        buysell: 1 if self wants to buy, -1 if self wants to sell
        number_of_shares: number of shares to be bought or sold by self
        """

        direction, buysell = self.work_with_alternative_inputs(direction,
                                                                buysell)
        assert(direction in [1, 0])
        assert(buysell in [1, -1])
        assert(number_of_shares >= 0)
        new_portfolio = self.portfolio.copy()
        new_portfolio[direction] += buysell * number_of_shares
        return new_portfolio

    @dispatch(object)
    def calculate_new_portfolio_after_trade(self, trade_vector):
        """
        trade_vector: The change in assets the trade would cause in self's portfolio
        e.g. a trade vector of [0.3, -1] makes a trade which give self assets that would
        pay out 0.3 if the proposition is false and -1 if it is true
        i.e. self is buying 0.3 bets on the proposition and selling 1 bet on the proposition
        """
        assert(len(trade_vector) == len(self.portfolio))
        new_portfolio = self.portfolio.copy()
        new_portfolio = new_portfolio + trade_vector
        return new_portfolio
