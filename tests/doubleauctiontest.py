import unittest
from experiments import run_experiment_series
from markets import DoubleAuctionMarket

class TestDoubleAuction(unittest.TestCase):

    def test_binary(self):
        config = {
            # The price a buyer and seller each have to pay the market
            # for facilitating their trade:
            'exchange_fee': 0.,
            # The kind of market. Is it a winner take all market answering a
            # binary question ('binary') or a regression market extimateing an index
            # ('index')
            'mode': 'binary',
            'market_class': DoubleAuctionMarket,
            'take_market_uncertainty_into_account': True,
            'pright': 0.5 # Probability that a participant in a binary market gets it right
        }

        x, y, _ = run_experiment_series([1000], config)
        self.assertGreater(y[-1], 0.9)
        
    def test_index(self):
        config = {
            # The price a buyer and seller each have to pay the market
            # for facilitating their trade:
            'exchange_fee': 0.,
            # The kind of market. Is it a winner take all market answering a
            # binary question ('binary') or a regression market extimateing an index
            # ('index')
            'mode': 'index',
            'market_class': DoubleAuctionMarket,
            'take_market_uncertainty_into_account': True,
            'pright': 0.4 # Probability that a participant in a binary market gets it right
        }

        x, y, _ = run_experiment_series([1000], config)
        self.assertLess(y[-1], 0.1)



if __name__ == '__main__':
    unittest.main()