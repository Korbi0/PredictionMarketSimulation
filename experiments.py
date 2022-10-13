import numpy as np
from random import random, seed
from markets import DoubleAuctionMarket, Market_Generic, MarketMakerMarket
from participants import get_participant
seed(1)

class ExperimentSeries():
    def __init__(self,
                    mode,
                    list_of_participant_numbers,
                    market_class=DoubleAuctionMarket,
                    pright=0.5,
                    experiments_per_number=10,
                    take_market_uncertainty_into_account=False):
        """
        mode: either 'binary' or 'index'
        list_of_participant_numbers: list of numbers of participants to run the experiment for
        pright: probability that a participant is correct in binary markets
        experiments_per_number: number of experiments to run for each number of participants
        """
        self.results = []
        self.uncertainties = []
        self.experiments = []
        self.mode = mode
        self.list_of_participant_numbers = list_of_participant_numbers
        self.pright = pright
        self.experiments_per_number = experiments_per_number
        self.take_market_uncertainty_into_account = take_market_uncertainty_into_account
        self.market_class = market_class

    def run_series(self):
        for p in self.list_of_participant_numbers:
            results, uncertainties = self.run_experiments(p)
            if self.mode == 'binary':
                self.results.append(results.count(True) / len(results))
            elif self.mode == 'index':
                self.results.append(np.mean(results))
            self.uncertainties.append(uncertainties)
        
        return self.list_of_participant_numbers, self.results, self.uncertainties

    def run_experiments(self, participants_and_iterations):
        results = []
        uncertainties = []

        for i in range(self.experiments_per_number):
            true_value = random()
            experiment = Experiment(true_value=true_value,
                                        number_of_participants=participants_and_iterations,
                                        number_of_trading_rounds=participants_and_iterations,
                                        mode=self.mode,
                                        pright=self.pright,
                                        take_market_uncertainty_into_account=self.take_market_uncertainty_into_account,
                                        market_class=self.market_class)
            output = experiment.run_experiment()
            self.experiments.append(experiment)
            results.append(output)
            uncertainties.append(experiment.uncertainties)
        return results, uncertainties

class Experiment():

    def __init__(self, true_value: float,
                    number_of_participants: int,
                    number_of_trading_rounds: int,
                    mode: str,
                    market_class=DoubleAuctionMarket,
                    pright=0.5,
                    exchange_fee=0.0,
                    take_market_uncertainty_into_account=False):
        """
        true_value: The true value of the propostion to be assessed.
        If mode is binary, the true value is binarized

        number_of_participants: The number of traders participating in the market

        number_of_trading_rounds: How many trading rounds are to take place in the market during the experiment

        mode: Either 'binary' or 'index'

        market_class: What kind of market is it? (Should be a subclass of the Market_Generic class,
        such as DoubleAuctionMarket

        pright: The probability for an individual trader to 'get it right'. Only relevant in binary markets

        exchange_fee: Fee to be paid by each trader for each trade

        take_market_uncertainty_into_account: When deciding whether to make a trade, does a trader compare
        their own uncertainty to the market's uncertainty?
        """
        self.true_value = true_value
        if mode == 'binary':
            self.true_value = round(self.true_value)
        self.number_of_participants = number_of_participants
        self.number_of_trading_rounds = number_of_trading_rounds
        self.mode = mode
        self.pright = pright
        self.traders = [get_participant(true_value=self.true_value, mode=self.mode, pright=self.pright, take_market_uncertainty_into_account=take_market_uncertainty_into_account) for b in range(number_of_participants)]
        self.market = market_class(traders=self.traders, mode=self.mode, exchange_fee=exchange_fee)
        self.uncertainties = []
    
    def run_experiment(self):
        """
        Runs a single experiment and returns whether the true value was found
        (in the case of binary markets)
        or the distance of the value which the market converged to from the true value
        (in the case of index markets)
        """
        self.yes_prices, self.no_prices = self.market.run_market(self.number_of_trading_rounds)

        self.uncertainties = None
        try:
            self.uncertainties = self.market.uncertainties
        except AttributeError:
            pass
        if len(self.yes_prices) > 0:
            res = np.mean(self.yes_prices)
        else:
            res = 0.5
        if self.mode == 'binary':
            return abs(round(res) - self.true_value) < 0.1
        elif self.mode == 'index':
            return abs(res - self.true_value)
        else:
            raise ValueError("mode must be binary or index")

def run_experiment_series(list_of_participant_numbers, config, n=10, true_value=None):
    series = ExperimentSeries(mode=config['mode'],
        market_class=config['market_class'],
        list_of_participant_numbers=list_of_participant_numbers,
        experiments_per_number=n, pright=config['pright'],
        take_market_uncertainty_into_account=config['take_market_uncertainty_into_account'])
    return series.run_series()
