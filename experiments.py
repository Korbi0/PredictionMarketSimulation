import numpy as np
from random import random, seed
from markets import DoubleAuctionMarket
from participants import get_participant
seed(1)

class ExperimentSeries():
    def __init__(self, mode, list_of_participant_numbers, pright=0.5, experiments_per_number=10):
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

    def run(self):
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
            experiment = Experiment(true_value=true_value, number_of_participants=participants_and_iterations, number_of_trading_rounds=participants_and_iterations, mode=self.mode, pright=self.pright)
            output = experiment.run()
            self.experiments.append(experiment)
            results.append(output[0])
            uncertainties.append(output[1])
        return results, uncertainties

class Experiment():
    def __init__(self, true_value: float, number_of_participants: int, number_of_trading_rounds: int, mode: str, market_class=DoubleAuctionMarket, pright=0.5, exchange_fee=0.0):
        self.true_value = true_value
        if mode == 'binary':
            self.true_value = round(self.true_value)
        self.number_of_participants = number_of_participants
        self.number_of_trading_rounds = number_of_trading_rounds
        self.mode = mode
        self.pright = pright
        self.traders = [get_participant(true_value=self.true_value, mode=self.mode, pright=self.pright) for b in range(number_of_participants)]
        self.market = market_class(x=true_value, traders=self.traders, mode=self.mode, pright=self.pright, exchange_fee=exchange_fee)
        self.uncertainties = []
    
    def run(self):
        self.yes_prices, self.no_prices = self.market.run(self.number_of_trading_rounds)
        self.uncertainties = self.market.uncertainties
        if len(self.yes_prices) > 0:
            res = np.mean(self.yes_prices)
        else:
            res = 0.5
        if self.mode == 'binary':
            return abs(round(res) - self.true_value) < 0.1, self.uncertainties
        elif self.mode == 'index':
            return abs(res - self.true_value), self.uncertainties
        else:
            raise ValueError("mode must be binary or index")

def run_experiment_series(list_of_participant_numbers, config, n=10, true_value=None):
    series = ExperimentSeries(mode=config['mode'], list_of_participant_numbers=list_of_participant_numbers, experiments_per_number=n, pright=config['pright'])
    return series.run()
