import unittest
from numpy import array
from participants import Agent

class TestAgent(unittest.TestCase):

    def testCalcNewPortfolio(self):
        agent = Agent(initial_portfolio=array([3, 7]))
        new_portfolio = agent.calculate_new_portfolio_after_trade(1, 1, 1)
        self.assertEqual(new_portfolio[0], 3)
        self.assertEqual(new_portfolio[1], 8)

        agent = agent = Agent(initial_portfolio=array([-1, 2]))
        new_portfolio = agent.calculate_new_portfolio_after_trade(1, 1, 0)
        self.assertEqual(new_portfolio[0], -1)
        self.assertEqual(new_portfolio[1], 2)

        agent = agent = Agent(initial_portfolio=array([0, 0]))
        new_portfolio = agent.calculate_new_portfolio_after_trade(1, 1, 2)
        self.assertEqual(new_portfolio[0], 0)
        self.assertEqual(new_portfolio[1], 2)

        agent = agent = Agent(initial_portfolio=array([43, 193]))
        new_portfolio = agent.calculate_new_portfolio_after_trade(1, -1, 2)
        self.assertEqual(new_portfolio[0], 43)
        self.assertEqual(new_portfolio[1], 191)

        agent = agent = Agent(initial_portfolio=array([43, 193]))
        new_portfolio = agent.calculate_new_portfolio_after_trade(0, 1, 1)
        self.assertEqual(new_portfolio[0], 44)
        self.assertEqual(new_portfolio[1], 193)

        agent = agent = Agent(initial_portfolio=array([43, 193]))
        new_portfolio = agent.calculate_new_portfolio_after_trade(1, -1, 10)
        self.assertEqual(new_portfolio[0], 43)
        self.assertEqual(new_portfolio[1], 183)

        agent = agent = Agent(initial_portfolio=array([43, 193]))
        new_portfolio = agent.calculate_new_portfolio_after_trade(0, -1, 10)
        self.assertEqual(new_portfolio[0], 33)
        self.assertEqual(new_portfolio[1], 193)


        agent = agent = Agent(initial_portfolio=array([43, 193]))
        new_portfolio = agent.calculate_new_portfolio_after_trade(array([1., 1.]))
        self.assertEqual(new_portfolio[0], 44)
        self.assertEqual(new_portfolio[1], 194)

        agent = agent = Agent(initial_portfolio=array([43, 193]))
        new_portfolio = agent.calculate_new_portfolio_after_trade(array([-1., 1.]))
        self.assertEqual(new_portfolio[0], 42)
        self.assertEqual(new_portfolio[1], 194)

        agent = agent = Agent(initial_portfolio=array([43, 193]))
        new_portfolio = agent.calculate_new_portfolio_after_trade(array([1., -1.]))
        self.assertEqual(new_portfolio[0], 44)
        self.assertEqual(new_portfolio[1], 192)



if __name__ == '__main__':
    unittest.main()