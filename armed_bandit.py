# import sys
# sys.path.append("C:/Users/louis/Documents/ISF/Reinforcement_learning")

import armed_bandit_strategies
import numpy as np


#____________ class : ArmedBanditRunner________________

# __init__ :
# machines : function ({1,...,n_machines}) -> float
# n_machines : int ; machine max id (np.inf in case of infinite system)
# n_trials : int ; number of tests & updates / experience (default 1000)
# n_experiences int ; number of time the algorith will be launched (default 1)
# initial_weights float[n_machines] ; abstract weights for the machine choice (default zeros(n_machines))
# strategy str in ['greedy', 'eps_greedy', 'UB', 'gradient']; general way to update values & weights
# strategy_param None or float. epsilon value for eps-greedy, c for UB (upper bond coefficient), alpha value for gradient (intensity of update)
#test bool: default = False (debug/test param. If True unnecessary params will be stored)

# ______________attributes________________
#PUBLIC
#machines, n_machines, n_trials, n_experiences, strategy_param as defined before
#strategy a function according to the strategy in  __init__ strategy
#returns : float[n_experience]; to store the returns/experience




class ArmedBanditRunner() :

######### Init & Input check #############


    def _check_init_machines(self, machines):
        assert callable(machines), "machines is not callable"
        test = machines(0)
        assert type(test) in [float, int], "machines do not return a real number"
        self.machines = machines




    def _check_init_n_trials(self, n_trials):
        assert type(n_trials) == int, "n_trials is not an integer"
        self.n_trials = n_trials



    def _check_init_n_experiences(self, n_experiences):
        assert type(n_experiences) == int, "n_experiences is not an integer"
        self.n_experiences = n_experiences



    def _check_n_machines(self, n_machines):
        assert type(n_machines) == int, 'The machine number is not an integer'
        assert n_machines > 0, "The machine number is <= 0"



    def _check_init_initial_weights(self, initial_weights, n_machines):
        assert type(initial_weights) == list, "initial_weights is not a list"
        assert len(initial_weights) == n_machines, "ERROR : more machines than weights"
        assert type(initial_weights[0]) in [int, float], "It seems that the elements of weights are not numbers"
        self.initial_weights = initial_weights



    def _check_init_strategy(self, strategy):
        if strategy == 'greedy' :
            self.strategy = armed_bandit_strategies.greedy
        elif strategy == 'eps_greedy' :
            self.strategy = armed_bandit_strategies.eps_greedy
        elif strategy == 'UB' :
            self.strategy = armed_bandit_strategies.UB_update
        elif strategy == 'gradient' :
            self.strategy = armed_bandit_strategies.gradient_update
        else :
            print("Strategy not available")
            raise(ValueError)




    def _check_init_strategy_param(self, strategy_param, strategy) :
        assert (strategy == 'greedy') or ( strategy == 'UB' and strategy_param > 0) or (strategy in ['gradient', 'eps_greedy'] and 0 <= strategy_param and strategy_param <= 1), 'Wrong parameter for the chosen strategy'
        self.strategy_param = strategy_param




    def __init__(self, machines, n_machines, n_trials, n_experiences, initial_weights, strategy, strategy_param, test = False):
        self._check_init_machines(machines)
        self._check_init_n_trials(n_trials)
        self._check_init_n_experiences(n_experiences)
        self._check_n_machines(n_machines)
        self._check_init_initial_weights(initial_weights, n_machines)
        self._check_init_strategy(strategy)
        self._check_init_strategy_param(strategy_param, strategy)

        self.returns = [0] * n_experiences
        self.test = test



###########CORE##########

    def _run_exp(self, n_experience, returns, test) :

        #initialization
        weights = self.initial_weights
        choices = [0] * len(weights)
        machines = self.machines
        returns = self.returns
        strategy_param = self.strategy_param
        n_trials = self.n_trials
        if test :
            values_t = []
            chosen_machines = []
        n_trial = 1

        #FOR
        if test :

            while n_trial <= n_trials:
                chosen_one, value = self.strategy(machines, weights, returns, choices, n_trial, n_experience, strategy_param, test)
                chosen_machines.append(chosen_one)
                values_t.append(value)
                if n_trial == 1 and self.strategy.__name__ == "UB_update" :
                    n_trial += len(weights)
                else :
                    n_trial += 1

            OUTPUT ={'weights' : weights, 'values' : values_t, 'choices' : chosen_machines, 'return' : returns[n_experience - 1]}
            return OUTPUT


        else :
            for n_trial in range(1, n_trials + 1):
                self.strategy(machines, weights, returns, choices, n_trial, n_experience, strategy_param, test)



    def run(self):

        #initialization
        returns = self.returns
        n_experiences = self.n_experiences
        test = self.test

        if test :

            weights = []
            values = []
            chosen_machines = []

            for n_experience in range (1, n_experiences + 1) :
                result = self._run_exp(n_experience, returns, test)
                weights.append(result['weights'])
                values.append(result['values'])
                chosen_machines.append(result['choices'])

            OUTPUT ={'weights' : weights, 'values' : values, 'choices' : chosen_machines, 'return' : returns}
            return OUTPUT

        else :

            for n_experience in range (n_experiences) :
                result = self._run_exp(n_experience, returns, test)

            return returns




############TESTS##############


def _machines(x) : return x


def __main__():


    (">>>>> Single experience >>>>>>>>>>")
    first_test = ArmedBanditRunner(_machines, 10, 10, 1, [0]*10, 'gradient', 0.1, True)
    print(first_test.run())
    print("<<<<<<<Single experience<<<<<<<<<<")

    print(">>>>> 2 experiences >>>>>>>>>>")
    first_test = ArmedBanditRunner(_machines, 10, 20, 2, [0]*10, 'UB', 0.1, True)
    print(first_test.run())
    print("<<<<<<<2 experiences<<<<<<<<<<")


if __name__ == "__main__" : __main__()