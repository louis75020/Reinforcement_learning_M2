import numpy as np
import random
import scipy.stats

############ CORE ################


def _play_this_machine(chosen_one, machines, choices, returns, n_experience):
    #we play with the chosen machine
    random_value = machines(chosen_one)
    #we update the previous choices
    choices[chosen_one] += 1
    #we update returns
    returns[n_experience - 1] += random_value
    #OUTPUT
    return random_value



def _play_best_machine(machines, weights, returns, choices, n_experience):
    #best machines id
    where_max = np.argwhere(weights == np.max(weights)).flatten()
    #we choose randomly among the best machines
    chosen_one = where_max[ random.randint(0, len(where_max) - 1) ]
    #GAME
    random_value = _play_this_machine(chosen_one, machines, choices, returns, n_experience)
    #OUTPUT
    return chosen_one, random_value



############### Available strategies ###############

def gradient_update(machines, weights, returns, choices, n_trial, n_experience, strategy_param, test = False):
    chosen_one, random_value = _play_best_machine(machines, weights, returns, choices, n_experience)
    #Boltzmann distribution
    exp_weights = list( map( lambda x : np.exp(x), weights ))
    sum_exp_weights = sum(exp_weights)
    standard_exp_weights = list( map( lambda x : x/sum_exp_weights, exp_weights ))
    pi_t = standard_exp_weights.copy()

    #SGA
    pi_t[:] = list( map( lambda x : -x, pi_t ))[:]
    pi_t[chosen_one] = 1 - standard_exp_weights[chosen_one]
    weights[:] = list( map( lambda x, y : x + strategy_param * (random_value - returns[n_experience - 1] /n_trial) * y, weights, pi_t ) )[:]
    if test: return chosen_one, random_value
    return 0





#UB = upper bound
def UB_update(machines, weights, returns, choices, n_trial, n_experience, strategy_param, test = False):

    #At inception, we play one time every machine
    if n_trial == 1 :
        for i in range(len(weights)) :
            random_value = _play_this_machine(i, machines, choices, returns, n_experience)
        return np.inf, np.inf

    #GAME
    chosen_one, random_value = _play_best_machine(machines, weights, returns, choices, n_experience)
    #we update weights
    weights[:] = list( map(lambda x, y : x + strategy_param * np.sqrt(np.log(n_trial)/y), weights, choices) )[:]

    #OUTPUT
    if test: return chosen_one, random_value
    return 0






def eps_greedy(machines, weights, returns, choices, n_trial, n_experience, strategy_param, test = False):
    #Let's choose between greed and exploration
    #0=greed, 1=exploration
    tmp_bool = float(scipy.stats.bernoulli.rvs(strategy_param, size = 1))
    print(tmp_bool)
    #greedy case
    if tmp_bool == 0. :
        return greedy(machines, weights, returns, choices, n_trial, n_experience, strategy_param, test)
    #exploratory case
    tmp_weights = weights.copy()
    #We don't wanna choose the best one
    where_max = np.argmax(weights)
    tmp_weights.remove(where_max)
    chosen_one = tmp_weights[ random.randint(0, len(tmp_weights) - 1) ]
    #GAME
    random_value = _play_this_machine(chosen_one, machines, choices, returns, n_experience)
    if test: return chosen_one, random_value
    return 0





def greedy(machines, weights, returns, choices, n_trial, n_experience, strategy_param = None, test = False):
    #GAME
    chosen_one, random_value = _play_best_machine(machines, weights, returns, choices, n_experience)
    #we update weights
    weights[chosen_one] += 1/choices[chosen_one] * (random_value - weights[chosen_one])
    #OUTPUT
    if test: return chosen_one, random_value
    return 0





















#############TESTS############

def _strategy_to_str(weights, returns, choices, n_trial, n_experience) :
    print("# experience :")
    print(n_experience)
    print("# trial :")
    print(n_trial)
    print("# choices :")
    print(choices)
    print("returns :")
    print(returns)
    print("weights :")
    print(weights)

def _machines(x) : return x

def __main__():



    print(">>>>>>>>>GREEDY>>>>>>>>>>>>>")
    n_experience = 1
    n_trial = 0
    weights = [0] * 3
    returns = [0] * n_experience
    choices = [0] * 3
    print("____INITITAL STATE____")
    _strategy_to_str(weights, returns, choices, n_trial, n_experience)
    print("______1st trial_______")
    n_trial += 1
    chosen_one, value = greedy(_machines, weights, returns, choices, n_trial, n_experience, test = True)
    print(" Chosen one, value :")
    print(chosen_one, value)
    _strategy_to_str(weights, returns, choices, n_trial, n_experience)
    print("_________2nd trial__________")
    n_trial += 1
    chosen_one, random_value = greedy(_machines, weights, returns, choices, n_trial, n_experience, test = True)
    print(" Chosen one, value :")
    print(chosen_one, value)
    _strategy_to_str(weights, returns, choices, n_trial, n_experience)
    print("<<<<<<<<<<<GREEDY<<<<<<<<")





    print(">>>>>>>>>epsilon - GREEDY>>>>>>>>>>>>>")
    n_experience = 1
    n_trial = 0
    weights = [0] * 3
    returns = [0] * n_experience
    choices = [0] * 3
    print("____INITITAL STATE____")
    _strategy_to_str(weights, returns, choices, n_trial, n_experience)
    print("______1st trial_______")
    n_trial += 1
    chosen_one, value = eps_greedy(_machines, weights, returns, choices, n_trial, n_experience, strategy_param = 0.1, test = True)
    print(" Chosen one, value :")
    print(chosen_one, value)
    _strategy_to_str(weights, returns, choices, n_trial, n_experience)
    print("_________2nd trial__________")
    n_trial += 1
    chosen_one, random_value = eps_greedy(_machines, weights, returns, choices, n_trial, n_experience, strategy_param = 0.1, test = True)
    print(" Chosen one, value :")
    print(chosen_one, value)
    _strategy_to_str(weights, returns, choices, n_trial, n_experience)
    print("<<<<<<<<<<<epsilon - GREEDY<<<<<<<<")


    print(">>>>>>>>>UB>>>>>>>>>>>>>")
    n_experience = 1
    n_trial = 0
    weights = [0] * 3
    returns = [0] * n_experience
    choices = [0] * 3
    print("____INITITAL STATE____")
    _strategy_to_str(weights, returns, choices, n_trial, n_experience)
    print("______1st trial_______")
    n_trial += 1
    chosen_one, value = UB_update(_machines, weights, returns, choices, n_trial, n_experience, strategy_param = 0.1, test = True)
    print(" Chosen one, value :")
    print(chosen_one, value)
    _strategy_to_str(weights, returns, choices, n_trial, n_experience)
    print("_________2nd trial__________")
    n_trial += 3
    chosen_one, value = UB_update(_machines, weights, returns, choices, n_trial, n_experience, strategy_param = 0.1, test = True)
    print(" Chosen one, value :")
    print(chosen_one, value)
    _strategy_to_str(weights, returns, choices, n_trial, n_experience)
    print("<<<<<<<<<<<UB<<<<<<<<")



    print(">>>>>>>>>SGA>>>>>>>>>>>>>")
    n_experience = 1
    n_trial = 0
    weights = [0] * 3
    returns = [0] * n_experience
    choices = [0] * 3
    print("____INITITAL STATE____")
    _strategy_to_str(weights, returns, choices, n_trial, n_experience)
    print("______1st trial_______")
    n_trial += 1
    chosen_one, value = gradient_update(_machines, weights, returns, choices, n_trial, n_experience, strategy_param = 0.1, test = True)
    print(" Chosen one, value :")
    print(chosen_one, value)
    _strategy_to_str(weights, returns, choices, n_trial, n_experience)
    print("_________2nd trial__________")
    n_trial += 3
    chosen_one, value = gradient_update(_machines, weights, returns, choices, n_trial, n_experience, strategy_param = 0.1, test = True)
    print(" Chosen one, value :")
    print(chosen_one, value)
    _strategy_to_str(weights, returns, choices, n_trial, n_experience)
    print("<<<<<<<<<<<SGA<<<<<<<<")


if __name__ == "__main__":
    __main__()