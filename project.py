import numpy as np

# """
# initization:
# Parameter that initialize the MDP.
# at the end of this part, we will have the basic component of a MDP

# Agent:
#     @state: [will be placed into the main loop, so it wont't be here]
#             t, At, LAMBDA
#             t: time
#             At: total number of assets on hand
#             LAMBDA: demand level
            
#     @action:
#             # FIXME:
#             action was influenced by MAX_NUMBER_BUY and BUDGET in hand
#                 MAX_NUMBER_BUY: total number that can buy
#                 BUDGET: initial budget on hand
#                 hitorical actions and budget used
#                 ???????????????????????????????????
#                 ???????????????????????????????????
                
#     @Reward:
#             Immediate rewards = -at * X_t + gamma(LAMBDA) * At * X_t
#             # COMMENT: possible to use constant or depends on state on the second X_t
            
#     @transition_probability:
#             probability to trainsiton from state to state
#             P[(t + dt, A(t+dt), LAMBDA'| (t, A(t), LAMBDA)), at]
            
#             # FIXME:
#                 possible rules
#                 P(S'|S, a)
#                     1. [LAMBDA = 0 (high)] = [P00, P01, P02]
#                         P00(t, At, at) = P00(0) - f1(At+at) - f2(t)
#                         P01(t, At, at) = P01(0) + g11(At+at) + g12(At+at)
#                         P02(t, At, at) = P02(0) + g21(At+at) + g22(At+at)
                        
#                         P00(0) + P01(0) + P02(0) = 1
#                         P00(t, At, at) + P01(t, At, at) + P02(t, At, at) = 1
#                     2. [LAMBDA = 1 (same)] = [P10, P11, P12]
#                         P10(t, At, at) = P10(0)
#                         P11(t, At, at) = P11(0) - h1(At+at) - h2(At+at)
#                         P12(t, At, at) = P12(0) + h1(At+at) + h2(At+at)
                        
#                         P10(0) + P11(0) + P12(0) = 1
#                         P10(t, At, at) + P11(t, At, at) + P12(t, At, at) = 1
#                     3. [LAMBDA = 2 (low)] = [P20, P21, P22]
#                         = [0., 0., 1.]
# Enviroment:
#     @Xt: per unit of asset's price at time t, can only compute in expectation sense
#         possible solution:
#         since only can compute in expectation sense, compute expected per assets price at each time point and
#         formualate an array
# """


#
N = None # total number of years
DAYS_PER_YEAR = 252
T = N * DAYS_PER_YEAR # total number of years
DECISION_EPOCHES = np.linspace(0, N, T) # notice that no dicision can be made at termnial


# ENVIORMENT
# TODO
X0 = None # initial price
MU_DIFFUSION = None #
SIGMA_DIFFUSION = None
INTENSITY_JUMP = None
MU_JUMP = None
SIGMA_JUMP = None
K = np.exp(MU_JUMP + 0.5 * SIGMA_JUMP ** 2) - 1
EXPECT_PRICE = X0 * np.exp((MU_DIFFUSION - 0.5 * SIGMA_DIFFUSION ** 2 - INTENSITY_JUMP * K + \
                            INTENSITY_JUMP * MU_JUMP) * DECISION_EPOCHES)


# AGENT

# transition 
# TODO, FIXME
LAMBDA_SAME_A = None
LAMBDA_SAME_t = None
LAMBDA_HIGH_A = None
LAMBDA_HIGH_t = None
LAMBDA_SAME = [LAMBDA_SAME_A, LAMBDA_SAME_t]
LAMBDA_HIGH = [LAMBDA_HIGH_A, LAMBDA_HIGH_t]

def transition(state, action):
    """
    transition probability from s -> s'
    @state = [t, At, LAMBDA]
    @action = number to buy
    """
    # TODO: [to_high, to_same, to_low]
    if state[2] == 0:
        prob = np.zeros(shape = (1, 3))
    elif state[2] == 1:
        prob = np.zeros(shape = (1, 3))
    else:
        prob = np.array([0., 0., 1.])
    return prob  

# action
# FIXME

# REWARD
# TODO: gamma_high, gamma_same, gamma_low
GAMMA = [None, None, None]
  

def immediate_reward(state, action, Xt):
    """
    reward
    """
    t, At, LAMBDA = state
    return -COST_PER_UNIT * action * Xt + GAMMA[LAMBDA] * At * Xt

# TODO: terminal reward
def penalty(At):
    pass
        


# action constraint
MAX_NUMBER_BUY = 20
BUDGET = None
COST_PER_UNIT = None
# FIXME: find possible actions
def possible_actions(budget, At, Xt):
    ceiling = min(MAX_NUMBER_BUY - At, np.floor(Xt / BUDGET))
    return np.arange(0, ceiling, 1)


    


def ValueFunction(state, action_list, Xt, value_next_epoch):
    """
    U{t+1}(S') = R(S(t), a(t)) + P([t+dt, At', high]|[t, At, LAMBDA], a(t))
                               + P([t+dt, At', same]|[t, At, LAMBDA], a(t))
                               + P([t+dt, At', low]|[t, At, LAMBDA], a(t))
    
    @state = t, At, LAMBDA
    @action_list = possible actions
    @Xt = cuurent price
    @value_next_epoch: 3 by 1 vector 
    """
    t, At, LAMBDA = state
    if t == T:
        return immediate_reward(state, 0, Xt) + penalty(At), 0
    
    values = np.array([],dtype=float)
    for action in action_list:
        reward = immediate_reward(state, action)
        prob = transition(state, action)
        reward += prob @ value_next_epoch
        values = np.append(values, reward)
    
    return np.max(values), action_list[np.argmax(values)]

    

        
for i, t in enumerate(DECISION_EPOCHES[::-1]):
    At_list = None
    for At in At_list:
        for LAMBDA in [0, 1, 2]:
            state = t, At, LAMBDA
            action_list = possible_actions(budget, At，EXPECT_PRICE[-i - 1])
        

    





    
    
    