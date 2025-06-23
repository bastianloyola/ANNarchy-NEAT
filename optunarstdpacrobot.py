from ANNarchy import *
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from ns_gym.wrappers import NSClassicControlWrapper
from ns_gym.schedulers import ContinuousScheduler, PeriodicScheduler
from ns_gym.update_functions import RandomWalk, IncrementUpdate
from ns_gym import base
import ns_gym.utils as utils
from typing import Union, Any, Optional,Type
import random as rd
import optuna

IZHIKEVICH = Neuron(
    parameters="""
        a = 0.02 : population
        b = 0.2 : population
        c = -65.0 : population
        d = 8.0 : population
        I = 0.0
        tau_I = 10.0 : population
    """,
    equations="""
        dv/dt = 0.04*v*v + 5*v + 140 - u + I + g_exc - g_inh : init=-65
        du/dt = a*(b*v - u) : init=-14.0
        tau_I * dg_exc/dt = -g_exc
        tau_I * dg_inh/dt = -g_inh
    """,
    spike="v >= 30.0",
    reset="v = c; u += d"
)



class R_STDP(Synapse):
    """
    R-STDP con trazas pre y post, y modulación por recompensa.
    La actualización del peso depende de la coincidencia temporal (STDP) y el refuerzo externo.
    """

    _instantiated = []

    def __init__(self, tau_c=20.0, a=0.1,
                 A_plus=0.01, A_minus=0.01,
                 tau_plus=20.0, tau_minus=20.0,
                 w_min=0.0, w_max=1.0):

        parameters = """
            tau_c = %(tau_c)s : projection
            a = %(a)s : projection
            A_plus = %(A_plus)s : projection
            A_minus = %(A_minus)s : projection
            tau_plus = %(tau_plus)s : projection
            tau_minus = %(tau_minus)s : projection
            w_min = %(w_min)s : projection
            w_max = %(w_max)s : projection
            reward = 0.0 : projection
        """ % locals()

        equations = """
            tau_c * dc/dt = -c : event-driven
            tau_plus  * dx/dt = -x : event-driven
            tau_minus * dy/dt = -y : event-driven
        """

        pre_spike = """
            g_target += w
            x += A_plus
            c += y
            w += ite(
                (c < 0.0) and (reward < 0.0),
                clip(a * abs(c) * reward, -abs(w)*a, abs(w)*a),     
                abs(clip(a * c * reward, -abs(w)*a, abs(w)*a)))
        """

        post_spike = """
            y -= A_minus
            c += x
            w += ite(
                (c < 0.0) and (reward < 0.0),
                clip(a * abs(c) * reward, -abs(w)*a, abs(w)*a),     
                abs(clip(a * c * reward, -abs(w)*a, abs(w)*a)))
        """

        Synapse.__init__(self,
                         parameters=parameters,
                         equations=equations,
                         pre_spike=pre_spike,
                         post_spike=post_spike,
                         name="R-STDP")

        self._instantiated.append(True)



class BoundedRandomWalk(base.UpdateFn):
    def __init__(self, scheduler: Type[base.Scheduler], mu: float = 0, sigma: float = 1,
                 min_val: Optional[float] = None, max_val: Optional[float] = None, seed=None):
        super().__init__(scheduler)
        self.mu = mu
        self.sigma = sigma
        self.min_val = min_val
        self.max_val = max_val
        self.rng = np.random.default_rng(seed=seed)

    def __call__(self, param: float, t: float) -> tuple[float, bool]:
        return super().__call__(param, t)

    def update(self, param: float, t: float) -> float:
        #get random value between min val and max val
        updated_param = rd.uniform(self.min_val,self.max_val)
        while updated_param == param:
            updated_param = rd.uniform(self.min_val,self.max_val)
        return updated_param



def objective(trial):


    base_env = gym.make("Acrobot-v1")

    scheduler = PeriodicScheduler(period=5)
    scheduler2 = PeriodicScheduler(period=5)
    scheduler3 = PeriodicScheduler(period=5)
    update1 = BoundedRandomWalk(scheduler, min_val=0.8, max_val=2.0)
    update2 = BoundedRandomWalk(scheduler2, min_val=0.8, max_val=3.0)
    update3 = BoundedRandomWalk(scheduler3, min_val=0.2, max_val=0.8)



    tunable_params = {
        "LINK_LENGTH_1": update1,
        "LINK_LENGTH_2": update1,
        "LINK_MASS_1": update2,
        "LINK_MASS_2": update2,
        "LINK_COM_POS_1": update3,
        "LINK_COM_POS_2": update3,
    }

    env = NSClassicControlWrapper(base_env, tunable_params, change_notification=True)

    tau_c = trial.suggest_float('tau_c', 10, 30)
    a_plus = trial.suggest_float('a_plus', 0.001, 0.09)
    a_minus = trial.suggest_float('a_minus', 0.001, 0.09)
    tau_plus = trial.suggest_float('tau_plus', 10, 30)
    tau_minus = trial.suggest_float('tau_minus', 10, 30)


    input_index = list(range(12))
    output_index = [12, 13, 14]
    matrix = np.zeros((15, 15))
    for i in range(12):
        for j in range(3):
            np.random.seed(i+j)
            matrix[i][j+12] = np.random.uniform(0, 110)
    from scipy import sparse
    matrix = sparse.csr_matrix(matrix)

    limites = [
        (-1, 1), (-1, 1), (-1, 1), (-1, 1),
        (-12.5663706, 12.5663706), (-28.2743339, 28.2743339)
    ]

    def normalize(val, min_val, max_val):
        return (val - min_val) / (max_val - min_val)

    pop = Population(15, IZHIKEVICH)
    syn = Projection(pop, pop, target='exc', synapse=R_STDP(tau_c, a_plus, a_minus, tau_plus, tau_minus))
    syn.connect_from_sparse(matrix)

    compile()

    episodes = 200
    total_return = 0.0

    for ep in range(episodes):
        observation, _ = env.reset()
        terminated = False
        truncated = False
        episode_return = 0.0
        returns = []
        M = Monitor(pop, ['spike'])
        while not terminated and not truncated:
            i = k = 0
            for val in observation.state:
                val = normalize(val, *limites[k])
                pop[input_index[i]].I = val * 30 if val < 0 else 0
                pop[input_index[i+1]].I = 0 if val < 0 else val * 30
                i += 2
                k += 1
            theta1 = np.arccos(observation.state[0])
            theta2 = np.arccos(observation.state[2])
            reward = 1 - (-np.cos(theta1) - np.cos(theta2 + theta1))
            returns.append(reward)
            r = reward - np.mean(returns)
            syn.reward = r
            simulate(50.0)
            spikes = M.get('spike')
            output1 = np.size(spikes[output_index[0]])
            output2 = np.size(spikes[output_index[1]])
            output3 = np.size(spikes[output_index[2]])
            action = np.argmax([output1, output2, output3])
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += reward.reward
            M.reset()
            pop.reset()
            syn.reward = 0.0
        total_return += episode_return
        simulate(50.0)
        M.reset()
        pop.reset()
    clear()
    env.close()
    return total_return / episodes

# Estudio Optuna
study_name = "acrobot-rstdp"
storage_name = "sqlite:///{}.db".format(study_name)

study = optuna.create_study(
    study_name=study_name,
    storage=storage_name,
    direction='maximize',
    sampler=optuna.samplers.TPESampler(),
    pruner=optuna.pruners.HyperbandPruner(),
    load_if_exists=True
)
study.optimize(objective, n_trials=5)

print("Mejores parámetros encontrados:", study.best_params)
print("Mejor valor:", study.best_value)
