import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import gym

from algorithms.actor_critic import ActorCritic
from algorithms.reinforce_with_baseline import ReinforceWithBaseline
from algorithms.episodic_semigradient_nstep_sarsa import EpisodicSemigradientNStepSarsa
from environment.gridworld import GridWorld687

from multiprocessing import Process

# Setting Random Seed for reproducability of results
SEED = 687
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


def learn_and_plot(envname, algoname, args, filename, gamma):

    print(f"{filename} Started")

    # Run the same algorithm 20 times and collect the learning plot values
    episodes_return_curves = []
    action_episodescompleted_curves = []

    for i in range(20):
        
        if envname == 'gridworld':
            env = GridWorld687()
        else:
            env = gym.make(envname)   
        env.reset(seed=SEED)

        if algoname == 'reinforce':
            algorithm = ReinforceWithBaseline(*args)
        elif algoname == 'ac':
            algorithm = ActorCritic(*args)
        elif algoname == 'sarsa':
            algorithm = EpisodicSemigradientNStepSarsa(*args)

        episodes_return_curve, action_episodescompleted_curve = algorithm.learn_policy(env=env, gamma=gamma)

        episodes_return_curves.append(episodes_return_curve)
        action_episodescompleted_curves.append(action_episodescompleted_curve)

        if "gridworld" not in filename:
            env.close()

        print()

    # Plot no of episode completed vs return from the initial state
    episodes_return_curve_mean = np.mean(np.array(episodes_return_curves), axis=0)
    episodes_return_curve_std = np.std(np.array(episodes_return_curves), axis=0)
    plt.figure()
    markers, caps, bars = plt.errorbar(np.array(range(len(episodes_return_curve_mean))), episodes_return_curve_mean, episodes_return_curve_std)
    [bar.set_alpha(0.2) for bar in bars]
    plt.xlabel("# of Episodes Completed")
    plt.ylabel("Return")
    plt.title("Learning curve")
    plt.grid(True)
    plt.savefig(f"{filename}_er")

    # Plot no of actions taken vs no of episodes completed
    min_len = min([len(x) for x in action_episodescompleted_curves])
    action_episodescompleted_curves = [x[0:min_len] for x in action_episodescompleted_curves]
    action_episodescompleted_curve_mean = np.mean(np.array(action_episodescompleted_curves), axis=0)
    action_episodescompleted_curve_std = np.std(np.array(action_episodescompleted_curves), axis=0)
    plt.figure()
    plt.plot(np.array(range(len(action_episodescompleted_curve_mean))), action_episodescompleted_curve_mean)
    plt.xlabel("# of Actions Taken")
    plt.ylabel("# of Episodes Completed")
    plt.title("Learning curve")
    plt.grid(True)
    plt.savefig(f"{filename}_ae")

    print(f"{filename} Completed")


if __name__ == "__main__": 

    # # Algorithm 1 - REINFORCE with Baseline

    # Environment - Experimental
    # processes = []
    # for alpha_policy in [1e-3]:
    #     for alpha_value in [1e-3]:
    #         for hidden_units in [10]:
    #             env = gym.make('LunarLander-v2')
    #             filename = f"Figures/reinforce_lunarlander_{str(alpha_policy).replace('.', '#')}_{str(alpha_value).replace('.', '#')}_{hidden_units}"
    #             p = Process(target=learn_and_plot, args=('LunarLander-v2', 'reinforce', [env.observation_space.shape[0], [hidden_units], env.action_space.n, alpha_policy, alpha_value], filename, 0.9))
    #             processes.append(p)
    #             p.start()
    # for p in processes:
    #     p.join()
    # print()

    # Environment - 687 Gridworld
    processes = []
    for alpha_policy in [1e-3, 1e-4, 1e-2]:
        for alpha_value in [1e-3, 1e-4, 1e-2]:
            for hidden_units in [10, 32, 64]:
                env = GridWorld687()
                filename = f"Figures/reinforce_gridworld_{str(alpha_policy).replace('.', '#')}_{str(alpha_value).replace('.', '#')}_{hidden_units}"
                p = Process(target=learn_and_plot, args=('gridworld', 'reinforce', [10, [hidden_units], 4, alpha_policy, alpha_value], filename, 0.9))
                processes.append(p)
                p.start()
    for p in processes:
        p.join()
    print()

    # Environment - Acrobot
    processes = []
    for alpha_policy in [1e-3, 1e-4, 1e-2]:
        for alpha_value in [1e-3, 1e-4, 1e-2]:
            for hidden_units in [10, 32, 64]:
                env = gym.make('Acrobot-v1')
                filename = f"Figures/reinforce_acrobot_{str(alpha_policy).replace('.', '#')}_{str(alpha_value).replace('.', '#')}_{hidden_units}"
                p = Process(target=learn_and_plot, args=('Acrobot-v1', 'reinforce', [env.observation_space.shape[0], [hidden_units], env.action_space.n, alpha_policy, alpha_value], filename, 1))
                processes.append(p)
                p.start()
    for p in processes:
        p.join()
    print()
    print()


    # # Algorithm 2 - Actor Critic

    # # Environment - Experimental
    # processes = []
    # for alpha_policy in [1e-3]:
    #     for alpha_value in [1e-3]:
    #         for hidden_units in [10]:
    #             env = gym.make('LunarLander-v2')
    #             filename = f"Figures/ac_lunarlander_{str(alpha_policy).replace('.', '#')}_{str(alpha_value).replace('.', '#')}_{hidden_units}"
    #             p = Process(target=learn_and_plot, args=('LunarLander-v2', 'ac', [env.observation_space.shape[0], [hidden_units], env.action_space.n, alpha_policy, alpha_value], filename, 0.9))
    #             processes.append(p)
    #             p.start()
    # for p in processes:
    #     p.join()
    # print()

    # Environment - 687 Gridworld
    processes = []
    for alpha_policy in [1e-3, 1e-4, 1e-2]:
        for alpha_value in [1e-3, 1e-4, 1e-2]:
            for hidden_units in [10, 32, 64]:
                for exploration in [0.6, 1]:
                    env = GridWorld687()
                    filename = f"Figures/ac_gridworld_{str(alpha_policy).replace('.', '#')}_{str(alpha_value).replace('.', '#')}_{hidden_units}_{str(exploration).replace('.', '#')}"
                    p = Process(target=learn_and_plot, args=('gridworld', 'ac', [10, [hidden_units], 4, alpha_policy, alpha_value, exploration], filename, 0.9))
                    processes.append(p)
                    p.start()
    for p in processes:
        p.join()
    print()

    # Environment - Acrobot
    processes = []
    for alpha_policy in [1e-3, 1e-4, 1e-2]:
        for alpha_value in [1e-3, 1e-4, 1e-2]:
            for hidden_units in [10, 32, 64]:
                env = gym.make('Acrobot-v1')
                filename = f"Figures/ac_acrobot_{str(alpha_policy).replace('.', '#')}_{str(alpha_value).replace('.', '#')}_{hidden_units}"
                p = Process(target=learn_and_plot, args=('Acrobot-v1', 'ac', [env.observation_space.shape[0], [hidden_units], env.action_space.n, alpha_policy, alpha_value], filename, 1))
                processes.append(p)
                p.start()
    for p in processes:
        p.join()
    print()
    print()


    # # Algorithm 3 - Episodic Semigraident N Step SARSA

    # # Environment - Lunar
    # processes = []
    # for N in [5]:
    #     for alpha_value in [1e-3]:
    #         for hidden_units in [10]:
    #             for epsilon in [0.1]:
    #                 env = gym.make('LunarLander-v2')
    #                 filename = f"Figures/sarsa_lunar_{str(alpha_value).replace('.', '#')}_{N}_{hidden_units}_{str(epsilon).replace('.', '#')}"
    #                 p = Process(target=learn_and_plot, args=('LunarLander-v2', 'sarsa', [env.observation_space.shape[0], [hidden_units], env.action_space.n, alpha_value, N, epsilon], filename, 0.9))
    #                 processes.append(p)
    #                 p.start()
    # for p in processes:
    #     p.join()
    # print()

    # processes = []
    # for N in [5]:
    #     for alpha_value in [1e-3]:
    #         for hidden_units in [10]:
    #             for beta in [0.9, 0.95]:
    #                 env = gym.make('LunarLander-v2')
    #                 filename = f"Figures/sarsa_lunar_{str(alpha_value).replace('.', '#')}_{N}_{hidden_units}_{str(epsilon).replace('.', '#')}_{str(beta).replace('.', '#')}_decay"
    #                 p = Process(target=learn_and_plot, args=('LunarLander-v2', 'sarsa', [env.observation_space.shape[0], [hidden_units], env.action_space.n, alpha_value, N, 1, True, beta], filename, 0.9))
    #                 processes.append(p)
    #                 p.start()
    # for p in processes:
    #     p.join()
    # print()

    # # Environment - 687 Gridworld
    processes = []
    for N in [5, 10]:
        for alpha_value in [1e-3, 1e-2, 1e-4]:
            for hidden_units in [10, 32, 64]:
                for epsilon in [0.1, 0.05]:
                    filename = f"Figures/sarsa_gridworld_{str(alpha_value).replace('.', '#')}_{N}_{hidden_units}_{str(epsilon).replace('.', '#')}"
                    p = Process(target=learn_and_plot, args=('gridworld', 'sarsa', [10, [hidden_units], 4, alpha_value, N, epsilon], filename, 0.9))
                    processes.append(p)
                    p.start()
    for p in processes:
        p.join()
    print()

    processes = []
    for N in [5, 10]:
        for alpha_value in [1e-3, 1e-2, 1e-4]:
            for hidden_units in [10, 32, 64]:
                for beta in [0.9, 0.95, 0.97]:
                    filename = f"Figures/sarsa_gridworld_{str(alpha_value).replace('.', '#')}_{N}_{hidden_units}_{str(1).replace('.', '#')}_{str(beta).replace('.', '#')}"
                    p = Process(target=learn_and_plot, args=('gridworld', 'sarsa', [10, [hidden_units], 4, alpha_value, N, 1, True, beta], filename, 0.9))
                    processes.append(p)
                    p.start()
    for p in processes:
        p.join()
    print()

    # Environment - Acrobot
    processes = []
    for N in [5, 10]:
        for alpha_value in [1e-3, 1e-2, 1e-4]:
            for hidden_units in [10, 32, 64]:
                for epsilon in [0.1, 0.05]:
                    env = gym.make('Acrobot-v1')
                    filename = f"Figures/sarsa_acrobot_{str(alpha_value).replace('.', '#')}_{N}_{hidden_units}_{str(epsilon).replace('.', '#')}"
                    p = Process(target=learn_and_plot, args=('Acrobot-v1', 'sarsa', [env.observation_space.shape[0], [hidden_units], env.action_space.n, alpha_value, N, epsilon], filename, 1))
                    processes.append(p)
                    p.start()
    for p in processes:
        p.join()
    print()
    print()

    processes = []
    for N in [5, 10]:
        for alpha_value in [1e-3, 1e-2, 1e-4]:
            for hidden_units in [10, 32, 64]:
                for beta in [0.9, 0.95, 0.97]:
                    env = gym.make('Acrobot-v1')
                    filename = f"Figures/sarsa_acrobot_{str(alpha_value).replace('.', '#')}_{N}_{hidden_units}_{str(1).replace('.', '#')}_{str(beta).replace('.', '#')}"
                    p = Process(target=learn_and_plot, args=('Acrobot-v1', 'sarsa', [env.observation_space.shape[0], [hidden_units], env.action_space.n, alpha_value, N, 1, True, beta], filename, 1))
                    processes.append(p)
                    p.start()
    for p in processes:
        p.join()
    print()
    print()


