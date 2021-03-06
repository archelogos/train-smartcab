import random
import operator
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

N_TRIALS = 100
N_SIMULATIONS = 1
simulation_rates = []
last_errors = []


def get_simulation_params(i):
    """Alpha, Gamma, Epsilon"""
    params = [[0.9, 0.1, 0.1],[0.0, 0.0, 0.0]]
    return params[i]

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.trial = 0 # [0,99]
        self.unlocked = False
        self.trial_end = False
        # TODO: Initialize any additional variables here
        # Valid actions
        self.actions = [None, 'forward', 'left', 'right']

        # Q-Learning
        # Alpha (learning rate)
        self.alpha = get_simulation_params(0)[0] # should decay with t too?
        # Gamma (discount factor)
        self.gamma = get_simulation_params(0)[1] #
        self.epsilon = get_simulation_params(0)[2] # equal chance 0.5, progressive decay with t value
        self.decay_factor = 1.0
        self.Q = {}
        self.Q_default_value = 0.0 #not learning yet

        # Report
        self.total_reward = []
        self.total_penalties = []
        self.trial_reward = 0
        self.trial_penalty = 0
        self.success = 0
        self.failure = 0
        self.last_failure = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

        if self.unlocked:
            self.trial += 1
            self.decay_factor = ((N_TRIALS+1)-self.trial)/float(N_TRIALS)

            #print "{0:.2f}".format(self.decay_factor)

            self.alpha = self.alpha #* self.decay_factor
            self.gamma = self.gamma
            self.epsilon = self.epsilon * self.decay_factor

            self.trial_end = False
            self.trial_reward = 0;
            self.trial_penalty = 0;
        else:
            # Locked for first trial
            self.unlocked = True

    def end(self):
        self.total_reward.append(self.trial_reward)
        self.total_penalties.append(self.trial_penalty)

        if N_SIMULATIONS == 1: #Report each TRIAL info if N_SIMULATIONS == 1
            print "Trial Finished: Total Reward {}".format(self.trial_reward)

            if self.trial == N_TRIALS - 1: #When simulation finishes
                success_rate = (float(self.success)/float(N_TRIALS))
                failure_rate = (float(self.failure)/float(N_TRIALS))

                simulation_rates.append(success_rate)
                last_errors.append(self.last_failure)

                print "------------------------------------------------------------------"  # [debug]
                print "CONCLUSION"
                print "Success Rate {}".format(success_rate)
                print "Failure Rate {}".format(failure_rate)
                print "Last Failure {}".format(self.last_failure)

                y = np.array(self.total_reward)
                z = np.array(self.total_penalties)
                mean = y.mean()
                std = y.std()
                print "Reward distribution Mean {}".format(mean)
                print "Reward distribution Std Deviation {}".format(std)

                plt.figure(1)

                plt.subplot(221)
                plt.plot(y)
                plt.title('Reward/Iteration')
                plt.xlabel('Trial')
                plt.ylabel('Reward Value')
                plt.text(0, 0, r'Success Rate %s'%(success_rate))

                plt.subplot(222)
                plt.hist(y)
                plt.title('Reward Distribution')
                plt.xlabel('Reward Value')
                plt.ylabel('# trials')
                plt.text(0, 0, r'$\mu=%s,\ \sigma=%s$'%(mean,std))

                plt.subplot(223)
                plt.plot(z)
                plt.title('Penalties/trial')
                plt.xlabel('# trial')
                plt.ylabel('Penalties')

                plt.show()

        else: #N_SIMULATIONS > 1
            if self.trial == N_TRIALS - 1: #end of each simulation
                success_rate = (float(self.success)/float(N_TRIALS))
                failure_rate = (float(self.failure)/float(N_TRIALS))

                simulation_rates.append(success_rate)
                last_errors.append(self.last_failure)


        print "------------------------------------------------------------------"  # [debug]


    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        # STEP_TWO: Status definition, tuple of useful inputs
        # Is deadline input necessary?
        self.state = (
            ('next', self.next_waypoint),
            ('light', inputs['light']))
            #('oncoming', inputs['oncoming']),
            #('left', inputs['left']),
            #('right', inputs['right']),
            #('deadline', deadline))

        state = self.state
        #print "LearningAgent.update(): state = {}".format(state)  # [debug]

        # TODO: Select action according to your policy
        # STEP_ONE: Random action
        # Exploring -> random
        # Best choice -> agent experience Q matrix
        action = random.choice(self.actions)
        if random.random() > self.epsilon:
            best_action = self.best_Q_action(state)
            #print "Best Action = {}".format(best_action)
            if best_action:
                action = best_action

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.trial_reward += reward

        # TODO: Learn policy based on state, action, reward
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        state_prime = (
            ('next', self.next_waypoint),
            ('light', inputs['light']))
            #('oncoming', inputs['oncoming']),
            #('left', inputs['left']),
            #('right', inputs['right']),
            #('deadline', deadline))

        self.update_Q(state, action, reward, state_prime)

        # Reporting
        if deadline == 0: # end of the trial
            if reward > 2.0:
                self.success += 1
            else:
                self.failure += 1
                self.last_failure = self.trial

            self.trial_end = True
        else:
            if reward > 2.0: # reaches the destination
                self.success += 1
                self.trial_end = True

            if reward < 0.0: #tracking a penalty
                self.trial_penalty += 1

        if self.trial_end:
            self.end()

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        #print "------------------------------------------------------------------"  # [debug]


    def best_Q_action(self, state):
        """Select max Q action based on given state.

        Args:
            state(tuple)

        Returns:
            string: action [None, "forward", "left", "right"], False in other case
        """
        state_Q = {}

        for action in self.actions:
            if (state, action) not in self.Q:
                return False
            else:
                state_Q[(state, action)] = self.Q[(state, action)]

        return max(state_Q.iteritems(), key=operator.itemgetter(1))[0][1]


    def max_Q_by_state(self, state):
        """Best Q given a state

        Args:
            state(tuple)

        Returns:
            int: Q value if that state is stored, 0.0 in other case
        """
        max_q = []

        for action in self.actions:
            if (state, action) not in self.Q:
                return self.Q_default_value
            else:
                max_q.append(self.Q[(state, action)])

        return max(max_q)


    def update_Q(self, state, action, reward, state_prime):
        """Update the Q Matrix

        Args:
            state(tuple)
            action(string)
            reward(int)
            state_prime(tuple)

        """
        # print "Updating Q MATRIX"

        # Q(s,a) = (1- alpha)*Q(s,a) + alpha*(reward + gamma * max_Q(s', a'))

        # Init value if it doesn't exist: Q(self.state, self.action) = 0
        if (state, action) not in self.Q:
                self.Q[(state, action)] = self.Q_default_value

        self.Q[(state, action)] = (1 - self.alpha) * self.Q[(state, action)] + \
            self.alpha * (reward + self.gamma * self.max_Q_by_state(state_prime))


def run():
    """Run the agent for a finite number of trials."""

    for simulation in range(0, N_SIMULATIONS):

        # Set up environment and agent
        e = Environment()  # create environment (also adds some dummy traffic)
        a = e.create_agent(LearningAgent)  # create agent
        e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
        # NOTE: You can set enforce_deadline=False while debugging to allow longer trials
        # TODO: Change later enforce_deadline=True

        # Now simulate it
        sim = Simulator(e, update_delay=0.001, display=False)  # create simulator (uses pygame when display=True, if available)
        # NOTE: To speed up simulation, reduce update_delay and/or set display=False

        sim.run(n_trials=N_TRIALS)  # run for a specified number of trials
        # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

        if simulation == N_SIMULATIONS - 1:

            with open('results.csv', 'a') as csvfile:
                fieldnames = ['alpha', 'gamma', 'epsilon', 'success_rate', 'last_failure']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                for index in range(0,len(simulation_rates)):
                    writer.writerow({
                        'alpha': get_simulation_params(0)[0],
                        'gamma': get_simulation_params(0)[1],
                        'epsilon': get_simulation_params(0)[2],
                        'success_rate': simulation_rates[index],
                        'last_failure': last_errors[index]})


            if N_SIMULATIONS > 1: #multiple simulation AND last simulation

                plt.figure(1)

                plt.subplot(211)
                plt.plot(simulation_rates)
                plt.title('Success Rate/Simulation')
                plt.xlabel('# Simulation')
                plt.ylabel('Success Rate')

                plt.subplot(212)
                plt.plot(last_errors)
                plt.title('Last failed trial per simulation')
                plt.xlabel('# Simulation')
                plt.ylabel('Last failed trial')

                plt.show()


if __name__ == '__main__':
    run()
