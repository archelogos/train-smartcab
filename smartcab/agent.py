import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        # Valid actions
        self.actions = [None, 'forward', 'left', 'right']

        # Q-Learning
        # Alpha (learning rate)
        self.alpha = 0.5
        # Gamma (discount factor)
        self.gamma = 0.5
        self.epsilon = 0
        self.Q = {}
        self.Q_default_value = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

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
            # ('oncoming', inputs['oncoming']),
            # ('left', inputs['left']),
            # ('right', inputs['right']))
            # ('deadline', deadline))

        state = self.state
        print "LearningAgent.update(): state_zero = {}".format(state)  # [debug]

        # TODO: Select action according to your policy
        # STEP_ONE: Random action
        # Exploring -> random
        # Best choice -> agent experience Q matrix
        action = random.choice(actions)

        print "LearningAgent.update(): action = {}".format(action)  # [debug]


        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        state_prime = (
            ('next', self.next_waypoint),
            ('light', inputs['light']))
            # ('oncoming', inputs['oncoming']),
            # ('left', inputs['left']),
            # ('right', inputs['right']))
            # ('deadline', deadline))
        print "LearningAgent.update(): state_one = {}".format(state_prime)  # [debug]

        self.update_Q(state, action, reward, state_prime)

        print "------------------------------------------------------------------"  # [debug]
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

    def get_max_Q_for_state(self, state):
        max_q = []

        for action in self.actions:
            if (state, action) not in self.Q:
                return 0
            else:
                max_q.append(self.Q[(state, action)])

        return max(max_q)


    def update_Q(self, state, action, reward, state_prime):
        print "Updating Q MATRIX"

        # Q(s,a) = (1- alpha)*Q(s,a) + alpha*(reward + gamma * max_Q(s', a'))

        # Init value if it doesn't exist: Q(self.state, self.action) = 0
        if (state, action) not in self.Q:
                self.Q[(state, action)] = self.Q_default_value

        self.Q[(state, action)] = (1 - self.alpha) * self.Q[(state, action)] + \
            self.alpha * (reward + self.gamma * \
                self.get_max_Q_for_state(self.state_prime))


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials
    # TODO: Change later enforce_deadline=True

    # Now simulate it
    sim = Simulator(e, update_delay=2, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
