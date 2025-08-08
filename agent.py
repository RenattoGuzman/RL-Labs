import numpy as np
import pprint


class State(object):
    """
    Represents a state or a point in the grid.

    coord: coordinate in grid world
    """

    def __init__(self, coord, is_terminal):
        self.coord = coord
        self.action_state_transitions = self._getActionStateTranstions()
        self.is_terminal = is_terminal
        self.reward = 5 if is_terminal else -1

    # Returns a dictionary mapping each action to the following state
    # it would put the agent in from the currrent state
    def _getActionStateTranstions(self):
        action_state_transitions = {}
        # Action 0 - up
        if self._isFirstRowState():
            action_state_transitions[0] = self.coord
        else:
            # prev row, same col
            action_state_transitions[0] = (self.coord[0] - 1, self.coord[1])

        # Action 1 - right
        if self._isLastColState():
            action_state_transitions[1] = self.coord
        else:
            # same row, next col
            action_state_transitions[1] = (self.coord[0], self.coord[1] + 1)

        # Action 2 - down
        if self._isLastRowState():
            action_state_transitions[2] = self.coord
        else:
            # next row, same col
            action_state_transitions[2] = (self.coord[0] + 1, self.coord[1])

        # Action 3 - left
        if self._isFirstColState():
            action_state_transitions[3] = self.coord
        else:
            # same row, prev col
            action_state_transitions[3] = (self.coord[0], self.coord[1] - 1)

        return action_state_transitions

    def _isFirstRowState(self):
        return self.coord[0] == 0

    def _isLastRowState(self):
        return self.coord[0] == 3

    def _isFirstColState(self):
        return self.coord[1] == 0

    def _isLastColState(self):
        return self.coord[1] == 3

    # Returns if the current state is a terminal state
    def isTerminal(self):
        return self.is_terminal

    # Gets the action required to move the agent from the current state
    # to some state s2. If the agent cannot move to s2 it returns None
    def getActionTransiton(self, s2):
        for action, next_state in self.action_state_transitions.items():
            if next_state == s2.coord:
                return action
        return None

    # Returns the likelihood of ending up in state s_prime after taking
    # action a from the current state
    def getNextStateLikelihood(self, a, s_prime):
        if self.action_state_transitions[a] == s_prime.coord:
            return 1
        else:
            return 0

    # Return the reward for stepping into this state
    def getReward(self):
        return self.reward


class DynamicProgrammingAgent(object):
    """
    Base implementation of a Dynamic Programming Agent for the Grid World Problem

    env: Gym env the agent will be trained on
    """

    def __init__(self, gamma):
        self.gamma = gamma

        # of states and actions for the grid world problem
        self.num_states = 16
        self.num_actions = 4

    # Prints the values of each state on the grid
    def _printStateValues(self, V):
        grid = np.zeros([4, 4])

        for state, value in V.items():
            x = state.coord[0]
            y = state.coord[1]
            grid[x, y] = value

        print("Value Function--------------------------")
        pprint.pprint(grid)
        print('\n')

    # Prints the policy as a grid of arrows
    def _printPolicy(self, pi):
        grid = np.zeros([4, 4])

        for state, actions in pi.items():
            x = state.coord[0]
            y = state.coord[1]
            action = np.argmax(actions)
            grid[x, y] = action

        # Convert actions to arrows
        arrow_grid = []
        for row_index, row in enumerate(grid):
            arrow_grid_row = []
            for col_index, action in enumerate(row):
                arrow_char = ''
                if (row_index == 0 and col_index == 0) or (row_index == 3 and col_index == 3):
                    arrow_grid_row.append('T')  # Terminal states
                else:
                    if action == 0:
                        arrow_char = '↑'
                    elif action == 1:
                        arrow_char = '→'
                    elif action == 2:
                        arrow_char = '↓'
                    elif action == 3:
                        arrow_char = '←'
                    arrow_grid_row.append(arrow_char)
            arrow_grid.append(arrow_grid_row)

        print("Policy--------------------------")
        pprint.pprint(arrow_grid)
        print('\n')

    # Initialize the states (S), state value function (V), and the policy (pi)
    def initSVAndPi(self):
        self.S = []
        V = {}
        pi = {}
        for r in range(4):
            for c in range(4):
                # Create the state
                is_terminal = False
                if (r == 0 and c == 0) or (r == 3 and c == 3):
                    is_terminal = True
                s = State((r, c,), is_terminal)
                self.S.append(s)
                # Initialize the value of every state to 0
                V[s] = 0
                # Begin with a policy that selects every action with equal probability
                pi[s] = self.num_actions * [0.25]
        return V, pi

    # Gets the action values for a state by getting the expected return
    # of taking each action
    def getActionValuesForState(self, s, V):
        action_values = []
        for action in range(self.num_actions):
            action_value = 0
            for s_prime in self.S:
                p = s.getNextStateLikelihood(action, s_prime)
                action_value += p * (s_prime.getReward() + self.gamma * V[s_prime])
            action_values.append(action_value)
        return action_values


class PolicyIterationAgent(DynamicProgrammingAgent):
    """
    Implementation of an agent that uses policy iteration to derive
    the optimal policy (pi*) and state value function (v*) on the 4x4 grid world.

    gamma: discount factor
    """

    def __init__(self, gamma):
        # Call base class constructor
        super().__init__(gamma)
        print("Policy Iteration Agent")

    def policyIterate(self):
        """
        Policy Iteration Algorithm:
        1. Initialize policy π arbitrarily
        2. Repeat:
            - Policy Evaluation: compute V^π using iterative policy evaluation
            - Policy Improvement: improve π using policy improvement
        Until policy is stable
        """
        print("Starting Policy Iteration...")

        # Initialize states, value function, and policy
        V, pi = self.initSVAndPi()

        iteration = 0
        policy_stable = False

        while not policy_stable:
            iteration += 1
            print(f"\n=== Iteration {iteration} ===")

            # Policy Evaluation: compute state-value function for current policy
            V = self._policyEvaluation(pi, V)

            # Print current state values
            print(f"State Values after Policy Evaluation:")
            self._printStateValues(V)

            # Policy Improvement: improve policy based on current value function
            pi, policy_stable = self.policyImprove(pi, V)

            # Print current policy
            print(f"Policy after Policy Improvement:")
            self._printPolicy(pi)

            if policy_stable:
                print("Policy is stable! Convergence achieved.")
                break

        print(f"\nPolicy Iteration converged in {iteration} iterations!")
        print("Final Results:")
        print("==============")
        self._printStateValues(V)
        self._printPolicy(pi)

        return V, pi

    def _policyEvaluation(self, pi, V, theta=0.001, max_iterations=1000):
        """
        Iterative Policy Evaluation:
        Compute the state-value function V^π for a given policy π

        Args:
            pi: current policy (dict mapping states to action probabilities)
            V: current value function (dict mapping states to values)
            theta: threshold for convergence
            max_iterations: maximum number of iterations to prevent infinite loops
        """
        iteration = 0

        while iteration < max_iterations:
            delta = 0

            # For each state, update its value
            for s in self.S:
                if s.isTerminal():
                    continue  # Terminal states have fixed value

                v_old = V[s]
                v_new = 0

                # Sum over all actions weighted by policy probabilities
                for action in range(self.num_actions):
                    action_prob = pi[s][action]

                    # Sum over all possible next states
                    for s_prime in self.S:
                        transition_prob = s.getNextStateLikelihood(action, s_prime)
                        reward = s_prime.getReward()
                        v_new += action_prob * transition_prob * (reward + self.gamma * V[s_prime])

                V[s] = v_new
                delta = max(delta, abs(v_old - v_new))

            iteration += 1

            # Check for convergence
            if delta < theta:
                print(f"Policy Evaluation converged in {iteration} iterations (delta = {delta:.6f})")
                break

        return V

    def policyImprove(self, pi, V):
        """
        Policy Improvement:
        Create an improved policy based on the current value function

        Args:
            pi: current policy (dict mapping states to action probabilities)
            V: current value function (dict mapping states to values)

        Returns:
            new_pi: improved policy
            policy_stable: boolean indicating if policy has converged
        """
        new_pi = {}
        policy_stable = True

        # For each state, find the best action(s)
        for s in self.S:
            if s.isTerminal():
                # Terminal states don't need actions
                new_pi[s] = pi[s]  # Keep the same (doesn't matter)
                continue

            # Get old action (the one with highest probability in current policy)
            old_action = np.argmax(pi[s])

            # Calculate action values for this state
            action_values = self.getActionValuesForState(s, V)

            # Find the best action(s)
            best_action = np.argmax(action_values)
            max_value = action_values[best_action]

            # Create new policy: deterministic policy selecting best action(s)
            new_policy = [0.0] * self.num_actions

            # Handle ties by distributing probability equally among best actions
            best_actions = [i for i, val in enumerate(action_values) if abs(val - max_value) < 1e-10]
            prob_per_action = 1.0 / len(best_actions)

            for action in best_actions:
                new_policy[action] = prob_per_action

            new_pi[s] = new_policy

            # Check if policy changed for this state
            if old_action != best_action:
                policy_stable = False

        return new_pi, policy_stable