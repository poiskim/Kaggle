# valueIterationAgents.py
# -----------------------
'''
Copyright (C) Computer Science & Engineering, Soongsil University. This material is for educational uses only. 
Some contents are based on the material provided by other paper/book authors and may be copyrighted by them. 
Written by Haneul Pyeon, November 2024.
'''


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp : mdp.MarkovDecisionProcess, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()

        for state in states:
            self.values[state] = 0

        for _ in range(self.iterations):
            nextValues = util.Counter()

            for state in states :
                if self.mdp.isTerminal(state):
                    continue

                action = self.computeActionFromValues(state)
                nextValues[state] = self.computeQValueFromValues(state, action)
            
            self.values = nextValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        avg = 0
        for nextState, probability in self.mdp.getTransitionStatesAndProbs(state, action):
            avg += probability * (self.mdp.getReward(state, action, nextState) + self.discount * self.getValue(nextState))
            
        return avg
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        values = [(action, self.computeQValueFromValues(state, action)) for action in self.mdp.getPossibleActions(state)]

        if len(values) == 0 :
            return None

        value = max(values, key=lambda x : x[1])

        return value[0]

        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp : mdp.MarkovDecisionProcess, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):

        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()

        for state in states:
            self.values[state] = 0

        for idx in range(self.iterations):
            state = states[idx%len(states)]

            if self.mdp.isTerminal(state):
                continue

            action = self.computeActionFromValues(state)
            self.values[state] = self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp : mdp.MarkovDecisionProcess, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        states = self.mdp.getStates()
        pq = util.PriorityQueue()
        predecessor = {}

        for state in states:
            self.values[state] = 0
            
            for action in self.mdp.getPossibleActions(state):
                for nextState, probaility in self.mdp.getTransitionStatesAndProbs(state, action):
                    if nextState not in predecessor:
                        predecessor[nextState] = set()

                    if probaility > 0:
                        predecessor[nextState].add(state)

        for state in states:
            if self.mdp.isTerminal(state):
                continue

            action = self.computeActionFromValues(state)
            value = self.computeQValueFromValues(state, action)
            diff = abs(value - self.values[state])
            pq.push(state, -diff)
        
        for _ in range(self.iterations):
            if pq.isEmpty():
                break
            
            state = pq.pop()

            if not self.mdp.isTerminal(state):
                action = self.computeActionFromValues(state)
                value = self.computeQValueFromValues(state, action)
                self.values[state] = value

            if state not in predecessor:
                continue

            for prevState in predecessor[state]:
                action = self.computeActionFromValues(prevState)
                value = self.computeQValueFromValues(prevState, action)
                diff = abs(value - self.values[prevState])
                if diff > self.theta:
                    pq.update(prevState, -diff)
