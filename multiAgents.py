# multiAgents.py
# --------------
'''
Copyright (C) Computer Science & Engineering, Soongsil University. This material is for educational uses only. 
Some contents are based on the material provided by other paper/book authors and may be copyrighted by them. 
Written by Haneul Pyeon, October 2024.
'''


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        x, y = newPos

        def getDist(i, j, x=x, y=y):
            return abs(i-x) + abs(j-y)

        dist = []
        for s in newGhostStates:
            i, j = s.getPosition()
            dist.append(getDist(i, j))
        
        if min(dist) == 0:
            return -1000000
        elif min(dist) == 1:
            return -100000
        elif min(dist) == 2:
            return -10000

        nowFood = currentGameState.getFood()
        q = util.Queue()
        visited = {(x, y)}
        q.push((x, y, 0))
        di = [1, 0, -1, 0]
        dj = [0, 1, 0, -1]

        while not q.isEmpty():
            i, j, c = q.pop()
            if i < 0 or j < 0:
                continue
            try:
                if nowFood[i][j]:
                    return successorGameState.getScore() - c
                if successorGameState.hasWall(i, j):
                    continue
            except: continue

            for d in range(4):
                ni = i + di[d]
                nj = j + dj[d]

                if (ni, nj) in visited:
                    continue

                visited.add((ni,nj))
                q.push((ni, nj, c+1))

        return successorGameState.getScore() - 200

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        from pacman import GameState

        def selectMinMaxNode(depth, agentIndex, state : GameState):
            #depth를 넘어섰다면 그만하기
            if depth > self.depth : 
                return state.getScore()

            if state.isWin():
                return 1234567890
            
            if state.isLose():
                return -1234567890
            
            #현재 노드에서 받을 수 있는 점수 리스트 가져오기
            miniMaxList = getMiniMaxList(depth, agentIndex, state)

            #index에 따라 최소, 최대 선택
            if agentIndex == 0:
                minimaxScore = max(miniMaxList)
            else:
                minimaxScore = min(miniMaxList)

            bestIndices = [index for index in range(len(miniMaxList)) if miniMaxList[index] == minimaxScore]
            chosenIndex = random.choice(bestIndices) # Pick randomly among the best
            #리턴
            return minimaxScore, chosenIndex

        #Seeing all legal action in given state and get minimax score with list type and nextState
        def getMiniMaxList(depth, agentIndex, state : GameState):
            nextIndex = agentIndex + 1

            n = state.getNumAgents()
            if nextIndex >= n:
                nextIndex -= n
                depth += 1
            
            miniMaxList = []
            for action in state.getLegalActions():
                print(state, action)
                nextState = state.generateSuccessor(agentIndex, action)
                miniMaxList.append(selectMinMaxNode(nextIndex, depth, nextState)[0])
            
            return miniMaxList

        _, index = selectMinMaxNode(0, 0, gameState)
        return gameState.getLegalActions()[index]
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
