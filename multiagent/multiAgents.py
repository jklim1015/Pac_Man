# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        score = 0.5 * successorGameState.getScore()
        successorGhostPos = successorGameState.getGhostPositions()
        newFoodLst = newFood.asList()
        capsule = currentGameState.getCapsules()

        if newFoodLst:
            minD = min([manhattanDistance(newPos, food) for food in newFoodLst])
            score += 0.5/minD
        if successorGhostPos:
            minD = min([manhattanDistance(newPos, ghost) for ghost in successorGhostPos])
            if minD < 3 and max(newScaredTimes) == 0:
                score += 1.5*minD
        if capsule and newPos in capsule:
            score += 1000
        return score

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        possMoves = gameState.getLegalActions(0)
        return max(possMoves, key=lambda action: self.helper(1, 0, gameState.generateSuccessor(0, action)))

    def helper(self, agent, depth, state):
        currMoves = state.getLegalActions(agent)

        if depth == self.depth or state.isLose() or state.isWin():
            return self.evaluationFunction(state)

        elif agent == 0:
            newStates = [state.generateSuccessor(agent, action) for action in currMoves]
            return max([self.helper(agent + 1, depth, newState) for newState in newStates])

        else:
            newStates = [state.generateSuccessor(agent, action) for action in currMoves]
            if agent != state.getNumAgents() - 1:
                return min([self.helper(agent + 1, depth, newState) for newState in newStates])
            newStates = [state.generateSuccessor(agent, action) for action in currMoves]
            return min([self.helper(0, depth + 1, newState) for newState in newStates])


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.helper(0, 0, gameState, float("-inf"), float("inf"))[1]

    def helper(self, agent, depth, state, alpha, beta):
        currMoves = state.getLegalActions(agent)

        if depth == self.depth or state.isLose() or state.isWin():
            return self.evaluationFunction(state), None

        elif agent == 0:
            action, val = None, float("-inf")
            for move in currMoves:
                currVal = self.helper(agent + 1, depth, state.generateSuccessor(agent, move), alpha, beta)[0]
                if val < currVal:
                    val, action = currVal, move
                if currVal > beta:
                    return currVal, move
                alpha = max(alpha, currVal)
            return val, action

        else:
            action, val = None, float("inf")
            for move in currMoves:
                if agent == state.getNumAgents() - 1:
                    currVal = self.helper(0, depth + 1, state.generateSuccessor(agent, move), alpha, beta)[0]
                else:
                    currVal = self.helper(agent + 1, depth, state.generateSuccessor(agent, move), alpha, beta)[0]
                if val > currVal:
                    val, action = currVal, move
                if currVal < alpha:
                    return currVal, move
                beta = min(beta, currVal)
            return val, action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        possMoves = gameState.getLegalActions(0)
        return max(possMoves, key=lambda move: self.helper(1, 0, gameState.generateSuccessor(0, move)))

    def helper(self, agent, depth, state):
        currMoves = state.getLegalActions(agent)

        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        elif agent == 0:
            newStates = [state.generateSuccessor(agent, move) for move in currMoves]
            return max([self.helper(agent + 1, depth, state) for state in newStates])

        else:
            newStates = [state.generateSuccessor(agent, move) for move in currMoves]
            if agent != state.getNumAgents() - 1:
                return sum([self.helper(agent + 1, depth, state) * 1 / len(newStates) for state in newStates])
            return sum([self.helper(0, depth + 1, state) * 1 / len(newStates) for state in newStates])


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <For ghosts, we take into account the all the ghosts w/ ManhattanDistance as well as the ghost scared timer for the closest ghost (if distance less than 1.5). 50 and 20 are numbers gained from testing.
    For food, we take into account all the foods w/ ManhattanDistance and sort by distance. If there is more than one pellet we subtract the score by the second closest food.
    If there is one food left we subtract the score by that distance. >
    """
    "*** YOUR CODE HERE ***"
    position = currentGameState.getPacmanPosition()
    score = currentGameState.getScore()
    ghostStates = currentGameState.getGhostStates()
    foodLst = currentGameState.getFood().asList()
    dist = []

    for ghost in ghostStates:
        if manhattanDistance(position, ghost.getPosition()) < 1.5:
            if ghost.scaredTimer != 0:
                score = score + 20
            else:
                score = score - 50
        continue

    if len(foodLst) > 0:
        for food in foodLst:
            dist.append(manhattanDistance(food, position))
        dist.sort()
        if len(dist) > 1:
            score = score - dist[1]
        else:
            score = score - dist[0]
    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
