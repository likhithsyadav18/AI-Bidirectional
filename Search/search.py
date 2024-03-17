# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from searchAgents import manhattanHeuristic
import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def getGoalState(self):
        """
        Returns the end state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def graphTreeSearch(problem, fringe, search_type, heuristic):
    startingNode = problem.getStartState()
    if problem.isGoalState(startingNode):
        return []
    
    if search_type in ['dfs', 'bfs']:
        fringe.push((startingNode, []))
    elif search_type in ['ucs', 'a*s']:
        fringe.push((startingNode, [], 0), problem.getCostOfActions([]))
    else:
        return []

    visitedNodes = list()

    while not fringe.isEmpty():

        if search_type in ['dfs', 'bfs']:
            currentNode, actions = fringe.pop()
        elif search_type in ['ucs', 'a*s']:
            currentNode, actions, prevCost = fringe.pop()
        else:
            return []
    
        if currentNode not in visitedNodes:
            visitedNodes.append(currentNode)

            if problem.isGoalState(currentNode):
                return actions
            
            childNodes = problem.getSuccessors(currentNode)
            for nextNode, action, cost in childNodes:
                newAction = actions + [action]
                
                if search_type in ['a*s', 'ucs']:
                    newCostToNode = prevCost + cost
                    if search_type == 'a*s':
                        priority = newCostToNode + heuristic(nextNode, problem)
                    else:           # search_type == 'ucs'
                        priority = newCostToNode
                    fringe.push((nextNode, newAction, newCostToNode), priority)
                else:   # search_type == 'dfs' and 'bfs'
                    fringe.push((nextNode, newAction))

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    stack = util.Stack()
    return graphTreeSearch(problem, stack, 'dfs', None)
    # util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = util.Queue()
    return graphTreeSearch(problem, queue, 'bfs', None)
    # util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    priority_queue = util.PriorityQueue()
    return graphTreeSearch(problem, priority_queue, 'ucs', None)
    # util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    priority_queue = util.PriorityQueue()
    return graphTreeSearch(problem, priority_queue, 'a*s', heuristic)
    # util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

#################################################################
'''
Here is the implementation done for the team project part. 

Project topic 1. Bi-directional search
Reference Paper: Bidirectional Search That Is Guaranteed to Meet in the Middle
'''
#################################################################

def flipActionPath(actions):
    directionsReverse = {'East': 'West', 'West': 'East', 'North': 'South', 'South': 'North'}
    actionsReverse = [directionsReverse[action] for action in reversed(actions)]
    return actionsReverse

def graphTreeBiSearch(problem, fringe_forward, fringe_backward, search_type, heuristic):
    startingNodeF = endingNodeB = problem.getStartState()
    endingNodeF = startingNodeB = problem.getGoalState()

    if search_type == 'MM0':
        heuristic = nullHeuristic
    else:
        heuristic = manhattanHeuristic

    if ((startingNodeF == startingNodeB) or (endingNodeF == endingNodeB)):
        return []
    
    opensetForwards = dict()            #Open set of unvisited nodes for forward search
    closedsetForwards = dict()          #Closed set of visited nodes for forward search

    opensetBackwards = dict()           #Open set of unvisited nodes for forward search
    closedsetBackwards = dict()         #Closed set of visited nodes for backward search

    # '''
    # A* works using the following formula, f(n) = g(n) + h(n) where,
    # f(n) is the total cost estimate of the optimal path.
    # g(n) is the actual cost function of the path to node 'n' which in terms of code is given by problem.getCostOfActions() function
    # h(n) is the heuristic function of the path to node 'n' which in terms of code is written as heuristic(state, problem)
    #
    # So, for a bidirectional search node should have following properties:
    #   {Node Value(n), Node Path(actions), Cost value(g(n)), Heuristic Value(h(n)), Priority Value} within a priority queue.
    # '''
    flipHeuristic = {}

    nodeProperties = dict()

    # For forward search
    closedsetForwards[startingNodeF] = 0

    flipHeuristic['flip'] = 'False'

    nodeProperties[startingNodeF] = {
        'actions': [],
        'costValue': problem.getCostOfActions([]),
        'heuristicValue': heuristic(startingNodeF, problem, flipHeuristic),
        'totalCostValue': 0,
        'priority': 0
    }

    fringe_forward.push(startingNodeF, 0)


    # For backward search
    closedsetBackwards[startingNodeB] = 0

    flipHeuristic['flip'] = 'True'

    nodeProperties[startingNodeB] = {
        'actions': [],
        'costValue': problem.getCostOfActions([]),
        'heuristicValue': heuristic(startingNodeB, problem, flipHeuristic),
        'totalCostValue': 0,
        'priority': 0
    }

    fringe_backward.push(startingNodeB, 0)

    '''
    U is the cost of the cheapest solution found so far in the state space.
    Initially, the value of U is considered as infinity as there is no cost found until there is a traversal in the state space.
    '''
    U = float('inf')

    '''
    Variable 'eps' is the cost of the cheapest edge in the state space. 
    Here in the pacman domain, the edge costs are unit. So, the value of eps is considered as 1.
    '''
    eps = 1 

    '''
    For tracing the resultant path, we have considered a list named actionPath.
    '''
    actionPath = list()

    recentNodeB = None
    recentNodeF = None

    while(not fringe_forward.isEmpty()) and (not fringe_backward.isEmpty()):
        
        # This is for the forward search
        curNodeF = fringe_forward.pop()
        opensetForwards[curNodeF] = 0
        closedsetForwards[curNodeF] = 1

        # For calculating custom priority of the node using the function priority(n) = max(f(n), 2*g(n)) according to the reference algorithm.
        nodeProperties[curNodeF]['totalCostValue'] = nodeProperties[curNodeF]['costValue'] + nodeProperties[curNodeF]['heuristicValue']
        fVal_Forwards = nodeProperties[curNodeF]['totalCostValue']
        gVal_Forwards = nodeProperties[curNodeF]['costValue']
        priorityMinForwards = max(fVal_Forwards, 2*gVal_Forwards)

        # This is for the backward search
        curNodeB = fringe_backward.pop()
        opensetBackwards[curNodeB] = 1
        closedsetBackwards[curNodeB] = 0

        # For calculating custom priority of the node using the function priority(n) = max(f(n), 2*g(n)) according to the reference algorithm.
        nodeProperties[curNodeB]['totalCostValue'] = nodeProperties[curNodeB]['costValue'] + nodeProperties[curNodeB]['heuristicValue'] 
        fVal_Backwards = nodeProperties[curNodeB]['totalCostValue']
        gVal_Backwards = nodeProperties[curNodeB]['costValue']
        priorityMinBackwards = max(fVal_Backwards, 2*gVal_Backwards)

        '''
        C is the cost of an optimal solution and it is given as C = min(priorityMinForwards, priorityMinBackwards)
        '''
        C = min(priorityMinForwards, priorityMinBackwards)

        '''
        U is the cost of the cheapest solution found so far. So, it is updated everytime when it satisfies the following condition.
        if U <= max(C, fminF, fminB, gminF +gminB + eps) or if U <= C
        then
            return U        --> This means the resultant cost found after the bidirectional search.
        
        In this program, we would like to find the resultant path as the pacman needs the path for traversal.

        Here, max(C, fminF, fminB, gminF +gminB + eps) is lower cost bound and 'eps' is the cost of the cheapest edge in the state space.
        '''
        lowerCostBound = max(C, fVal_Forwards, fVal_Backwards, gVal_Forwards + gVal_Backwards + eps)
        if (U <= lowerCostBound) or (U <= C):
            return actionPath
        
        if (C == priorityMinForwards):
            recentNodeF = curNodeF

            opensetForwards[curNodeF] = 0
            closedsetForwards[curNodeF] = 1
            opensetBackwards[curNodeB] = 1

            fringe_backward.push(curNodeB, priorityMinBackwards)
            
            childNodesF = problem.getSuccessors(curNodeF)
            for nextChildNodeF, actionChildNodeF, costChildNodeF in childNodesF:

                if (nextChildNodeF in opensetForwards):
                    nextNodeF = fringe_forward[nextChildNodeF]
                    closedsetForwards[nextChildNodeF] = 0

                elif (nextChildNodeF in closedsetForwards):
                    nextNodeF = fringe_forward[nextChildNodeF]
                    closedsetForwards[nextChildNodeF] = 0

                else:
                    nextNodeF = None

                if nextNodeF is not None:
                    if (nodeProperties[nextNodeF]['costValue'] <= nodeProperties[curNodeF]['costValue'] + costChildNodeF):
                        continue

                    nodeProperties[nextNodeF]['costValue'] = nodeProperties[curNodeF]['costValue'] + costChildNodeF
                    nodeProperties[nextNodeF]['heuristicValue'] = heuristic(nextNodeF, problem)
                    nodeProperties[nextNodeF]['totalCostValue'] = nodeProperties[nextNodeF]['costValue'] + nodeProperties[nextNodeF]['heuristicValue']
                    nodeProperties[nextNodeF]['actions'] = list(nodeProperties[curNodeF]['actions']) + [actionChildNodeF]
                    nodeProperties[nextNodeF]['priority'] = max( nodeProperties[nextNodeF]['totalCostValue'], 2 * nodeProperties[nextNodeF]['costValue'])
                    
                else:
                    nodeProperties[nextChildNodeF]['costValue'] = nodeProperties[curNodeF]['costValue'] + costChildNodeF
                    nodeProperties[nextChildNodeF]['heuristicValue'] = heuristic(nextChildNodeF, problem)
                    nodeProperties[nextChildNodeF]['totalCostValue'] = nodeProperties[nextChildNodeF]['costValue'] + nodeProperties[nextChildNodeF]['heuristicValue']
                    nodeProperties[nextChildNodeF]['actions'] = list(nodeProperties[curNodeF]['actions']) + [actionChildNodeF]
                    nodeProperties[nextChildNodeF]['priority'] = max( nodeProperties[nextChildNodeF]['totalCostValue'], 2 * nodeProperties[nextChildNodeF]['costValue'])

                    fringe_forward.push(nextChildNodeF, nodeProperties[nextChildNodeF]['priority'])


                fringe_forward.push(nextNodeF, nodeProperties[nextNodeF]['priority'])
                opensetForwards[nextNodeF] = 1

                if (nextChildNodeF in opensetBackwards):
                    nextNodeB = fringe_backward[nextChildNodeF]                
                    U = min(U, nodeProperties[nextNodeF]['costValue'] + nodeProperties[nextNodeB]['costValue'])
                    actionPath = list(nodeProperties[nextNodeF]['actions']) + flipActionPath(nodeProperties[nextNodeB]['actions'])
        
        else:
            recentNodeB = curNodeB

            opensetBackwards[curNodeB] = 0
            closedsetBackwards[curNodeB] = 1
            opensetForwards[curNodeF] = 1

            fringe_forward.push(curNodeF, priorityMinForwards)
            
            childNodesB = problem.getSuccessors(curNodeB)
            for nextChildNodeB, actionChildNodeB, costChildNodeB in childNodesB:

                if (nextChildNodeB in opensetBackwards):
                    nextNodeB = fringe_backward[nextChildNodeB]
                    closedsetBackwards[nextChildNodeB] = 0
                
                elif (nextChildNodeB in opensetBackwards):
                    nextNodeB = fringe_backward[nextChildNodeB]
                    closedsetBackwards[nextChildNodeB] = 0

                else:
                    nextNodeB = None

                if nextNodeB is not None:
                    if (nodeProperties[nextNodeB]['costValue'] <= nodeProperties[curNodeB]['costValue'] + costChildNodeB):
                        continue

                    nodeProperties[nextNodeB]['costValue'] = nodeProperties[curNodeB]['costValue'] + costChildNodeB
                    nodeProperties[nextNodeB]['heuristicValue'] = heuristic(nextNodeB, problem)
                    nodeProperties[nextNodeB]['totalCostValue'] = nodeProperties[nextNodeB]['costValue'] + nodeProperties[nextNodeB]['heuristicValue']
                    nodeProperties[nextNodeB]['actions'] = list(nodeProperties[curNodeB]['actions']) + [actionChildNodeB]
                    nodeProperties[nextNodeB]['priority'] = max( nodeProperties[nextNodeB]['totalCostValue'], 2 * nodeProperties[nextNodeB]['costValue'])
                    
                else:
                    nodeProperties[nextChildNodeB]['costValue'] = nodeProperties[curNodeB]['costValue'] + costChildNodeB
                    nodeProperties[nextChildNodeB]['heuristicValue'] = heuristic(nextChildNodeB, problem)
                    nodeProperties[nextChildNodeB]['totalCostValue'] = nodeProperties[nextChildNodeB]['costValue'] + nodeProperties[nextChildNodeB]['heuristicValue']
                    nodeProperties[nextChildNodeB]['actions'] = list(nodeProperties[curNodeB]['actions']) + [actionChildNodeB]
                    nodeProperties[nextChildNodeB]['priority'] = max( nodeProperties[nextChildNodeB]['totalCostValue'], 2 * nodeProperties[nextChildNodeB]['costValue'])

                    fringe_backward.push(nextChildNodeB, nodeProperties[nextChildNodeB]['priority'])


                fringe_backward.push(nextNodeB, nodeProperties[nextNodeB]['priority'])
                opensetBackwards[nextNodeB] = 1

                if (nextChildNodeB in opensetForwards):
                    nextNodeF = fringe_forward[nextChildNodeB]                
                    U = min(U, nodeProperties[nextNodeB]['costValue'] + nodeProperties[nextNodeF]['costValue'])
                    actionPath = list(nodeProperties[nextNodeB]['actions']) + flipActionPath(nodeProperties[nextNodeF]['actions'])

    return nodeProperties[recentNodeF]['actions'] + flipActionPath(nodeProperties[recentNodeB]['actions']) if ((recentNodeB is not None) and (recentNodeF is not None)) else []


def biDirectionalAStarSearch(problem, heuristic):
    priority_queue_fwd = util.PriorityQueue()
    priority_queue_bwd = util.PriorityQueue()
    return graphTreeBiSearch(problem, priority_queue_fwd, priority_queue_bwd, 'MM', heuristic)

def biDirectionalAStarSearchBruteForce(problem, heuristic):
    # biDirectionalAStarSearchBruteForce is biDirectionalAStarSearch with null heuristic.
    priority_queue_fwd = util.PriorityQueue()
    priority_queue_bwd = util.PriorityQueue()
    return graphTreeBiSearch(problem, priority_queue_fwd, priority_queue_bwd, 'MM0', heuristic)

# Abbreviations
bds_mm0 = biDirectionalAStarSearchBruteForce
bds_mm = biDirectionalAStarSearch