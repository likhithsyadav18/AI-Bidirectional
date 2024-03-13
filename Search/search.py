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


def graphTreeBiSearch(problem, fringe_forward, fringe_backward, search_type, heuristic):
    startingNode = problem.getStartState()
    endingNode = problem.getGoalState()

    if startingNode == endingNode:
        return []
    
    visitForwards = dict()
    visitBackwards = dict()

    '''
    A* works using the following formula, f(n) = g(n) + h(n) where,
    g(n) is the cost function which in terms of code is given by problem.getCostOfActions() function
    h(n) is the heuristic function which in terms of code is written as heuristic(state, problem)
    '''
    
    fringe_forward.push((startingNode), (problem.getCostOfActions({}) + heuristic(startingNode, problem, 'endState')))
    fringe_backward.push((endingNode), (problem.getCostOfActions({}) + heuristic(endingNode, problem, 'startState')))
    
    visitForwards[startingNode] = []
    visitBackwards[endingNode] = []

    U = float('inf')

    while((not fringe_forward.isEmpty()) and (not fringe_backward.isEmpty()):
          
    pass

def biDirectionalAStarSearch(problem, heuristic):
    priority_queue_fwd = util.PriorityQueue()
    priority_queue_bwd = util.PriorityQueue()
    return graphTreeBiSearch(problem, priority_queue_fwd, priority_queue_bwd, heuristic)