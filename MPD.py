# -*- coding: utf-8 -*-
"""

Author: flztiii

Policy iteration and value iteration to solve markov decision process

"""

import sys
import random
import copy
import numpy as np

# all actions
ACTIONS = ('l', 'r', 'u', 'd')

# state
class State:
    # construction
    def __init__(self, x, y):
        self.x_ = x
        self.y_ = y
    
    # update agent
    def update(self, action):
        if action == 'l':
            self.x_ -= 1
        elif action == 'r':
            self.x_ += 1
        elif action == 'u':
            self.y_ -= 1
        elif action == 'd':
            self.y_ += 1
        else:
            print("error action")
            sys.exit(0)

# environment setting (a maze with start and goal point)
class Env:
    # construction
    def __init__(self, width, height):
        assert(width > 2 and height > 2)
        self.width_ = width
        self.height_ = height
        self.goal_ = (random.randint(0, width - 1), random.randint(0, height - 1))
        while True:
            self.trap_ = (random.randint(0, width - 1), random.randint(0, height - 1))
            if self.trap_ != self.goal_:
                break

    # judge if valid state
    def verify(self, state):
        if state.x_ < 0 or state.x_ >= self.width_ or state.y_ < 0 or state.y_ >= self.height_:
            return False
        if state.x_ == self.trap_[0] and state.y_ == self.trap_[1]:
            return False
        if state.x_ == self.goal_[0] and state.y_ == self.goal_[1]:
            return False
        return True
    
    # judge if finished
    def done(self, raw_state, action):
        state = copy.deepcopy(raw_state)
        # update state
        state.update(action)
        # judge if outside
        if state.x_ < 0 or state.x_ >= self.width_ or state.y_ < 0 or state.y_ >= self.height_:
            return True
        # judge if trapped
        if state.x_ == self.trap_[0] and state.y_ == self.trap_[1]:
            return True
        # judge if arrive the goal
        if state.x_ == self.goal_[0] and state.y_ == self.goal_[1]:
            return True
        return False
        
        
    def reward(self, raw_state, action):
        state = copy.deepcopy(raw_state)
        # update state
        state.update(action)
        # judge if outside
        if state.x_ < 0 or state.x_ >= self.width_ or state.y_ < 0 or state.y_ >= self.height_:
            return -self.width_ * self.height_
        # judge if trapped
        if state.x_ == self.trap_[0] and state.y_ == self.trap_[1]:
            return -self.width_ * self.height_
        # judge if arrive the goal
        if state.x_ == self.goal_[0] and state.y_ == self.goal_[1]:
            return self.width_ * self.height_
        return -1;

# policy iteration solver
def policyIterationSolver(env):
    # actions
    global ACTIONS
    # discount
    r = 0.9
    # print goal and trap
    print("goal is ", env.goal_[0], env.goal_[1])
    print("trap is ", env.trap_[0], env.trap_[1])
    # random initialize values and policy
    values = np.zeros((env.height_, env.width_))
    policies = list()
    for y in range(0, env.height_):
        row = list()
        for x in range(0, env.width_):
            row.append(ACTIONS[random.randint(0, 3)])
        policies.append(row)
    # start to iteration
    while True:
        # policy evaluation
        while True:
            delta = 0
            for y in range(0, values.shape[0]):
                for x in range(0, values.shape[1]):
                    if env.verify(State(x, y)):
                        raw_value = values[y][x]
                        current_state = State(x, y)
                        action = policies[y][x]
                        reward = env.reward(current_state, action)
                        done = env.done(current_state, action)
                        current_state.update(action)
                        if not done:
                            values[y][x] = reward + r * values[current_state.y_][current_state.x_]
                        else:
                            values[y][x] = reward
                        delta = max(delta, np.abs(raw_value - values[y][x]))
            if delta < 1e-4:
                break
        # policy improvement
        policy_stable = True
        for y in range(0, values.shape[0]):
            for x in range(0, values.shape[1]):
                if env.verify(State(x, y)):
                    raw_policy = policies[y][x]
                    max_value = -float('inf')
                    new_policy = raw_policy
                    for action in ACTIONS:
                        current_state = State(x, y)
                        reward = env.reward(current_state, action)
                        done = env.done(current_state, action)
                        current_state.update(action)
                        if not done:
                            value = reward + r * values[current_state.y_][current_state.x_]
                        else:
                            value = reward
                        if value > max_value:
                            max_value = value
                            new_policy = action
                    policies[y][x] = new_policy
                    if new_policy != raw_policy:
                        policy_stable = False
        if policy_stable:
            break
    # iteration finished, visualize result
    for y in range(0, values.shape[0]):
        vis = ""
        for x in range(0, values.shape[1]):
            if x == env.goal_[0] and y == env.goal_[1]:
                vis += '@' + ','
            elif x == env.trap_[0] and y == env.trap_[1]:
                vis += 'X' + ','
            else:
                vis += policies[y][x] + ','
        print(vis)

# value iteration solver
def valueIterationSolver(env):
    # actions
    global ACTIONS
    # discount
    r = 0.9
    # print goal and trap
    print("goal is ", env.goal_[0], env.goal_[1])
    print("trap is ", env.trap_[0], env.trap_[1])
    # random initialize values
    values = np.zeros((env.height_, env.width_))
    # start to iteration
    while True:
        delta = 0
        for y in range(0, values.shape[0]):
            for x in range(0, values.shape[1]):
                if env.verify(State(x, y)):
                    raw_value = values[y][x]
                    max_value = -float("inf")
                    for action in ACTIONS:
                        current_state = State(x, y)
                        reward = env.reward(current_state, action)
                        done = env.done(current_state, action)
                        current_state.update(action)
                        if not done:
                            value = reward + r * values[current_state.y_][current_state.x_]
                        else:
                            value = reward
                        if value > max_value:
                            max_value = value
                    values[y][x] = max_value
                    delta = max(delta, np.abs(raw_value - max_value))
        if delta < 1e-4:
            break
    
    # initialize policy
    policies = list()
    for y in range(0, env.height_):
        row = list()
        for x in range(0, env.width_):
            row.append(ACTIONS[random.randint(0, 3)])
        policies.append(row)
    # calculate policy
    for y in range(0, env.height_):
        for x in range(0, env.width_):
            if env.verify(State(x, y)):
                raw_policy = policies[y][x]
                max_value = -float('inf')
                new_policy = raw_policy
                for action in ACTIONS:
                    current_state = State(x, y)
                    reward = env.reward(current_state, action)
                    done = env.done(current_state, action)
                    current_state.update(action)
                    if not done:
                        value = reward + r * values[current_state.y_][current_state.x_]
                    else:
                        value = reward
                    if value > max_value:
                        max_value = value
                        new_policy = action
                policies[y][x] = new_policy

    # visualize result
    for y in range(0, values.shape[0]):
        vis = ""
        for x in range(0, values.shape[1]):
            if x == env.goal_[0] and y == env.goal_[1]:
                vis += '@' + ','
            elif x == env.trap_[0] and y == env.trap_[1]:
                vis += 'X' + ','
            else:
                vis += policies[y][x] + ','
        print(vis)

# main function
def main():
    # init env
    env = Env(4, 3)
    # solve markov decision process
    policyIterationSolver(env)
    # solve markov decision process
    valueIterationSolver(env)

if __name__ == "__main__":
    main()
