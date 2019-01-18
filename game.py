import pygame, sys
from pygame.locals import *
import copy
import random
import math
import numpy as np
from collections import deque




pygame.init()

width = 800
height = 800

nbCellsWidth = 4
nbCellsHeight = 4

# set up the window
windowSurface = pygame.display.set_mode((width, height), 0, 32)
pygame.display.set_caption('Snake!')

# set up the colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

RIGHT = (1, 0)
LEFT = (-1, 0)
UP = (0, -1)
DOWN = (0, 1)

cellWidth = width/nbCellsWidth
cellHeight = height/nbCellsHeight


def compute_rewards(history):
    gamma = 0.99
    computed_rewards = [0 for i in history]
    running_add = 0
    bonus = 0.5
    for i in reversed(range(len(history))):
        running_add = running_add * gamma + history[i][2]
        if running_add <= 0:
            running_add = 0
            bonus = 0
        elif running_add > 0 and bonus == 0:
            bonus = 0.5
            
        computed_rewards[i] = bonus + (running_add/(cellWidth*cellHeight*2))
    return computed_rewards


class NeuralNet:

    # NN Construct

    def __init__(
        self,
        inData,
        outData,
        neurones,
        layersNB,
        learningRate,
        ):

        self.inData = inData
        self.outData = outData
        self.neurones = neurones
        self.layersNB = layersNB
        self.learningRate = learningRate
        self.reset()

    def reset(self):
        self.weights = []

        # Hidden layers initialisation

        for i in range(self.layersNB):
            layer = []
            for j in range(self.neurones):
                layer.append([random.uniform(-0.01, 0.01) for x in
                             range((self.inData if i
                             == 0 else self.neurones))])
            self.weights.append(layer)

        # Out layer

        layer = []
        connectedToFinal = self.neurones if self.layersNB > 0 else self.inData
        for j in range(self.outData):
            layer.append([random.uniform(-0.01, 0.01) for x in
                         range(connectedToFinal)])
        self.weights.append(layer)

    def activation(self, value):

        # Sigmoid fix for math range
        if value < -709:
            return 1
        return 1 / (1 + math.exp(-value))


    def get(self, data):
        # Propagate the input to the output to calculate the result of the neural network
        if len(data) != self.inData:
            print("Error input")
        inData = data
        for layer in range(len(self.weights)):
            outData = []
            for neurone in range(len(self.weights[layer])):
                    outData.append(self.activation(sum([a * b for (a, b) in
                                zip(inData,
                                self.weights[layer][neurone])])))
            inData = outData

        # Return the raw output of the neural network
        return inData

    def train(self, trainData, solutions, onlyUpdateOutputX=False):
        accuracy = 0

        nbIteration = len(trainData)

        for index in range(nbIteration):

            # ---------------------------- PROPAGATION -------------------------------

            outputLayers = []
            t = trainData[index]
            inData = t
            for layer in range(len(self.weights)):
                outputLayers.append(inData)
                outData = []
                for neurone in range(len(self.weights[layer])):
                        outData.append(self.activation(sum([a * b for a, b in zip(inData, self.weights[layer][neurone])])))
                inData = outData

            # ---------------------------- BACK PROP -------------------------------

            expected = solutions[index]
            newLayers = {}

            # Input Data for the back propagation
            if onlyUpdateOutputX == False:
                inBack = [(expected[c] - inData[c]) * inData[c] * (1 - inData[c]) for c in range(len(inData))]
            else:
                inBack = [0 for c in range(len(inData))]
                x = onlyUpdateOutputX[index]
                inBack[x] = (expected - inData[x]) * inData[x] * (1 - inData[x])

            # Propagate the graph reversed
            for layer in reversed(range(len(self.weights))):
                pLayer = layer
                newLayer = []

                # Calculate the correction for the weights
                for neurone in range(len(self.weights[layer])):
                    newWeights = []
                    for w in range(len(self.weights[layer][neurone])):
                        newWeights.append(self.weights[layer][neurone][w] + (outputLayers[pLayer][w] * inBack[neurone] * self.learningRate))

                    newLayer.append(newWeights)
                newLayers[layer] = newLayer

                # Calculate the new input for the next layer
                newInBack = []
                for i in range(len(self.weights[layer][0])):
                    r = 0
                    for j in range(len(self.weights[layer])):
                        r += self.weights[layer][j][i] * inBack[j]
                    newInBack.append(r * outputLayers[pLayer][i] * (1 - outputLayers[pLayer][i]))

                inBack = newInBack

            # Apply weights
            for l in range(len(self.weights)):
                self.weights[l] = newLayers[l]


def random_action():
    return random.randint(0,3)


class Game:
    def __init__(self, render, random):
        self.render = render
        self.random = random
        self.snake = [[0, 0, 2]]


    def generate_apple(self):
        appleWellPlaced = False
        while not appleWellPlaced:
            self.apple = [self.random.randint(0, nbCellsWidth - 1), self.random.randint(0, nbCellsHeight - 1)]
            appleWellPlaced = True
            for x, y, v, in self.snake:
                if self.apple[0] == x and self.apple[1] == y:
                    appleWellPlaced = False
                    break
    def getObs(self):
        global nbCellsWidth
        global nbCellsHeight
        inData = [0 for i in range(nbCellsHeight * nbCellsWidth)]
        for x, y, v in self.snake:
            inData[x + y*nbCellsWidth] = v
        inData[self.apple[0] + self.apple[1]*nbCellsWidth] = -1
        return inData

    def play(self, agent, maxSteps):
        global nbCellsWidth
        global nbCellsHeight
        global cellWidth
        global cellHeight

        self.score = 0

        self.generate_apple()
        cmp = 0

        while True:
            if self.render:
                pygame.event.get()
                windowSurface.fill(WHITE)
                for x, y, v, in self.snake:
                    pygame.draw.rect(windowSurface, BLUE, (x * cellWidth, y * cellHeight, cellWidth, cellHeight))
                pygame.draw.rect(windowSurface, GREEN, (self.apple[0] * cellWidth, self.apple[1] * cellHeight, cellWidth, cellHeight))
                pygame.time.wait(10)
                pygame.display.update()

            direction = agent.getPlay(self)
            tmp = copy.deepcopy(self.snake[-1])
            tmp[2] += 1
            tmp[0] += direction[0]
            tmp[1] += direction[1]

            cmp += 1
            if cmp > maxSteps:
                return self.score


            if (tmp[0] < 0 or tmp[0] >= nbCellsWidth) or tmp[1] < 0 or tmp[1] >= nbCellsHeight:
                return self.score

            for x, y, v in self.snake:
                if x == tmp[0] and y == tmp[1]:
                    return self.score

            if tmp[2] >= nbCellsHeight * nbCellsWidth:
                return self.score

            self.snake.append(tmp)

            if self.apple[0] != tmp[0] or self.apple[1] != tmp[1]:
                for s in range(len(self.snake)):
                    self.snake[s][2] -= 1
            else:
                self.generate_apple()
                self.score += 1
                cmp = 0

            self.snake = list(filter(lambda x: x[2] > 0, self.snake))

direction = RIGHT
directions = [RIGHT, LEFT, UP, DOWN]

class HumanAgent:
    def getPlay(self, game):
        global direction
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    direction = LEFT
                if event.key == pygame.K_RIGHT:
                    direction = RIGHT
                if event.key == pygame.K_DOWN:
                    direction = DOWN
                if event.key == pygame.K_UP:
                    direction = UP
        return direction


history = []
currentHistory = []


lastObs = []
lastA = -1
lastActions = []
lastOut = []
lastScore = 0
e = 0.5

class QAgent:
    def __init__(self):
        global nbCellsHeight
        global nbCellsWidth
        self.QNeuralNet = NeuralNet(nbCellsWidth*nbCellsHeight, 4, nbCellsWidth*nbCellsHeight, 1, 0.1)
        self.display = False
 
    def getPlay(self, game):
        global nbCellsWidth
        global nbCellsHeight
        global directions
        global lastObs
        global lastA
        global lastActions
        global lastScore
        global e
        obs = game.getObs()
        action = self.QNeuralNet.get(obs)
        lastOut = action
        if self.display:
            print("actions ->", action)
            self.display = False
        m = max(action)
        a = action.index(m)
        if random.uniform(0, 1) < e:
            a = random_action()
        if lastA != -1:
            currentHistory.append([lastObs, lastA, game.score - lastScore, obs, lastActions])
        lastA = a
        lastObs = obs
        lastActions = action
        lastScore = game.score
        return directions[a]


maxSteps = nbCellsWidth*nbCellsHeight
currentSeed = 2
batchsize = 10


AIagent = QAgent()
i = 0
experience_replay = deque([], 1000)
scores = []


while True:
    currentSeed = random.randint(0,65535)
    i += 1
    r = random.Random()
    r.seed(currentSeed)

    lastA = -1
    currentHistory = []
    lastScore = 0
    lastActions = []

    # AIagent.display = True

    game = Game(i % 100 == 0, r)
    score = game.play(AIagent, maxSteps)
    currentHistory.append([lastObs, lastA, score - lastScore, lastObs, lastActions])
    history.append(currentHistory)
    if i % batchsize == 0 and i > 0:
        for h in history:
            rewards = compute_rewards(h)
            states = [ob for ob, a, r, nob, actions in h]
            actionsDone = [a for ob, a, r, nob, actions in h]
            AIagent.QNeuralNet.train(states, rewards, actionsDone)
        history = []
    # print("End, score :", score)
    scores.append(score)

    if len(currentHistory) > 0:
        realCurrentHistory = []
        rewards_tmp = compute_rewards(currentHistory)
        for indx, v in enumerate(currentHistory):
            ob, a, r, nob, actions = v
            realCurrentHistory.append([ob, a, rewards_tmp[indx], nob, actions])
        experiences = random.sample(realCurrentHistory, min(50, len(realCurrentHistory)))
        for experience in experiences:
            experience_replay.appendleft(experience)

    if i % 100 == 0 and i > 0:
        lastAverageReward = sum(scores[-100:])/len(scores[-100:])
        print("Last mean : ", lastAverageReward, "( e =",e,")")

    if len(experience_replay) > 0:
        experiences = random.sample(experience_replay, min(200, len(experience_replay)))
        for experience in experiences:
            AIagent.QNeuralNet.train([experience[0]], [experience[2]], [experience[1]])

    e = 1./(i/5000+2)
    

print("End")

#agent = HumanAgent()
#game = Game(True, random.Random())
#game.play(agent,99999)