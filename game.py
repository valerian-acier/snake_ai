import pygame, sys
from pygame.locals import *
import copy
import random
import math

# set up pygame
pygame.init()

width = 800
height = 800

nbCellsWidth = 10
nbCellsHeight = 10

# set up the window
windowSurface = pygame.display.set_mode((width, height), 0, 32)
pygame.display.set_caption('Snake!')

# set up the colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

RIGHT = (1, 0)
LEFT = (-1, 0)
UP = (0, -1)
DOWN = (0, 1)

cellWidth = width/nbCellsWidth
cellHeight = height/nbCellsHeight



class Game:
    def __init__(self, render, random):
        self.render = render
        self.random = random
        self.snake = [[5, 5, 2]]


    def generate_apple(self):
        appleWellPlaced = False
        while not appleWellPlaced:
            self.apple = [self.random.randint(0, nbCellsWidth - 1), self.random.randint(0, nbCellsHeight - 1)]
            appleWellPlaced = True
            for x, y, v, in self.snake:
                if self.apple[0] == x and self.apple[1] == y:
                    appleWellPlaced = False
                    break

    def play(self, agent):
        global nbCellsWidth
        global nbCellsHeight
        global cellWidth
        global cellHeight

        score = 0

        self.generate_apple()
        cmp = 0

        while True:
            if self.render:
                windowSurface.fill(WHITE)
                for x, y, v, in self.snake:
                    pygame.draw.rect(windowSurface, BLUE, (x * cellWidth, y * cellHeight, cellWidth, cellHeight))
                pygame.draw.rect(windowSurface, GREEN, (self.apple[0] * cellWidth, self.apple[1] * cellHeight, cellWidth, cellHeight))
                pygame.display.update()
                pygame.time.wait(50)

            direction = agent.getPlay(self)
            tmp = copy.deepcopy(self.snake[-1])
            tmp[2] += 1
            tmp[0] += direction[0]
            tmp[1] += direction[1]

            cmp += 1
            if cmp > 50:
                return score + ((tmp[0] - self.apple[0]) + (tmp[1]-self.apple[1]))


            if (tmp[0] < 0 or tmp[0] >= nbCellsWidth) or tmp[1] < 0 or tmp[1] >= nbCellsHeight:
                return score - ((self.snake[-1][0] - self.apple[0]) + (self.snake[-1][1]-self.apple[1]))

            for x, y, v in self.snake:
                if x == tmp[0] and y == tmp[1]:
                    return score - ((self.snake[-1][0] - self.apple[0]) + (self.snake[-1][1]-self.apple[1]))

            if tmp[2] >= nbCellsHeight * nbCellsWidth:
                return score - ((self.snake[-1][0] - self.apple[0]) + (self.snake[-1][1]-self.apple[1]))

            self.snake.append(tmp)
            score += 1

            if self.apple[0] != tmp[0] or self.apple[1] != tmp[1]:
                for s in range(len(self.snake)):
                    self.snake[s][2] -= 1
            else:
                self.generate_apple()
                score += 100
                cmp = 0

            self.snake = list(filter(lambda x: x[2] > 0, self.snake))

direction = RIGHT

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


class NeuralNet:
    # NN Construct
    def __init__(self, inData, outData, neurones, layersNB, learningRate):
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
                layer.append(
                    [random.uniform(-1, 1) for x in range((self.inData + 1) if i == 0 else (self.neurones + 1))])

            self.weights.append(layer)

        # Out layer
        layer = []
        for j in range(self.outData):
            layer.append([random.uniform(-1, 1) for x in range(self.neurones + 1)])
        self.weights.append(layer)

    def activation(self, value):
        # Sigmoid fix for math range
        if value < -709:
            return 1
        return 1 / (1 + math.exp(-value))

    def cross(self, father, mother):
        for layer in range(len(self.weights)):
            for neurone in range(len(self.weights[layer])):
                for synapse in range(len(self.weights[layer][neurone])):
                    if random.randint(0, 1):
                        self.weights[layer][neurone][synapse] = father.weights[layer][neurone][synapse]
                    else:
                        self.weights[layer][neurone][synapse] = mother.weights[layer][neurone][synapse]

        '''for layer in range(len(self.weights)):
            for neurone in range(len(self.weights[layer])):
                p1 = father.weights
                p2 = mother.weights
                if random.randint(0, 1):
                    p1 = mother.weights
                    p2 = father.weights
                point = random.randint(0, len(self.weights[layer][neurone])-1)
                for synapse in range(len(self.weights[layer][neurone])):
                    if synapse < point:
                        self.weights[layer][neurone][synapse] = p1[layer][neurone][synapse]
                    else:
                        self.weights[layer][neurone][synapse] = p2[layer][neurone][synapse]'''
    def mutate(self, mutationRate):
        for layer in range(len(self.weights)):
            for neurone in range(len(self.weights[layer])):
                for synapse in range(len(self.weights[layer][neurone])):
                    if random.uniform(0, 1) < mutationRate:
                        self.weights[layer][neurone][synapse] *= 1 + ((random.uniform(0,1) - 0.5) * 3 + (random.uniform(0,1) - 0.5))

    def get(self, data):
        # Propagate the input to the output to calculate the result of the neural network
        inData = data + [1]  # bias
        for layer in range(len(self.weights)):
            outData = []
            for neurone in range(len(self.weights[layer])):
                outData.append(self.activation(sum([a * b for a, b in zip(inData, self.weights[layer][neurone])])))
            inData = outData
            if layer != len(self.weights) - 1:
                inData += [1]

        # Return the raw output of the neural network
        return inData


class NeuralNetAgent:
    def __init__(self):
        global nbCellsHeight
        global nbCellsWidth
        self.neural = NeuralNet(nbCellsWidth*nbCellsHeight, 4, 40, 2, 1)

    def getPlay(self, game):
        global nbCellsWidth
        global nbCellsHeight
        inData = [0 for i in range(nbCellsHeight * nbCellsWidth)]
        for x, y, v in game.snake:
            inData[x + y*nbCellsWidth] = v
        inData[game.apple[0] + game.apple[1]*nbCellsWidth] = -1

        #if game.render:
        #    print(inData)

        r = self.neural.get(inData)

        directions = [UP, RIGHT, DOWN, LEFT]
        i = r.index(max(r))
        pygame.event.get()
        return directions[i]

    def mutate(self, mutationRate):
        self.neural.mutate(mutationRate)


    def cross(self, father, mother):
        self.neural.cross(father.neural, mother.neural)


def geneticAlgorithm(nbSpecimen):
    agents = [[NeuralNetAgent(), -1] for i in range(nbSpecimen)]
    for x in range(10000):
        currentSeed = 1
        for a in range(len(agents)):
            r = random.Random()
            r.seed(currentSeed)
            game = Game(False, r)
            agents[a][1] = game.play(agents[a][0])

        agents = sorted(agents, key=lambda x: x[1], reverse=True)
        print("Best score : ", agents[0][1])
        print("Worst score : ", agents[-1][1])
        print(list(map(lambda k: k[1], agents)))
        elites = 300
        elimined = 100
        if agents[0][1] <= 0:
            agents = [[NeuralNetAgent(), -1] for i in range(nbSpecimen)]
            continue

        for i in range(elites, len(agents)-elimined):
            father, mother = random.sample(agents[:elites], 2)
            agents[i][0].cross(father[0], mother[0])
            agents[i][0].mutate(0.1)
            agents[i][1] = -1


        for i in range(len(agents)-elimined, len(agents)):
            agents[i] = [NeuralNetAgent(), -1]

        r = random.Random()
        r.seed(currentSeed)
        game = Game(True, r)
        game.play(agents[0][0])

geneticAlgorithm(1000)



#agent = HumanAgent()
#game = Game(True, 1)

#game.play(agent)