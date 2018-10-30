from genome import genome, ConnectionGene, NodeGene
from Species import Species
from copy import deepcopy
import gym
import numpy as np
import pickle
import os.path
import time
import curses

class population():

    SpeciesList = []
    InnovationList = []
    GlobalSpecies = 0
    GenomeID = 0

    def __init__(self, size,mutateWeightRate=0.7,mutateNodeRate=0.1, mutateConnectionRate=0.1, removeConnectionRate=0.03, c1=1, c2=1, c3=1, delta=2):
        self.mutateWeightRate = mutateWeightRate
        self.mutateNodeRate = mutateNodeRate
        self.mutateConnectionRate = mutateConnectionRate
        self.removeConnectionRate = removeConnectionRate
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.delta = delta
        self.populationSize = size

    def make_env(self, game_env):
        self.env = gym.make(game_env)
        self.env.reset()

    def closeGame(self):
        self.env.close()

    def findSpecies(self,genome1):
        for species in self.SpeciesList:
            if(genome1 in species.genomeList):
                return species.speciesType
        print("Genome doesn't belong to any species")

    def drawPop(self, gen=0):

        for s in self.SpeciesList:
            for g in s.genomeList:
                drawnet = 'Gen_' + str(gen) + '/Species_' + str(s.speciesType) + '/genome_' + str(g.id)
                g.draw_net(filename=drawnet)

    def printGenome(self, genome=None):

        if (genome):
            print('\nid ', genome.id, ' fitness ', genome.fitness)
            print([(i.inNode, i.outNode) for i in genome.ConnectionList])
            print([(i.id) for i in genome.NodeList], '\n')
        else:
            for s in self.SpeciesList:
                for g in s.genomeList:
                    print('id ', g.id, ' fitness ', g.fitness)
                    print([(i.inNode, i.outNode, i.enabled) for i in g.ConnectionList])
                    print([(i.id) for i in g.NodeList])

    def saveState(self, generation, innovationFile='Innovation.obj', generationFile='Generation.txt', PopulationFile='Population.obj'):
        innovationFile = 'Gen_' + str(generation) + '/'  + innovationFile
        generationFile = 'Gen_' + str(generation) + '/'  + generationFile
        PopulationFile ='Gen_' + str(generation) + '/'  + PopulationFile

        os.makedirs(os.path.dirname(innovationFile), exist_ok=True)
        os.makedirs(os.path.dirname(generationFile), exist_ok=True)
        os.makedirs(os.path.dirname(PopulationFile), exist_ok=True)
        # write_innovation = open(innovationFile, 'wb')
        # write_generation = open(generationFile, 'w')
        # write_population = open(PopulationFile, 'wb')

        with open(innovationFile, "wb") as writeInnovation, open(generationFile, "w") as writeGeneration, open(PopulationFile, 'wb') as writePopulation:
            pickle.dump(self.InnovationList, writeInnovation)
            pickle.dump(self.SpeciesList, writePopulation)
            a_string = str(generation) + ':' + str(self.GlobalSpecies) + ':' + str(self.GenomeID)
            writeGeneration.write(a_string)
            writeGeneration.close()

    def loadState(self, innovationFile='Innovation.obj', generationFile='Generation.txt', PopulationFile='Population.obj'):
        initial = False

        print('Loading Innovation File: ')
        try:
            read_innovation = open(innovationFile, 'rb')

            if (os.stat(innovationFile).st_size > 0):
                self.InnovationList = pickle.load(read_innovation)
            else:
                print('Empty Innovation File')
                initial = True
        except FileNotFoundError:
            print('Innovation File does not Exist')
            initial = True

        print('Loading Population File: ')
        try:
            read_population = open(PopulationFile, 'rb')
            if (os.stat(PopulationFile).st_size > 0):
                self.SpeciesList = pickle.load(read_population)
            else:
                print('Empty Population File')
                initial = True
        except FileNotFoundError:
            print('Population File does not Exist')
            initial = True

            print('Loading Generation File: ')
        try:
            write_generation = open(generationFile, 'r')
            if (os.stat(generationFile).st_size > 0):
                a_string = write_generation.read().split(':')
                gen = int(a_string[0])
                self.GlobalSpecies = int(a_string[1])
                self.GenomeID = int(a_string[2])
            else:
                print('Empty Generation File')
                gen = 0
                initial = True
        except FileNotFoundError:
            print('Generation File does not Exist')
            gen = 0
            initial = True

        return gen, initial

    def countGenes(self, genome1, genome2):
        excess = 0
        disjoint = 0
        weights = 0
        numGenes = 0
        innovation1 = []
        innovation2 = []

        for connection1 in genome1.ConnectionList:
            innovation1.append(connection1.innovation)
            for connection2 in genome2.ConnectionList:
                if (connection1.innovation == connection2.innovation):
                    weights =+ np.mean([connection1.weight,connection2.weight])
                    break
        for connection2 in genome2.ConnectionList:
            innovation2.append(connection2.innovation)

        if(len(genome1.ConnectionList) < 40 and len(genome2.ConnectionList) < 40):
            numGenes = 1
        elif(len(genome1.ConnectionList) >= len(genome2.ConnectionList)):
            numGenes = len(genome1.ConnectionList)
        else:
            numGenes = len(genome2.ConnectionList)

        innovation1 = sorted(innovation1)
        innovation2 = sorted(innovation2)
        max1 = innovation1[len(innovation1)-1]
        max2 = innovation2[len(innovation2)-1]
        diff1 = list(set(innovation1) - set(innovation2))
        diff2 = list(set(innovation2) - set(innovation1))

        excess += len([i for i in diff1 if i > max2])
        excess += len([i for i in diff2 if i > max1])
        disjoint = len(diff1) + len(diff2) - excess

        return excess, disjoint, weights, numGenes

    def putSpecies(self, genome1, minDistance=None):
        #Constants N are 1 for smaller populations of 20
        if(minDistance == None):
            minDistance = self.delta
        sameSpecies = False

        for a_species in self.SpeciesList:
            excess, disjoint, weights, numGenes = self.countGenes(genome1, a_species.mascot)

            minDelta = (self.c1 * excess + self.c2 * disjoint)/numGenes + self.c3 * weights
            #print('ID: ', genome1.id,  ' compared to: ', a_species.mascot.id, ' ', abs(minDelta), ' E: ', excess, 'D:', disjoint, ' w: ', weights,' N: ', numGenes)
            if (abs(minDelta) < minDistance):
                a_species.addGenome(genome1)
                #print(f'The genome {genome1.id} belongs to species: {a_species.speciesType}')
                sameSpecies = True
                break

        if(not sameSpecies):
            newSpecies = Species(self.GlobalSpecies+1,genome1,genome1)
            print(f'Adding new species: {newSpecies.speciesType} ID: {genome1.id}')
            self.SpeciesList.append(newSpecies)
            self.GlobalSpecies += 1

    def getFitness(self, a_genome, renderGame=False):
        initial_score = 50
        done = False
        ts = 0
        observations = []
        state = []
        choices = []
        score = initial_score
        phenotype = a_genome.buildNet()
        self.make_env('Pong-ramDeterministic-v4')
        #stdscr = curses.initscr()

        try:
            while (not done or ts > 30000):

                if(renderGame):
                    time.sleep(0.01)
                    self.env.render()

                if (len(observations) == 0):
                    #action = self.env.action_space.sample()
                    action = 1
                else:
                    outComes = np.array(a_genome.active(phenotype, state))
                    action = np.argmax(outComes) + 1
                    '''
                    if(r < outComes[0]):
                        action = 1
                    elif(r < (outComes[0] + outComes [1])):
                        action = 2
                    else:
                         action = 3
                    '''
                    choices.append(action)
                # ram values
                # 49 ball x position 32-CD
                # 54 ball y position (player's was 26-cB)
                # 56 ball y velocity -5 to 5
                # 58 ball x velocity -5 to 5
                # 60 player y 26-CB
                observations, reward, done, info = self.env.step(action)
                state = np.array([observations[i] for i in [49, 54, 56, 58, 60]], dtype='int32')

                if (state[2] > 5):
                    state[2] = int(float(state[2]) - 256.0)
                if (state[3] > 5):
                    state[3] -= 256
                '''
                for i in range(round(len(observations)/16)):
                    data = observations[i*16:(i+1)*16]
                    stdscr.addstr(0,0,"The Ram Data and Score: " + str(score)  + " Action: " + str(action) + " Outcomes " + str(outComes))
                    stdscr.addstr(i+1,0, " ".join(str(data)) )

                stdscr.addstr(11,0,str(state))
                stdscr.addstr(12,0,str(state[3]))
                stdscr.addstr(13,0,str(state[2]))

                stdscr.refresh()
                time.sleep(0.1)
                '''
                score += reward
                #if (ts == 0): print(observations)
                ts += 1

        finally: None
        digits = 5

        self.closeGame()

        if(score > initial_score):
            fitness = float(score + 1/(ts/100.0))
        else:
            fitness = float(score + ts/(10**digits))
        #print(fitness)
        #print(collections.Counter(choices))
        #print('Nothing-1:{%.3f}  Up-2:{%.3f}. Down-3: {%.3f}'.format(round(choices.count(1) / len(choices),3), round(choices.count(2) / len(choices),3),round(choices.count(3)/ len(choices),3)))
        return fitness

    def evaluateGenomes(self, numGames=1):
        highestFitness = 0
        lowestFitness = 100
        bestGenomeList = []
        best_genome = []
        oneInEach = []
        for a_species in self.SpeciesList:
            numGenomes =  len(a_species.genomeList)
            highestSpeciesFitness = 0
            worst_genome = None
            for each_genome in a_species.genomeList:

                genomeFitness = []
                for games in range(numGames):
                    genomeFitness.append(self.getFitness(each_genome))

                each_genome.fitness = np.average(genomeFitness)
                shared_fitness = each_genome.fitness / numGenomes
                a_species.totalAdjustedFitness += shared_fitness
                #print('Species type: ', a_species.speciesType,  'genome id', each_genome.id, ' Genome Score: ', each_genome.fitness, '  score ', a_species.totalAdjustedFitness)
                bestGenomeList.append(each_genome)

                if (each_genome.fitness > highestSpeciesFitness):
                    best_genome = each_genome
                    highestSpeciesFitness = each_genome.fitness

                if(each_genome.fitness > highestFitness):
                    highestFitness = each_genome.fitness

                if(each_genome.fitness < lowestFitness and len(a_species.genomeList) > 2):
                    lowestFitness = each_genome.fitness
                    worst_genome =  each_genome

            if(worst_genome):
                a_species.genomeList.remove(worst_genome)

            oneInEach.append(best_genome)

        bestGenomeList.sort(key=lambda x: x.fitness, reverse=True)

        if(len(bestGenomeList) > 20):
            keepGenomes = bestGenomeList[0:20]
        else:
            keepGenomes = bestGenomeList
        [keepGenomes.append(i) for i in oneInEach if i not in keepGenomes]

        return keepGenomes, highestFitness

    def breedPopulation(self, nextGenomes, mutationRate=None, addNodeRate=None, addConnectionRate=None, removeConnectionRate=None):
        if(mutationRate == None):
            mutationRate = self.mutateWeightRate
        if(addNodeRate == None):
            addNodeRate = self.mutateNodeRate
        if(addConnectionRate == None):
            addConnectionRate = self.mutateConnectionRate
        if(removeConnectionRate == None):
            removeConnectionRate = self.removeConnectionRate

        self.setAllowedOffspring(len(nextGenomes))

        for a_species in self.SpeciesList:
            offspring = 0
            while(offspring < a_species.allowedOffspring):

                if (len(a_species.genomeList) < 2):
                    childGenome = deepcopy(a_species.genomeList[0])
                else:
                    parent1 = self.getBiasedGenome(a_species)
                    parent2 = self.getBiasedGenome(a_species,parent1)

                    if(parent1.fitness > parent2.fitness):
                        childGenome = self.crossGenomes(parent1, parent2)
                    elif(parent2.fitness > parent1.fitness):
                        childGenome = self.crossGenomes(parent2, parent1)
                    else:
                        childGenome = self.crossGenomes(parent1, parent2, True)
                offspring += 1

                if (np.random.random() < mutationRate):
                    childGenome.mutateWeights()

                if (np.random.random() < addNodeRate):
                    self.InnovationList = childGenome.mutateAddNode(self.InnovationList)

                if (np.random.random() < addConnectionRate):
                    self.InnovationList = childGenome.mutateAddConnection(self.InnovationList)

                if (np.random.random() < removeConnectionRate):
                    self.InnovationList = childGenome.mutateAddConnection(self.InnovationList)

                nextGenomes.append(childGenome)

        self.makeSpeciesMascot()
        self.reset()

        for a_genome in nextGenomes:
            self.putSpecies(a_genome)

    def reset(self):
        count = 0
        removed = []
        for a_species in self.SpeciesList:
            if(len(a_species.genomeList) == 0):
                removed.append(a_species.speciesType)
                self.SpeciesList.remove(a_species)
                count += 1
            a_species.totalAdjustedFitness = 0
            a_species.genomeList = []

        print(f'Removed {count} species. Species type removed: ', *removed)

    def makeSpeciesMascot(self):
        for a_species in self.SpeciesList:
            if(len(a_species.genomeList) > 0):
                a_species.mascot = a_species.genomeList[np.random.randint(0,len(a_species.genomeList))]

    def splitNum(self, NumList):
        the_sum = sum([i[0] for i in NumList])
        floor_sum = sum(np.floor([i[0] for i in NumList]))
        sorted_Num = deepcopy(sorted(NumList, key=lambda l:(l[0] - np.floor(l[0])), reverse=True))
        count = 0
        for a_Num in sorted_Num:
            if(count < int(round(the_sum - floor_sum))):
                a_Num[0] = int(np.ceil(a_Num[0]))
            else:
                a_Num[0] = int(np.floor(a_Num[0]))
            count += 1

        sorted_Num.sort(key=lambda l:l[1])
        return sorted_Num

    def setAllowedOffspring(self, keepGenomes):
        remainingSize = self.populationSize -  keepGenomes
        numAllowed = []

        total_Score = sum([i.totalAdjustedFitness for i in self.SpeciesList])
        if(total_Score == 0):
            for a_species in self.SpeciesList:
                a_species.totalAdjustedFitness = 29
                total_Score = sum([i.totalAdjustedFitness for i in self.SpeciesList])

        for a_species in self.SpeciesList:
            numAllowed.append([remainingSize * a_species.totalAdjustedFitness/total_Score, a_species.speciesType])

        sorted_numAllowed = self.splitNum(numAllowed)
        self.SpeciesList = sorted(self.SpeciesList, key = lambda l:l.speciesType)

        #print([i.speciesType for i in self.SpeciesList])
        #print(sorted_numAllowed)

        for a_species, rank in zip(self.SpeciesList, sorted_numAllowed):
            a_species.allowedOffspring = rank[0]

            print('Species: ', a_species.speciesType, ' Allowed: ', a_species.allowedOffspring,  ' Species score: ', a_species.totalAdjustedFitness)

    def getBiasedGenome(self,someSpecies, genome2avoid=None):
        totalFitness = 0
        counter = 0
        tempList = someSpecies.genomeList.copy()

        if(genome2avoid):
            try:
                tempList.remove(genome2avoid)
            except:
                print('not in list')

        for each_genome in tempList:
            totalFitness += abs(each_genome.fitness)

        threshold = np.random.random() * totalFitness
        for each_genome in tempList:
            counter += abs(each_genome.fitness)
            if(counter >= threshold):
                return each_genome

    def crossGenomes(self, genome1, genome2, equal=False):
        #Parent 1 is always more fit
        parent1 = deepcopy(genome1.ConnectionList)
        parent2 = deepcopy(genome2.ConnectionList)

        tempParent2 = parent2.copy()
        childNodeList = deepcopy(genome1.NodeList)

        for a_node in genome2.NodeList:
            if(a_node.id not in [i.id for i in childNodeList]):
                childNodeList.append(a_node)

        childConnectionList = []

        for connection1 in parent1:
            disjoint = True
            for connection2 in parent2:
                if(connection1.innovation == connection2.innovation):
                    childConnectionList.append(connection1) if np.random.randint(0, 2) else childConnectionList.append(connection2)
                    try:
                        tempParent2.remove(connection2)
                    except ValueError:
                    #    None
                        print(f'Tried to remove innovation: {connection2.innovation} with connection {connection2.inNode} to {connection2.outNode} but not in list: {[(i.innovation, i.inNode, i.outNode) for i in tempParent2]}')

                    disjoint = False
                    break
            if(disjoint and equal and np.random.randint(0, 2)):
                childConnectionList.append(connection1)
            if(disjoint and not equal):
                childConnectionList.append(connection1)

        #print([(i.inNode, i.outNode) for i in parent2])
            #print('\nParent: ')
            #print([(i.inNode, i.outNode) for i in parent1])
            #print([(i.inNode, i.outNode) for i in parent2])
            #print('\n CHild')
            #print([(i.inNode, i.outNode) for i in childConnectionList])

        if(equal):
            [childConnectionList.append(i) for i in tempParent2 if np.random.randint(0, 2)]

        self.GenomeID += 1

        for a_node in childNodeList:
            if ((a_node.id not in [i.inNode for i in childConnectionList]) and (a_node.id not in [i.outNode for i in childConnectionList])):
                childNodeList.remove(a_node)

        childGenome = genome(childConnectionList,childNodeList,29, self.GenomeID)

        return childGenome

    def renderGames(self, genomeList):
         for i in genomeList:
             score = self.getFitness(i, True)

             print(f'Species: {self.findSpecies(i)} Genome Id: {i.id} with score: {score}  ')

    def addInitialGenomes(self,size):

        NodeList = []
        newGenomes = []

        newNode1 = NodeGene('INPUT', 0)
        newNode2 = NodeGene('INPUT', 1)
        newNode3 = NodeGene('INPUT', 2)
        newNode4 = NodeGene('INPUT', 3)
        newNode5 = NodeGene('INPUT', 4)
        outputNode1 = NodeGene('OUTPUT', 5)
        outputNode2 = NodeGene('OUTPUT', 6)
        outputNode3 = NodeGene('OUTPUT', 7)

        NodeList.extend([newNode1,newNode2,newNode3,newNode4,newNode5, outputNode1, outputNode2, outputNode3])

        for i in range(2):
            self.GenomeID += 1
            parent = genome([], NodeList,29,self.GenomeID)
            for i in range(5):
                self.InnovationList = parent.addConnection(i,5, self.InnovationList)
                self.InnovationList = parent.addConnection(i,6, self.InnovationList)
                self.InnovationList = parent.addConnection(i,7, self.InnovationList)
            self.putSpecies(parent)
            newGenomes.append(parent)
        self.breedPopulation(newGenomes, 0.7, 0.3, 0.3, 0.1)



if __name__ == '__main__':
    renderNum = 5
    Pop = population(100, c1=0.5, c2=0.5, c3=0.4, delta=3.0)
    gen, initial = Pop.loadState()
    if(initial):
        Pop.addInitialGenomes(100)

    for i in range(1000):

        print('\n\n\nEvaluating Genomes ############################################################################# \n')

        bestGenomesList,  highest = Pop.evaluateGenomes(5)
        bestSpecies = Pop.findSpecies(bestGenomesList [0])
        TypesinGen = [i.speciesType for i in Pop.SpeciesList]

        for each_genome in bestGenomesList[:10]:
            print(f'Genome ID {each_genome.id}    Score: {each_genome.fitness}    Species Type: {Pop.findSpecies(each_genome)}')

        print('\nBreeding New Pop ############################################################################# \n')

        Pop.breedPopulation(bestGenomesList)

        print('\nAfter Evalutations #############################################################################\n')

        for s in Pop.SpeciesList:
            print('Species Number: ', s.speciesType, ' Ammount: ', len(s.genomeList))

        print(f'\nGen: {gen + i}   Number of Species {len(Pop.SpeciesList)}    Types of Species: {Pop.GlobalSpecies}   Innovation: {max([i[0] for i in Pop.InnovationList])}')
        print('Current Species Type in Gen: ', *TypesinGen)
        print(f'Current best genome fitness:  {bestGenomesList[0].fitness}')

        if(i%5 == 0):

            if (len(bestGenomesList) > renderNum):
                index = renderNum
            else:
                index = len(bestGenomesList)

            Pop.renderGames(bestGenomesList[:index])
            Pop.saveState(gen + i)
            Pop.drawPop(gen + i)


