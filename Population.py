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

    #GenomeList = []
    SpeciesList = []
    CurrentInnovationList = []
    GlobalInnovation = 0
    GlobalSpecies = 0
    GenomeID = 0

    def __init__(self, size,mutateWeightRate=0.9,mutateNodeRate=0.05, mutateConnectionRate=0.05, c1=1, c2=1, c3=1, delta=2, initial=True):
        self.mutateWeightRate = mutateWeightRate
        self.mutateNodeRate = mutateNodeRate
        self.mutateConnectionRate = mutateConnectionRate
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.delta = delta
        self.populationSize = size
        if(initial):
            self.addInitialGenomes(size)

    def make_env(self, game_env):
        self.env = gym.make(game_env)
        self.env.reset()

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

        if(len(genome1.ConnectionList) < 50 and len(genome2.ConnectionList) < 50):
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

    def makeSpeciesMascot(self):
        for a_species in self.SpeciesList:
            if(len(a_species.genomeList) > 0):
                a_species.mascot = a_species.genomeList[np.random.randint(0,len(a_species.genomeList))]

    def putSpecies(self, genome1, minDistance=None):
        #Constants N are 1 for smaller populations of 20
        if(minDistance == None):
            minDistance = self.delta
        sameSpecies = False

        for a_species in self.SpeciesList:
            excess, disjoint, weights, numGenes = self.countGenes(genome1, a_species.mascot)
            minDelta = (self.c1 * excess + self.c2 * disjoint) + self.c3 * weights

            print(minDelta, ' E: ', excess, 'D:', disjoint, ' w: ', weights,' N: ', numGenes, ' Coefficent ', self.c1, self.c2, self.c3)

            if (minDelta < minDistance):
                a_species.addGenome(genome1)
                #print(f'The genome belongs to species: {a_species.speciesType}')
                sameSpecies = True
                break

        if(not sameSpecies):
            totalAdjustedfitness = genome1.fitness
            newSpecies = Species(self.GlobalSpecies+1,genome1,genome1, totalAdjustedfitness)
            print(f'Adding new species: {newSpecies.speciesType}')
            self.SpeciesList.append(newSpecies)
            self.GlobalSpecies += 1

    def findSpecies(self,genome1):
        for species in self.SpeciesList:
            if(genome1 in species.genomeList):
                return species.speciesType
        print("Genome doesn't belong to any species")

    def getFitness(self, a_genome, renderGame=False):
        self.make_env('Pong-ram-v0')
        done = False
        ts = 0
        observations = []
        state = []
        choices = []
        outComes = []
        score = 50
        phenotype = a_genome.buildNet()

        #stdscr = curses.initscr()
        #curses.noecho()
        #curses.cbreak()
        try:
            while (not done):

                if(renderGame):  self.env.render()

                if (len(observations) == 0):
                    #action = self.env.action_space.sample()
                    action = 2
                else:
                    r = np.random.random()
                    outComes = np.array(a_genome.active(phenotype, state))
                    action = np.argmax(outComes) + 1
                    action = np.random.randint(1, 4)
                    '''
                    if(r < outComes[0]):
                        action = 1
                    elif(r < (outComes[0] + outComes [1])):
                        action = 2
                    else:
                         action = 3
                    '''
                    choices.append(action)
                    #print(outComes, action)

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
            #curses.echo()
            #curses.cbreak()
            #curses.endwin()
        #print(score)
        digits = len(str(ts))
        if(digits == 3):
            digits = 4

        self.closeGame()

        fitness = float(score + ts/(10**digits))
        #print(fitness)
        #print(collections.Counter(choices))
        #print('Nothing-1:{%.3f}  Up-2:{%.3f}. Down-3: {%.3f}'.format(round(choices.count(1) / len(choices),3), round(choices.count(2) / len(choices),3),round(choices.count(3)/ len(choices),3)))
        return fitness

    def evaluateGenomes(self):
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

                each_genome.fitness = self.getFitness(each_genome)

                shared_fitness = each_genome.fitness / numGenomes
                a_species.totalAdjustedFitness += shared_fitness

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

        if(len(bestGenomeList) > 10):
            keepGenomes = bestGenomeList[0:10]
        else:
            keepGenomes = bestGenomeList
        [keepGenomes.append(i) for i in oneInEach if i not in keepGenomes]


        return keepGenomes, highestFitness

    def renderSomeGames(self, a_genome):
        self.env.reset()
        done = False
        ts = 0.0
        observations = []
        score = 50
        phenotype = a_genome.buildNet()

        while (not done):
            self.env.render()
            ts += 1
            if (len(observations) == 0):
                action = self.env.action_space.sample()
            else:
                observations = list(observations[:80])
                observations.extend(list(observations[-7:]))
                r = np.random.random()
                outComes = np.array(a_genome.activate(phenotype, observations))
                action = np.argmax(outComes)

                '''
                r = np.random.random()
                if (r < outComes[0]):
                    action = 1
                elif (r < (outComes[0] + outComes[1])):
                    action = 2
                else:
                    action = 3
                '''
            observations, reward, done, info = self.env.step(action)
            score += reward
        #digits = len(str(ts))
        #return float(score + ts/(10**digits))
        self.closeGame()
        return score, ts

    def breedPopulation(self, nextGenomes, mutationRate=None, addNodeRate=None, addConnectionRate=None):
        if(mutationRate == None):
            mutationRate = self.mutateWeightRate
        if(addNodeRate == None):
            addNodeRate = self.mutateNodeRate
        if(addConnectionRate == None):
            addConnectionRate = self.mutateConnectionRate

        self.setAllowedOffspring(len(nextGenomes))

        #while(len(nextGenomes) < self.populationSize):

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
                nextGenomes.append(childGenome)

                print('\nParent: ')
                print([(i.inNode, i.outNode) for i in parent1.ConnectionList])
                print([(i.inNode, i.outNode) for i in parent2.ConnectionList])
                print('\n CHild')
                print([(i.inNode, i.outNode) for i in childGenome.ConnectionList])

        self.makeSpeciesMascot()
        self.reset()

        for a_genome in nextGenomes:

            if(np.random.random() < mutationRate):
                a_genome.mutateWeights()

            if(np.random.random() < addNodeRate):
                self.CurrentInnovationList, self.GlobalInnovation = a_genome.mutateAddNode(self.GlobalInnovation, self.CurrentInnovationList)

            if(np.random.random() < addConnectionRate):
                self.CurrentInnovationList, self.GlobalInnovation = a_genome.mutateAddConnection(self.GlobalInnovation, self.CurrentInnovationList)

            self.putSpecies(a_genome)


    def setAllowedOffspring(self, keepGenomes):
        remainingSize = self.populationSize -  keepGenomes
        totalScore = 0

        for a_species in self.SpeciesList:
            totalScore += a_species.totalAdjustedFitness

        for a_species in self.SpeciesList:
            a_species.allowedOffspring = int(round(remainingSize * a_species.totalAdjustedFitness/totalScore))

    ###NOT USED

    def getBiasedSpecies(self, species2avoid=None):
        totalWeight = 0
        counter = 0
        temp = deepcopy(self.SpeciesList)

        if species2avoid:
            temp.remove(species2avoid)

        for a_species in temp:
            totalWeight += abs(a_species.totalAdjustedFitness)

        threshold = np.random.random() * totalWeight
        for a_species in temp:
            counter += abs(a_species.totalAdjustedFitness)
            if(counter >= threshold):
                return a_species

    ######

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

    def reset(self):
        count = 0
        removed = []
        for a_species in self.SpeciesList:
            a_species.totalAdjustedFitness = 0
            a_species.genomeList = []

            if(len(a_species.genomeList) == 0):
                removed.append(a_species.speciesType)
                self.SpeciesList.remove(a_species)
                count += 1

        print(f'Removed {count} species. Species type removed: ', *removed)

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
                    #try:
                    tempParent2.remove(connection2)
                    #except ValueError:
                    #    None
                        #print('Value already removed')

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

    def closeGame(self):
        self.env.close()

    def drawPop(self, gen=0):

        for s in self.SpeciesList:
            for g in s.genomeList:
                drawnet = 'Gen_' + str(gen) + '/Species_' + str(g.id)
                g.draw_net(filename=drawnet)

    def printGenome(self, genome=None):

        if(genome):
            print('\nid ', genome.id, ' fitness ', genome.fitness)
            print([(i.inNode, i.outNode) for i in genome.ConnectionList])
            print([(i.id) for i in genome.NodeList],'\n')
        else:
            for s in self.SpeciesList:
                for g in s.genomeList:
                    print('id ',g.id, ' fitness ', g.fitness)
                    print([(i.inNode, i.outNode, i.enabled) for i in g.ConnectionList])
                    print([(i.id) for i in g.NodeList])


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
            parent = genome([], NodeList,29,self.GenomeID)
            self.GenomeID += 1
            for i in range(5):
                #outputNode = np.random.randint(87, 91)
                self.CurrentInnovationList, self.GlobalInnovation = parent.addConnection(i,5, self.GlobalInnovation, self.CurrentInnovationList)
                self.CurrentInnovationList, self.GlobalInnovation = parent.addConnection(i,6, self.GlobalInnovation, self.CurrentInnovationList)
                self.CurrentInnovationList, self.GlobalInnovation = parent.addConnection(i,7, self.GlobalInnovation, self.CurrentInnovationList)

            self.putSpecies(parent)
            newGenomes.append(parent)

        #print('Breed Pop')
        self.printGenome()


        self.breedPopulation(newGenomes, 0.2, 0.5, 0.8)

if __name__ == '__main__':
    Pop_size = 100
    object_file = 'Pong_Population.obj'
    gen_filename = 'genNum.txt'

    if (os.path.isfile(object_file) and not os.stat(object_file).st_size == 0 ):
        Pop = population(50,c1=0.5,c2=1,c3=0.6,delta=3.5,initial=False)
        Pop_file = open(object_file,'rb')
        Pop.SpeciesList = pickle.load(Pop_file)

        if(os.path.isfile(gen_filename) and not os.stat(gen_filename).st_size == 0 ):
            gen_file = open(gen_filename,'r')
            a_string = (gen_file.read()).split(':'
                                               '')
            #print(a_string)
            gen = int(a_string[0])
            Pop.GlobalInnovation = int(a_string[1])
            Pop.GlobalSpecies = int(a_string[2])
            Pop.GenomeID = int(a_string[3])
            Pop.CurrentInnovationList = (a_string[4])
            '''
            print(Pop.GlobalInnovation)
            print(Pop.GlobalSpecies)
            print(Pop.GenomeID)
            print(Pop.CurrentInnovationList)
            '''

    else:
        Pop = population(50, c1=0.5, c2=1, c3=0.6, delta=3.5, initial=True)
        gen = 0

    bestSpeciesinGen = []
    highestScoreinGen = []
    SpeciesinGen = []

    for i in range(1000):
        bestGenomesList,  highest = Pop.evaluateGenomes()
        bestSpecies = Pop.findSpecies(bestGenomesList [0])
        check_pop = sum([len(i.genomeList) for i in Pop.SpeciesList])
        TypesinGen = [i.speciesType for i in Pop.SpeciesList]

        for each_genome in bestGenomesList:
            print(f'Genome ID {each_genome.id} Fitness score: {each_genome.fitness} Species Type: {Pop.findSpecies(each_genome)}')
        print(f'Gen: {gen + i}, Number of Species {len(Pop.SpeciesList)}, Types of Species: {Pop.GlobalSpecies} with Innovation: {Pop.GlobalInnovation}')
        print('Current Species Type in Gen: ', *TypesinGen)
        print(f'Best Species: {bestSpecies} Highest Fitness: {highest} Best Genome {bestGenomesList[0].id}')

        SpeciesinGen.append(Pop.SpeciesList)
        Pop.breedPopulation(bestGenomesList)
        #Pop.printGenome()



        if(i%10 == 0):

            Pop.drawPop(gen + i)
            file_Pop = open(object_file, 'wb')
            gen_file = open(gen_filename, 'w')
            pickle.dump(Pop.SpeciesList, file_Pop)
            a_string = str(gen + i) + ':' + str(Pop.GlobalInnovation) + ':' + str(Pop.GlobalSpecies) + ':' + str(Pop.GenomeID) + ':' + str(Pop.CurrentInnovationList)
            gen_file.write(a_string)
            gen_file.close()

            if(len(bestGenomesList) > 10):
                index = 10
            else:
                index = len(bestGenomesList)
            counter = 0
            for i in bestGenomesList[0:index]:
                drawnet = 'Species ' + str(counter)
                #i.draw_net(filename=drawnet)
                counter += 1
                genome_score = Pop.getFitness(i, True)
                print(f'Genome Number {counter} / {index} Score: {genome_score}  ID: {i.id} Species {Pop.findSpecies(i)}')


'''

Pop = population(5)
the_species = Pop.findSpecies(Pop.SpeciesList[1].genomeList[0])

'''

#GlobalPop.addInitialGenomes()
#childGenome = GlobalPop.crossGenomes(GlobalPop.SpeciesList[0].genome[0],GlobalPop.GenomeList[1], True)


#Testing Crossover Function
#print([[i.innovation, i.inNode, i.outNode, i.enabled] for i in childGenome.ConnectionList])

#GlobalPop.CurrentInnovationList, GlobalPop.GlobalInnovation = GlobalPop.GenomeList[0].mutateAddConnection(GlobalPop.GlobalInnovation, GlobalPop.CurrentInnovationList)
#GlobalPop.CurrentInnovationList, GlobalPop.GlobalInnovation = GlobalPop.GenomeList[0].mutateAddNode(GlobalPop.GlobalInnovation, GlobalPop.CurrentInnovationList)

'''
print("Genome 1 Connections")
print([[i.innovation, i.inNode, i.outNode, i.enabled] for i in GlobalPop.SpeciesList[0].genomeList[0].ConnectionList])
print('Genome 1 Nodes')
print([[i.layer, i.id] for i in GlobalPop.SpeciesList[0].genomeList[0].NodeList])
'''


#GlobalPop.putSpecies(GlobalPop.GenomeList[0],GlobalPop.GenomeList[1])
#print(f'excess {excess}, disjoint: {disjoint}, weights {weights}')