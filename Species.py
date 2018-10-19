
class Species():

    def __init__(self,  specicestype, a_genome, mascot, totalAdjustedFitness=29, allowedOffspring = 0):
        if(not isinstance(a_genome, list)):
            self.genomeList = [a_genome]
        else:
            self.genomeList = a_genome
        self.speciesType = specicestype
        self.mascot = mascot
        self.totalAdjustedFitness = totalAdjustedFitness
        self.allowedOffspring = allowedOffspring

    def getNumGenome(self):
        return len(self.genomeList)

    def addGenome(self,genome):
        self.genomeList.append(genome)
