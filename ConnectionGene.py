class ConnectionGene():

    def __init__(self, inNode, outNode, weight, enabled, innovation):
        self.inNode = inNode
        self.outNode = outNode
        self.weight = weight
        self.enabled = enabled
        self.innovation = innovation

    def disable(self):
        self.enabled = False

