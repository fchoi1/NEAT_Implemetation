class NodeGene():

    types = {'INPUT', 'OUTPUT', 'HIDDEN'}

    def __init__(self, type, id):
        if type not in self.types:
            raise ValueError("Node gene types must be one of %r" % self.types)

        self.layer = type
        self.id = id
