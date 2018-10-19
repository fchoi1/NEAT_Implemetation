import numpy as np
from custom_feed_forward import FeedForwardNetwork
import copy
import warnings

import graphviz
import matplotlib.pyplot as plt

class genome():

    def __init__(self, ConnectionList, NodeList, fitness=29, id=0):
        self.ConnectionList = ConnectionList
        self.NodeList = NodeList
        self.fitness = fitness
        self.id=id

    def addConnection(self, in_node, out_node, innovationNumber=0,  currentInnovationList=[]):
        innovationList = currentInnovationList.copy()
        innovationNum = innovationNumber
        assignInnovation = 0
        pre_innovation = False
        w = np.random.uniform(-1,1)

        for innovationConnections in innovationList:
            assignInnovation += 1
            if (in_node == innovationConnections[0] and out_node == innovationConnections[1]):
                pre_innovation = True
                break

        if (not pre_innovation):
            innovationNum += 1
            innovationList.append([in_node, out_node])
            assignInnovation = innovationNum

        newConnection = ConnectionGene(inNode=in_node, outNode=out_node, weight=w, enabled=True, innovation=assignInnovation)
        self.ConnectionList.append(newConnection)

        return innovationList, innovationNum

    def mutateAddConnection(self, innovationNumber=0, currentInnovationList=[]):
        n_1 = np.random.randint(0, len(self.NodeList))
        n_2 = np.random.randint(0, len(self.NodeList))
        w = np.random.uniform(-1,1)

        innovationNum = innovationNumber
        innovationList = currentInnovationList

        reversed = False
        pre_innovation = False

        while ((n_1 in [0, 1, 2, 3, 4] and n_2 in [0, 1, 2, 3, 4]) or (n_1 in [5,6,7] and n_2 in [5,6,7]) or (n_1 == n_2)):
            n_1 = np.random.randint(0, len(self.NodeList))
            n_2 = np.random.randint(0, len(self.NodeList))

        #print(self.NodeList[n_1].layer, self.NodeList[n_2].layer)
        #if ((self.NodeList[n_1].layer == 'HIDDEN' and self.NodeList[n_2] == 'INPUT') or (self.NodeList[n_1].layer == 'OUTPUT' and self.NodeList[n_2] == 'HIDDEN') or (self.NodeList[n_1].layer =='OUTPUT' and self.NodeList[n_2] == 'INPUT')):
        if ((n_1 not in [0, 1, 2, 3, 4, 5, 6, 7] and n_2 in [0, 1, 2, 3, 4]) or (n_1 in [5, 6, 7] and n_2 not in [0, 1, 2, 3, 4, 5, 6, 7]) or (n_1 in [5, 6, 7]  and n_2 in [0, 1, 2, 3, 4])):
            #print('\nNeeds to reverse')
            reversed = True

        if (reversed):
            temp = n_1
            # switching node numbers
            n_1 = n_2
            n_2 = temp

        for connection in self.ConnectionList:
            if (connection.inNode == n_1 and connection.outNode == n_2):
                #print('Connection Exists')
                #isConnection = True
                return innovationList, innovationNum

        for innovationConnections in innovationList:
            if(n_1 == innovationConnections[0] and n_2 == innovationConnections[1]):
                pre_innovation = True
                break
        if(not pre_innovation):
            innovationNum += 1
            innovationList.append([n_1,n_2])


        newConnectionGene = ConnectionGene(inNode=n_1, outNode=n_2, weight=w, enabled=True, innovation=innovationNum)
        self.ConnectionList.append(newConnectionGene)
        #print('Added Connection: ', n_1, ' ', n_2, ' to ID: ', self.id)
        return innovationList, innovationNum

    def mutateAddNode(self, innovationNumber=0, currentInnovationList=[]):

        n = np.random.randint(0, len(self.ConnectionList))
        self.ConnectionList[n].disable()
        #print('Disabled connection', (self.ConnectionList[n].inNode, self.ConnectionList[n].outNode), ' from ID ', self.id )

        in_Node= self.ConnectionList[n].inNode
        out_Node = self.ConnectionList[n].outNode
        w = self.ConnectionList[n].weight

        innovationNum = innovationNumber
        innovationList = currentInnovationList

        newID = len(self.NodeList)
        newNode = NodeGene(type='HIDDEN',id=newID)

        innovationNum += 1
        newConnectionGene1 = ConnectionGene(inNode= in_Node, outNode=newID, weight = 1, enabled=True, innovation=innovationNum)
        innovationNum += 1
        newConnectionGene2 = ConnectionGene(inNode=newID, outNode=out_Node, weight = w, enabled=True, innovation=innovationNum)

        innovationList.append([in_Node, newID])
        innovationList.append([newID, out_Node])

        self.NodeList.append(newNode)
        self.ConnectionList.append(newConnectionGene1)
        self.ConnectionList.append(newConnectionGene2)

        #print('ID: ', self.id, end=' ')
        #print('Added Node: ', newID, end= ' ')
        #print('Added C1:', newConnectionGene1.inNode, newConnectionGene1.outNode, end=' ')
        #print('Added C2:', newConnectionGene2.inNode, newConnectionGene2.outNode)

        return innovationList, innovationNum

    def mutateWeights(self):
        for connection in self.ConnectionList:
            r = np.random.random()
            if(r <= 0.9):
                connection.weight += connection.weight*np.random.normal(0,1)
            elif(r <= 0.95):
                connection.weight = np.random.random()

    def seperateNodes(self):

        inputNodes = []
        outputNodes = []
        for node in self.NodeList:
            #also reset nodes
            node.value = 0
            if(node.layer == 'INPUT'):
                inputNodes.append(node)
            elif(node.layer == 'OUTPUT'):
                outputNodes.append(node)

        return inputNodes, outputNodes

    def buildNet(self):

        In = []
        Out = []
        inputNodes, outputNodes = self.seperateNodes()
        for i in inputNodes:
            In.append(i.id)
        for i in outputNodes:
            Out.append(i.id)
        phenotype = FeedForwardNetwork.create(inputsList=In, outputList=Out, connectionList=self.ConnectionList)

        return phenotype

    def active(self, phenotype, inputs):
        outComes = phenotype.activate(inputs)
        return outComes

    def draw_net(self, view=False, filename=None, node_names=None, show_disabled=False, prune_unused=False, node_colors=None, fmt='png'):
        """ Receives a genome and draws a neural network with arbitrary topology. """
        # Attributes for network nodes.

        inputNodes, outputNodes = self.seperateNodes()


        if graphviz is None:
            warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
            return

        if node_names is None:
            node_names = {}

        assert type(node_names) is dict

        if node_colors is None:
            node_colors = {}

        assert type(node_colors) is dict

        node_attrs = {
            'shape': 'circle',
            'fontsize': '9',
            'height': '0.2',
            'width': '0.2'}

        dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

        inputs = set()
        for k in inputNodes:
            inputs.add(k.id)
            name = node_names.get(k.id, str(k.id))
            input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k.id, 'lightgray')}
            dot.node(name, _attributes=input_attrs)

        outputs = set()
        for k in outputNodes:
            outputs.add(k.id)
            name = node_names.get(k.id, str(k.id))
            node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k.id, 'lightblue')}
            dot.node(name, _attributes=node_attrs)



        if prune_unused:
            connections = set()
            #for cg in genome.connections.values():
            for cg in self.ConnectionList:
                if cg.enabled or show_disabled:
                    connections.add((cg.inNode, cg.outNode))

            used_nodes = copy.copy(outputs)
            pending = copy.copy(outputs)
            while pending:
                new_pending = set()
                for a, b in connections:
                    if b in pending and a not in used_nodes:
                        new_pending.add(a)
                        used_nodes.add(a)
                pending = new_pending
        else:
            used_nodes = set([i.id for i in self.NodeList])


        for n in used_nodes:
            if n in inputs or n in outputs:
                continue

            attrs = {'style': 'filled',
                     'fillcolor': node_colors.get(n, 'white')}
            dot.node(str(n), _attributes=attrs)
        #print('\nID', self.id)
        #print([(i.inNode, i.outNode) for i in self.ConnectionList])
        #print([(i.inNode, i.outNode) for i in self.ConnectionList if i.enabled])
        for cg in self.ConnectionList:
            if cg.enabled or show_disabled:
                # if cg.input not in used_nodes or cg.output not in used_nodes:
                #    continue

                input = cg.inNode
                output = cg.outNode
                #print((input, output), end='')
                a = node_names.get(input, str(input))
                b = node_names.get(output, str(output))
                style = 'solid' if cg.enabled else 'dotted'
                color = 'green' if cg.weight > 0 else 'red'
                width = str(0.1 + abs(cg.weight / 5.0))
                dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

        dot.render(filename, view=view)

        return dot




class ConnectionGene():

    def __init__(self, inNode, outNode, weight, enabled, innovation):
        self.inNode = inNode
        self.outNode = outNode
        self.weight = weight
        self.enabled = enabled
        self.innovation = innovation

    def disable(self):
        self.enabled = False

class NodeGene():

    types = {'INPUT', 'OUTPUT', 'HIDDEN'}

    def __init__(self, type, id):
        if type not in self.types:
            raise ValueError("Node gene types must be one of %r" % self.types)

        self.layer = type
        self.id = id
