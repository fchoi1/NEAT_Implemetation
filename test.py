from copy import deepcopy
import numpy as np
from genome import genome, ConnectionGene, NodeGene

from multiprocessing import Process, Queue
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from statistics import median, mean

from itertools import chain

test_1, test_2 = np.array([1,2])
'''
Embed = nn.Embedding(11, 3)
test = torch.LongTensor([[1, 2, 3, 4], [1, 2, 8, 7]])
output = Embed(test)
#print(output)
test2 = torch.LongTensor([[1,4,5,6], [2,4,12,12]])


test3 = ([[1,2],[3,4],[5,6],[7,8]])

test4 = []
for i in test3:
	test4.append([i,([1,0])])
	#test4 = np.append(test4,[i,[[1,0,0]]])

print(test4)
test4 = np.array(test4,np.float)
print(test4[:,0])
#print(test4.dtype)

test5 = torch.from_numpy(np.array(test4[:,0]))
print(test5)

loss = nn.CrossEntropyLoss()
input = torch.randn(128, 128, 1, requires_grad=True)

target = torch.empty(128, 1, dtype=torch.long).random_(3)
print(input.size())
print(target.size())
print(target)
output = loss(input, target)
output.backward()
print(output)





y1 = torch.FloatTensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]])



_, targets = y1.max(dim=0)
print(targets)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_fitmodule import FitModule
from sklearn.datasets import make_multilabel_classification


SEED = 1701


def print_title(s):
    print("\n\n{0}\n{1}\n{0}".format("="*len(s), s))


##### Generate training set #####
print_title("Generating data set")

n_feats, n_classes = 200, 5
X, y = make_multilabel_classification(
    n_samples=10000, n_features=n_feats, n_classes=n_classes, n_labels=0.01,
    length=50, allow_unlabeled=False, sparse=False, return_indicator='dense',
    return_distributions=False, random_state=SEED
)
y = np.argmax(y, axis=1)
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).long()

print(f'array {X}, size: {X.size()}')
print(f'array {y}, size: {y.size()}')

##### Define model #####
print_title("Building model")

class MLP(FitModule):
    def __init__(self, n_feats, n_classes, hidden_size=50):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_feats, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_classes)
    def forward(self, x):

   		x = self.fc1(x)
   		x = F.relu(x)
   		x = self.fc2(x)
   		x = F.log_softmax(x)
   		return x
        #return F.log_softmax(self.fc2(F.relu(self.fc1(x))))

f = MLP(n_feats, n_classes)


##### Train model #####
print_title("Training model")

def accuracy(y_true, y_pred):
    return np.mean(y_true.numpy() == np.argmax(y_pred.numpy(), axis=1))
f.fit(X,y,epochs=3,batch_size=10)
#f.fit(    X, y, epochs=10, validation_split=0.3, seed=SEED, metrics=[accuracy] )

rand_X = torch.rand(10000,200)
rand_X=torch.round(rand_X)


pred = f(rand_X)
print(pred)
print(pred.size())

test =[[[0],[0],[0],[0]]]*10

print(test)
test[1] = [[-2],[1,2,5,6,7,8,2,1],[0,1,0],[1234]]
test[2] = [[-3],[1,2,5,6,7,8,2,1],[0,1,0],[1234]]
test[3] = [[1],[4,2,5,6,7,8,2,13],[0,0,1],[994]]
test[4] = [[],[],[],[]]


#print([[i[0] for i in test], [i[1] for i in test]])

test.sort(key=lambda l:l[1])
print(test)

print(len(test))
#print('\n\n\n\n\n')
#print([i for i in test])
#print([i[0] for i in test])

#print(mean([i[0][0] for i in test]))
#print([i[0][0] for i in test])
filter(True,test)
print(len(test))
print(test)

def delete(x):
  try:
      if (x[0] >= -21):
        return True
      else:
        return False
  except IndexError:
    return False



test_2 = [[-22.0,0.0,[0]]]* 6

print(test_2)
test_2.append([-10.0,1543,[564,1312]])
test_2.append([-31.0,1521,[342,5352]])
test_2.append([-18.0,1000,[123,1231]])
test_2.append([])
print(test_2)

test_2=list(filter(delete,test_2))

#print('Array out of index')
print(test_2)


class Cup:
    def __init__(self, color):
        self._color = color    # protected variable
        self.__content = None  # private variable

    def fill(self, beverage):
        self.__content = beverage

    def empty(self):
        self.__content = None

redCup = Cup("red")
redCup._Cup__content = "tea"
print(redCup._color)

this_1 = 5


'''

a = [1,2,3,4,5]
b = [2,4,6]
c = list(set(a + b))

NodeList1 = []
NodeList2 = []
Con1 = []
Con2 = []

newNode1 = NodeGene('INPUT', 0)
newNode2 = NodeGene('INPUT', 1)
newNode3 = NodeGene('INPUT', 2)
newNode4 = NodeGene('INPUT', 3)
newNode5 = NodeGene('INPUT', 4)

hiddenNode1 = NodeGene('HIDDEN',8)
hiddenNode2 = NodeGene('HIDDEN',9)
hiddenNode3 = NodeGene('HIDDEN',10)
hiddenNode4 = NodeGene('HIDDEN',11)

outputNode1 = NodeGene('OUTPUT', 5)
outputNode2 = NodeGene('OUTPUT', 6)
outputNode3 = NodeGene('OUTPUT', 7)

NodeList1.extend([newNode1, newNode2, newNode3, newNode4, newNode5, outputNode1, outputNode2, outputNode3, hiddenNode2, hiddenNode3,hiddenNode4])
NodeList2.extend([newNode1, newNode2, newNode3, newNode4, newNode5, outputNode1, outputNode2, outputNode3, hiddenNode1,hiddenNode4])
counter = 0
for i in range(5):
    newConnectionGene = ConnectionGene(inNode=i, outNode=5, weight=1, enabled=True, innovation=counter*3 )
    newConnectionGene2 = ConnectionGene(inNode=i, outNode=6, weight=2, enabled=True, innovation=counter*3 + 1)
    newConnectionGene3 = ConnectionGene(inNode=i, outNode=7, weight=3, enabled=True, innovation=counter*3 + 2)
    Con1.extend([newConnectionGene, newConnectionGene2, newConnectionGene3])
    Con2.extend([newConnectionGene, newConnectionGene2, newConnectionGene3])
    counter += 1


hiddenConnection1 = ConnectionGene(inNode=0, outNode=8, weight=1, enabled=True, innovation=21)
hiddenConnection2 = ConnectionGene(inNode=1, outNode=8, weight=2, enabled=True, innovation=22)
hiddenConnection3 = ConnectionGene(inNode=2, outNode=8, weight=3, enabled=True, innovation=23)
hiddenConnection4 = ConnectionGene(inNode=0, outNode=8, weight=4, enabled=True, innovation=24)
hiddenConnection5 = ConnectionGene(inNode=1, outNode=9, weight=5, enabled=True, innovation=25)
hiddenConnection6 = ConnectionGene(inNode=2, outNode=10, weight=6, enabled=True, innovation=26)

Con1.extend([hiddenConnection1, hiddenConnection2, hiddenConnection3, hiddenConnection4, hiddenConnection5, hiddenConnection6])
Con2.extend([hiddenConnection1, hiddenConnection3, hiddenConnection5])

parent1 = genome(Con1, NodeList1, 29, 1)
parent2 = genome(Con2, NodeList2, 29, 2)


def crossGenomes(genome1, genome2, equal=False):
    # Parent 1 is always more fit

    CListparent1 = list(genome1.ConnectionList)
    Clistparent2 = list(genome2.ConnectionList)
    parent2temp = list(genome2.ConnectionList)
    #CListparent1 = deepcopy(genome1.ConnectionList)
    #Clistparent2 = deepcopy(genome2.ConnectionList)
    #parent2temp = deepcopy(genome2.ConnectionList)

    childConnectionList = []
    childNodeList = list(set(genome1.NodeList + genome2.NodeList))

    for connection1 in CListparent1:
        disjoint = True
        for connection2 in Clistparent2:
            if (connection1.innovation == connection2.innovation):
                childConnectionList.append(connection1) if np.random.randint(0, 2) else childConnectionList.append(
                    connection2)
                try:
                    parent2temp.remove(connection2)
                except ValueError:
                    None
                    # print('Value already removed')

                disjoint = False
                break

        if (disjoint and equal and np.random.randint(0, 2)):
            childConnectionList.append(connection1)

        if (disjoint and not equal):
            childConnectionList.append(connection1)

        # print([(i.inNode, i.outNode) for i in parent2])

    if (equal):
        [childConnectionList.append(i) for i in parent2temp if np.random.randint(0, 2)]



    for a_node in childNodeList:
        if ((a_node.id not in [i.inNode for i in childConnectionList]) and (a_node.id not in [i.outNode for i in childConnectionList])):
            print('Removed node: ', a_node.id)
            childNodeList.remove(a_node)

    childGenome = genome(childConnectionList, childNodeList, 29, 3)
    return childGenome


child = crossGenomes(parent1,parent2,True)



def mutateAddNode(child):
    n = np.random.randint(0, len(child.ConnectionList))
    child.ConnectionList[n].disable()
    print('Disabled connection', (child.ConnectionList[n].inNode, child.ConnectionList[n].outNode), ' from ID ', child.id)

    in_Node = child.ConnectionList[n].inNode
    out_Node = child.ConnectionList[n].outNode
    w = child.ConnectionList[n].weight


    newID = len(child.NodeList)
    newNode = NodeGene(type='HIDDEN', id=newID)

    newConnectionGene1 = ConnectionGene(inNode=in_Node, outNode=newID, weight=1, enabled=True, innovation=50)
    newConnectionGene2 = ConnectionGene(inNode=newID, outNode=out_Node, weight=w, enabled=True, innovation=51)

    child.NodeList.append(newNode)
    child.ConnectionList.append(newConnectionGene1)
    child.ConnectionList.append(newConnectionGene2)

    print('ID: ', child.id, end=' ')
    print('Added Node: ', newID, end=' ')
    print('Added C1:', newConnectionGene1.inNode, newConnectionGene1.outNode, end=' ')
    print('Added C2:', newConnectionGene2.inNode, newConnectionGene2.outNode)

    return child




print('\nResults2')
print('\nResults 1')

print('Parents')
print([(i.inNode, i.outNode, i.enabled) for i in parent1.ConnectionList])
print([(i.inNode, i.outNode, i.enabled) for i in parent2.ConnectionList])
print('child')
print([(i.inNode, i.outNode, i.enabled) for i in child.ConnectionList])

child = mutateAddNode(child)

print('\n\n\n')
print('Parents')
print([(i.inNode, i.outNode, i.enabled) for i in parent1.ConnectionList])
print([(i.inNode, i.outNode, i.enabled) for i in parent2.ConnectionList])
print('child')
print([(i.inNode, i.outNode, i.enabled) for i in child.ConnectionList])



copy = deepcopy(child)
print(child.fitness)
copy.fitness =31

print(child.fitness, copy.fitness)