#TODO: NODE:
#   - parents: list of parents (maybe intrested in name of parents)
#   - cpt: conditional probability table (a CPT object tbd)
#   - name: name of the node (string)
#
#   - children: list of children (important for ancestral sampling)
#   - sample: sample from the node (randomly given all parents outputs) -> call update on children
#   - update: pass the new evidence to the node (from the parent) -> call sample on self

import random


class Distribution:
    def sample(self) -> int:
        ...

class Bernoulli(Distribution):
    def __init__(self, p:float):
        self.p = p
    def __str__(self):
        return f'p={self.p}'

    def sample(self) -> int:
        return 1 if random.random() < self.p else 0
    
class Multinomial(Distribution):
    def __init__(self, pList:list[float]):
        self.pList = pList
        if sum(pList) != 1:
            raise('The sum of the probabilities should be 1')
    def __str__(self):
        return f'pList={self.pList}'
    
    def sample(self) -> int:
        r = random.random()
        for i, p in enumerate(self.pList):
            if r < p:
                return i
            r -= p
        return len(self.pList)-1

class UniformMultinomial(Multinomial):
    def __init__(self, n:int):
        self.n = n
        self.pList = [1/n]*n

    def __str__(self):
        return f'n={self.n}'

    def sample(self) -> int:
        return random.randint(0, self.n-1)
    
class CPT:
    '''
    A Fancy Hash table
    '''
    # should allow quick access to the probability of a certain value given the values of the parents
    # also if parents are not all observed, should return something that make sense (i.e. a probability distribution)
    # alternatively, throw an error
    # alernatively return a CPT resticted to the unobserved parents
    # is this just a big matrix?
    #take a list of touple: [(id, max1), (id, max2), ...]
    def __init__(self, conditioners:list[tuple[str, int]] = []):
        self.conditionersIdOrder = [name for name, _ in conditioners]
        self.table = {}
        self.keys = ['']
        for name, max in conditioners:
            tempKeys = []
            for i in range(max):
                tempKeys += [f'{key}{name}{i}' for key in self.keys]
            self.keys = tempKeys

    def _getKeyFromQuery(self, valuedParents:dict[str:int], safeCreation:bool=False)->str:
        '''Get the key from the query'''
        key = '' 
        for conditioner in self.conditionersIdOrder:
            try:
                key += f'{conditioner}{valuedParents[conditioner]}'
            except KeyError:
                raise('Not all parents ID are present')
        if safeCreation and (key not in self.keys):
            raise('Key not in the table')
        return key

    def checkComplete(self)->bool:
        '''Check if the table is complete (i.e. all possible values are defined)'''
        return len(self.keys) == len(self.table)

    def setDistribution(self, valuedParents:dict[str:int], distribution:Distribution)->None:
        '''Set the distribution for a certain set of parents
        
        valuedParents: dict (id, value) for the parents that are observed
        distribution: the distribution to use when the parents are observed
        
        raises 'Not all parents are present' if not all parents are present
        raises 'Key not in the table' if the key is not in the table
        '''
        key = self._getKeyFromQuery(valuedParents)
        self.table[key] = distribution

    def getDistribution(self, valuedParents:dict[str:int])->Distribution:
        '''Get the distribution for a certain set of parents
        
        valuedParents: dictionary (id, value) for the parents that are observed

        raises 'Table is not complete' if the table is not complete
        raises 'Not all parents are present' if not all parents are present
        raises 'Key not in the table' if the key is not in the table
        '''
        if not self.checkComplete():
            raise('Table is not complete')
        key = self._getKeyFromQuery(valuedParents)
        return self.table[key]

class Node:
    def __init__(self, name:str, id:str, parents:list, cpt:CPT):
        self.name = name
        if id is None or id == '':
            raise('id cannot be None or empty')
        self.id = id
        self.parents = parents
        self.cpt = cpt
        self.children = []
        for parent in parents:
            parent.addChild(self)
        #current value of the node
        self.value = None
        #current value of the parents
        self.observed = {}

    def __str__(self):
        return f'{self.name} -> ({self.parents})'
    
    def addChild(self, child):
        self.children.append(child)
    
    def sample(self) -> int:
        self.value = self.cpt.getDistribution(self.observed).sample()
        print(f'{self.name} -> {self.value}')
        for child in self.children:
            child.update(self.id, self.value)

    def update(self, parent:str, value:int):
        self.observed[parent] = value
        if len(self.parents) == len(self.observed):
            # if something does not work maybe 
            # make a full check of the parents
            self.sample()
    
    def reset(self):
        self.value = None
        self.observed = {}
        

class BayesianNetwork:
    # check if the graph is acyclic
    # check if all tables are complete
    def __init__(self, nodes:dict[str, Node]):
        self._nodes = nodes
        self._checkUniqueId()
        self._checkCompleteTables()
        self._checkAcyclic()
        print(nodes)
        self._orphans = self._computeOrphans()

    def _computeOrphans(self)->list[Node]:
        orphans = []
        for node in self._nodes.values():
            if len(node.parents) == 0:
                print('orphan', node.name)
                orphans.append(node)
        return orphans
    
    def _checkAcyclic(self):
        # Create a dictionary to store the in-degree of each node
        in_degree = {node: 0 for node in self._nodes.values()}
        # Calculate the in-degree of each node
        for node in self._nodes.values():
            for child in node.children:
                in_degree[child] += 1
        # Create a queue to store nodes with in-degree 0
        queue = [node for node, degree in in_degree.items() if degree == 0]
        # Perform exploration
        while queue:
            node = queue.pop(0)
            for child in node.children:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        # Check if there are any nodes with non-zero in-degree
        for degree in in_degree.values():
            if degree != 0:
                raise Exception('Graph is cyclic')

    def _checkUniqueId(self):
        for node in self._nodes.values():
            for otherNode in self._nodes.values():
                if node != otherNode and node.id == otherNode.id:
                    raise(f'Ids are not unique: {node.name} and {otherNode.name} have id {node.id}')
                
    def _checkCompleteTables(self):
        for node in self._nodes.values():
            if not node.cpt.checkComplete():
                print(f'Not all tables are complete, check -> {node.name} <-')
                raise(f'Not all tables are complete, check -> {node.name} <-')
    
    def sample(self) -> dict[str:int]:
        print("orphans", self._orphans)
        for orphan in self._orphans:
            orphan.sample()
        results = {}
        for node in self._nodes.values():
            results[node.id] = node.value
        return results
    
    def sampleN(self, n:int) -> list[dict[str,int]]:
        samples = []
        for _ in range(n):
            samples.append(self.sample())
            self.reset()
        return samples
    
    def reset(self):
        for node in self._nodes.values():
            node.reset()

if __name__ == '__main__':
    cpt = CPT([('A', 2), ('B', 2)])
    cpt.setDistribution({'A':0, 'B':0}, Bernoulli(0.1))
    cpt.setDistribution({'A':0, 'B':1}, Bernoulli(0.1))
    cpt.setDistribution({'A':1, 'B':0}, Multinomial([0.1, 0.5, 0.4]))
    cpt.setDistribution({'A':1, 'B':1}, Multinomial([0.1, 0.5, 0.4]))
    print(cpt.getDistribution({'A':0, 'B':0}).sample())
    print(cpt.getDistribution({'A':0, 'B':1}))
    print(cpt.getDistribution({'A':1, 'B':0}).sample())
    print(cpt.getDistribution({'A':1, 'B':1}))