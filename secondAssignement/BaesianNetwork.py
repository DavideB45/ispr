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
    def sample(self):
        ...

class Bernoulli(Distribution):
    def __init__(self, p:float):
        self.p = p
    def __str__(self):
        return f'p={self.p}'

    def sample(self):
        return 0 if random.random() < self.p else 1
    
class Multinomial(Distribution):
    def __init__(self, pList:list[float]):
        self.pList = pList
        if sum(pList) != 1:
            raise('The sum of the probabilities should be 1')
    def __str__(self):
        return f'pList={self.pList}'
    
    def sample(self):
        r = random.random()
        for i, p in enumerate(self.pList):
            if r < p:
                return i
            r -= p
        return len(self.pList)-1
    
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
    def __init__(self, conditioners):
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
        self.observed = None

    def __str__(self):
        return f'{self.name} -> ({self.parents})'
    
    def sample(self):
        pass

    def update(self, parent:str, value:int):
        pass

class BaesianNetwork:
    # check if the graph is acyclic
    # check if all names are unique
    # check if all tables are complete
    def __init__(self, nodes):
        self.nodes = nodes
        self.graph = {}
        self.variables = {}

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