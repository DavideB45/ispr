#TODO: NODE:
#   - parents: list of parents (maybe intrested in name of parents)
#   - cpt: conditional probability table (a CPT object tbd)
#   - name: name of the node (string)
#
#   - children: list of children (important for ancestral sampling)
#   - sample: sample from the node (randomly given all parents outputs) -> call update on children
#   - update: pass the new evidence to the node (from the parent) -> call sample on self


class CPT:
    # should allow quick access to the probability of a certain value given the values of the parents
    # also if parents are not all observed, should return something that make sense (i.e. a probability distribution)
    # alternatively, throw an error
    # alernatively return a CPT resticted to the unobserved parents
    # is this just a big matrix?
    #take a list of touple: [(name1, max1), (name2, max2), ...]
    def __init__(self, conditioners):
        self.conditionersNameOrder = [name for name, _ in conditioners]
        self.table = {}
        self.keys = ['']
        for name, max in conditioners:
            for i in range(max):
                self.keys = [f'{key}{name}{i}' for key in self.keys]
                
                
            

    def checkComplete(self):
        # check if the table is complete
        pass

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