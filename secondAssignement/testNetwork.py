from BayesianNetwork import *
from BayesianNetwork import BayesianNetwork

nodes = {}

# 0 Take free time
CPT0 = CPT()
# 4 levels of free time
# 0=none, 1=low, 2=medium, 3=high
CPT0.setDistribution(valuedParents={}, distribution=UniformMultinomial(4))
Z = Node('take free time', 'F', [], CPT0)
nodes[Z.id] = Z

# 1 Want to be healthy
CPTA = CPT()
CPTA.setDistribution(valuedParents={}, distribution=Bernoulli(0.5))
A = Node('want to be healty', 'H', [], CPTA)
nodes[A.id] = A

# 2 Eat healthy
CPTB = CPT([('H', 2)])
CPTB.setDistribution(valuedParents={'H':0}, distribution=Bernoulli(0.3))
CPTB.setDistribution(valuedParents={'H':1}, distribution=Bernoulli(0.8))
B = Node('eat healthy', 'E', [A], CPTB)
nodes[B.id] = B

# 3 Exercise
CPTC = CPT([('H', 2)])
CPTC.setDistribution(valuedParents={'H':0}, distribution=Bernoulli(0.2))
CPTC.setDistribution(valuedParents={'H':1}, distribution=Bernoulli(0.8))
C = Node('exercise', 'X', [A], CPTC)
nodes[C.id] = C

# 4 get injured
CPTD = CPT([('X', 2)])
# 3 levels of injury 0=none, 1=minor, 2=severe
CPTD.setDistribution(valuedParents={'X':0}, distribution=Multinomial([0.94, 0.05, 0.01]))
CPTD.setDistribution(valuedParents={'X':1}, distribution=Multinomial([0.85, 0.1, 0.05]))
D = Node('get injured', 'I', [C], CPTD)
nodes[D.id] = D

# 5 Be phisically healthy
# there could be many levels of health, but for simplicity we will consider only 2
CPTE = CPT([('E', 2), ('X', 2), ('I', 3)])
for injuryLev in range(3):
    penalty = injuryLev
    if injuryLev == 2:
        # if severe injury, the person is very unlikely to be healthy
        penalty = 4
    CPTE.setDistribution(valuedParents={'E':0, 'X':0, 'I':injuryLev}, distribution=Bernoulli(0.5 - penalty*0.1))
    CPTE.setDistribution(valuedParents={'E':0, 'X':1, 'I':injuryLev}, distribution=Bernoulli(0.7 - penalty*0.1))
    CPTE.setDistribution(valuedParents={'E':1, 'X':0, 'I':injuryLev}, distribution=Bernoulli(0.7 - penalty*0.1))
    CPTE.setDistribution(valuedParents={'E':1, 'X':1, 'I':injuryLev}, distribution=Bernoulli(0.9 - penalty*0.1))
E = Node('be healthy', 'PH', [B, C, D], CPTE)
nodes[E.id] = E

# 6 mental health
CPTF = CPT([('PH', 2), ('F', 4)])
for phisicalHealty in range(2):    
    CPTF.setDistribution(valuedParents={'PH':phisicalHealty, 'F':0}, distribution=Bernoulli(0.1 + phisicalHealty*0.2))
    CPTF.setDistribution(valuedParents={'PH':phisicalHealty, 'F':1}, distribution=Bernoulli(0.5 + phisicalHealty*0.2))
    CPTF.setDistribution(valuedParents={'PH':phisicalHealty, 'F':2}, distribution=Bernoulli(0.6 + phisicalHealty*0.2))
    CPTF.setDistribution(valuedParents={'PH':phisicalHealty, 'F':3}, distribution=Bernoulli(0.7 + phisicalHealty*0.2))
F = Node('mental health', 'MH', [E, Z], CPTF)
nodes[F.id] = F

# 7 being able to walk
CPTG = CPT([('I', 3)])
# maybe also if the person is healthy, the person could be tied to a bed (but it's unlikely)
CPTG.setDistribution(valuedParents={'I':0}, distribution=Bernoulli(0.9999))
CPTG.setDistribution(valuedParents={'I':1}, distribution=Bernoulli(0.98))
CPTG.setDistribution(valuedParents={'I':2}, distribution=Bernoulli(0.5))
G = Node('able to walk', 'W', [D], CPTG)
nodes[G.id] = G

# 8 study
CPTH = CPT([('MH', 2), ('F', 4)])
CPTH.setDistribution(valuedParents={'MH':0, 'F':0}, distribution=Bernoulli(0.5)) # going crazy (hard to study in this case)
CPTH.setDistribution(valuedParents={'MH':0, 'F':1}, distribution=Bernoulli(0.6)) # a bit realxed (may decide to study)
CPTH.setDistribution(valuedParents={'MH':0, 'F':2}, distribution=Bernoulli(0.7)) # relaxed (may decide to study)
CPTH.setDistribution(valuedParents={'MH':0, 'F':3}, distribution=Bernoulli(0.4)) # spend all time relaxing (no time to study)
CPTH.setDistribution(valuedParents={'MH':1, 'F':0}, distribution=Bernoulli(0.9)) # mentally fine (a mistery how it's possible)
CPTH.setDistribution(valuedParents={'MH':1, 'F':1}, distribution=Bernoulli(0.8)) # mentally fine (balanced life)
CPTH.setDistribution(valuedParents={'MH':1, 'F':2}, distribution=Bernoulli(0.7)) # mentally fine (balanced life)
CPTH.setDistribution(valuedParents={'MH':1, 'F':3}, distribution=Bernoulli(0.5)) # chilled guy with no time to study
H = Node('study', 'S', [F, Z], CPTH)
nodes[H.id] = H

# 9 go to the lectures
CPTI = CPT([('S', 2), ('W', 2)])
CPTI.setDistribution(valuedParents={'S':0, 'W':0}, distribution=Bernoulli(0.2)) # not studying and not able to walk
CPTI.setDistribution(valuedParents={'S':0, 'W':1}, distribution=Bernoulli(0.5)) # not studying and able to walk
CPTI.setDistribution(valuedParents={'S':1, 'W':0}, distribution=Bernoulli(0.3)) # studying and not able to walk
CPTI.setDistribution(valuedParents={'S':1, 'W':1}, distribution=Bernoulli(0.9)) # studying and able to walk
I = Node('go to the lectures', 'L', [H, G], CPTI)
nodes[I.id] = I

network = BayesianNetwork(nodes=nodes)

samples = network.sampleN(10000)
#check if being healthy is dependent of wanting to be healthy
countWantHealty = 0
countWantHealtyAndHealty = 0
countWantHealtyAndNotHealty = 0
totSamples = 0
for sample in samples:
    if sample['H'] == 1 and sample['PH'] == 1:
        countWantHealtyAndHealty += 1
    if sample['H'] == 1:
        countWantHealty += 1
    if sample['H'] == 0 and sample['PH'] == 1:
        countWantHealtyAndNotHealty += 1
    totSamples += 1

print("P(Healty|WantHealty)    = ", countWantHealtyAndHealty/countWantHealty)
print("P(Healty|NotWantHealty) = ", countWantHealtyAndNotHealty/(totSamples - countWantHealty))

#TODO: use multinomial to check if the distribution is correct