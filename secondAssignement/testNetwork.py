from BayesianNetwork import *
from BayesianNetwork import BayesianNetwork

# 1 Want to be healthy
CPTA = CPT()
CPTA.setDistribution(valuedParents={}, distribution=Bernoulli(0.5))
A = Node('want to be healty', 'H', [], CPTA)

# 2 Eat healthy
CPTB = CPT([('H', 2)])
CPTB.setDistribution(valuedParents={'H':0}, distribution=Bernoulli(0.3))
CPTB.setDistribution(valuedParents={'H':1}, distribution=Bernoulli(0.8))
B = Node('eat healthy', 'E', [A], CPTB)

# 3 Exercise
CPTC = CPT([('H', 2)])
CPTC.setDistribution(valuedParents={'H':0}, distribution=Bernoulli(0.2))
CPTC.setDistribution(valuedParents={'H':1}, distribution=Bernoulli(0.8))
C = Node('exercise', 'X', [A], CPTC)

# 4 Be phisically healthy
CPTD = CPT([('E', 2), ('X', 2)])
CPTD.setDistribution(valuedParents={'E':0, 'X':0}, distribution=Bernoulli(0.5))
CPTD.setDistribution(valuedParents={'E':0, 'X':1}, distribution=Bernoulli(0.7))
CPTD.setDistribution(valuedParents={'E':1, 'X':0}, distribution=Bernoulli(0.7))
CPTD.setDistribution(valuedParents={'E':1, 'X':1}, distribution=Bernoulli(0.9))
D = Node('be healthy', 'PH', [B, C], CPTD)

network = BayesianNetwork({'H': A, 'E': B, 'X': C, 'PH': D})

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