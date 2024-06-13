import numpy
import random
from math import exp
import matplotlib.pyplot as plt

def selectArm(policy):
    selectedArm = random.randint(0, (policy.count(max(policy))-1))
    for i in range(len(policy)):
        if selectedArm == 0 and policy[i] == max(policy):
            return i
        elif policy[i] == max(policy):
            selectedArm -= 1

def selectGradiant(policy):
    eSum = 0
    for i in policy:
        eSum += exp(i)
    selectedArm = random.random()*eSum
    count = 0
    while selectedArm > 0:
        selectedArm -= exp(policy[count])
        count += 1
    return count-1, eSum

def averageReward(policy, actuals, epsilon=False):
    indices = []
    for i in range(len(policy)):
        if policy[i] == max(policy):
            indices.append(i)
    average = 0
    for i in indices:
        average += actuals[i]
    if epsilon != False:
        return ((average/len(indices)) * (1-epsilon)) + ((sum(actuals)/len(actuals)) * epsilon)
    return average/len(indices)

def optimalProbability(optimal, policy, epsilon=False, gradient=False):
    if epsilon != False:
        indices = []
        for i in range(len(policy)):
            if policy[i] == max(policy):
                indices.append(i)
        if optimal in indices:
            return (1/len(policy)) * epsilon + (1/len(indices)) * (1-epsilon)
        else:
            return (1/len(policy)) * epsilon
    elif gradient != False:
        return exp(policy[optimal])/gradient
    else:
        indices = []
        for i in range(len(policy)):
            if policy[i] == max(policy):
                indices.append(i)
        if optimal in indices:
            return 1/len(indices)
        else:
            return 0

def singleIterationPart1():
    #Gradient is spelled wrong consistently, I only noticed this towards the end of writing the code and I didn't want to go back to change it.
    means = []
    nonOptimisticPolicy = []
    nonOptimisticCounts = []
    epsilonPolicy = []
    epsilonCounts = []
    epsilonValue = 0.1
    optimisticPolicy = []
    optimisticCounts = []
    gradiantPolicy = []
    gradiantRewardSum = 0
    gradiantCount = 1
    gradiantZeroCheck = True
    gradiantAlpha = 0.05
    for i in range(10):
        means.append(numpy.random.normal(0, 1))
        nonOptimisticPolicy.append(0)
        epsilonPolicy.append(0)
        gradiantPolicy.append(0)
        nonOptimisticCounts.append(0)
        epsilonCounts.append(0)
        optimisticCounts.append(3)
    for i in range(10):
        optimisticPolicy.append(max(means)+3)
    for i in range(len(means)):
        if max(means) == means[i]:
            optimalChoice = i
    rewards = []
    probabilities = []
    timeStep = 0
    while timeStep < 10000:
        selectedArm, eSum = selectGradiant(gradiantPolicy)
        rewards.append([averageReward(nonOptimisticPolicy, means), averageReward(optimisticPolicy, means), averageReward(epsilonPolicy, means, epsilonValue)])
        probabilities.append([optimalProbability(optimalChoice, nonOptimisticPolicy), optimalProbability(optimalChoice, optimisticPolicy), optimalProbability(optimalChoice, epsilonPolicy, epsilonValue), optimalProbability(optimalChoice, gradiantPolicy, False, eSum)])
        selectedArm = selectArm(nonOptimisticPolicy)
        nonOptimisticPolicy[selectedArm] = (numpy.random.normal(means[selectedArm], 1) + nonOptimisticCounts[selectedArm]*nonOptimisticPolicy[selectedArm])/(nonOptimisticCounts[selectedArm]+1)
        nonOptimisticCounts[selectedArm] += 1
        if random.random() <= epsilonValue:
            selectedArm = random.randint(0, 9)
        else:
            selectedArm = selectArm(epsilonPolicy)
        epsilonPolicy[selectedArm] = (numpy.random.normal(means[selectedArm], 1) + epsilonCounts[selectedArm]*epsilonPolicy[selectedArm])/(epsilonCounts[selectedArm]+1)
        epsilonCounts[selectedArm] += 1
        selectedArm = selectArm(optimisticPolicy)
        optimisticPolicy[selectedArm] = (numpy.random.normal(means[selectedArm], 1) + optimisticCounts[selectedArm]*optimisticPolicy[selectedArm])/(optimisticCounts[selectedArm]+1)
        optimisticCounts[selectedArm] += 1
        gradiantReward = numpy.random.normal(means[selectedArm], 1)
        gradiantPolicy[selectedArm] = gradiantPolicy[selectedArm] + gradiantAlpha*(gradiantReward - (gradiantRewardSum/gradiantCount))*(1-(exp(gradiantPolicy[selectedArm])/eSum))
        for i in range(10):
            gradiantPolicy[i] = gradiantPolicy[i] - gradiantAlpha*(gradiantReward - (gradiantRewardSum/gradiantCount))*(exp(gradiantPolicy[i])/eSum)
        gradiantRewardSum += gradiantReward
        if gradiantZeroCheck:
            gradiantZeroCheck = False
        else:
            gradiantCount += 1
        timeStep += 1
    return rewards, probabilities, means


probabilities = []
averages = []
for i in range(4):
    newLine = []
    for j in range(10000):
        newLine.append(0)
    probabilities.append(newLine)
for i in range(3):
    newLine = []
    for j in range(10000):
        newLine.append(0)
    averages.append(newLine)
iterations = 5
for i in range(iterations):
    res = singleIterationPart1()
    for i in range(len(res[0])):
        for j in range(3):
            averages[j][i] += res[0][i][j]
        for j in range(4):
            probabilities[j][i] += res[1][i][j]
for i in range(3):
    for j in range(10000):
        averages[i][j] = averages[i][j]/iterations
for i in range(4):
    for j in range(10000):
        probabilities[i][j] = probabilities[i][j]/iterations

#Showing gradient probability of correct decision
plt.title("Gradient correct decision probability")
plt.xlabel("Time")
plt.ylabel("Probability of correct decision")
plt.plot(probabilities[3])
plt.show()
#Showing epsilon-greedy probability of correct decision
plt.title("Epsilon-greedy correct decision probability")
plt.xlabel("Time")
plt.ylabel("Probability of correct decision")
plt.plot(probabilities[2])
plt.show()
#Showing optimistic probability of correct decision
plt.title("Optimistic greedy correct decision probability")
plt.xlabel("Time")
plt.ylabel("Probability of correct decision")
plt.plot(probabilities[1])
plt.show()
#Showing non-optimistic probability of correct decision
plt.title("Non-Optimistic greedy correct decision probability")
plt.xlabel("Time")
plt.ylabel("Probability of correct decision")
plt.plot(probabilities[0])
plt.show()
#Showing epsilon-greedy average reward
plt.title("Epsilon-greedy average reward")
plt.xlabel("Time")
plt.ylabel("Average reward")
plt.plot(averages[2])
plt.show()
#Showing optimistic average reward
plt.title("Optimistic average reward")
plt.xlabel("Time")
plt.ylabel("Average reward")
plt.plot(averages[1])
plt.show()
#Showing non-optimistic average reward
plt.title("Non-optimistic average reward")
plt.xlabel("Time")
plt.ylabel("Average reward")
plt.plot(averages[0])
plt.show()