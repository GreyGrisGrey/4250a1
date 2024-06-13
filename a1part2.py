import numpy
import random
import matplotlib.pyplot as plt

#Permutes Means
def permute(means):
    newMeans = [None, None, None, None, None, None, None, None, None, None]
    for i in range(10):
        nextSpot = random.randint(0, 9-i)
        for j in range(10):
            if newMeans[j] == None and nextSpot == 0:
                newMeans[j] = means[i]
                nextSpot -= 1
            elif newMeans[j] == None:
                nextSpot -= 1
    return newMeans


#Greedily selects a one-armed bandit given a policy
def selectArm(policy):
    selectedArm = random.randint(0, (policy.count(max(policy))-1))
    for i in range(len(policy)):
        if selectedArm == 0 and policy[i] == max(policy):
            return i
        elif policy[i] == max(policy):
            selectedArm -= 1

#Given a policy and the actual means for each one-armed bandit, calculates the average reward an algorithm will receive
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

def singleIterationPart2a():
    #Setting up values to be used later
    #Would have been cleaner as a couple arrays of arrays but it shouldn't effect performance
    means = []
    epsilonPolicy = []
    epsilonCounts = []
    epsilonMoving = []
    epsilonValue = 0.02
    optimisticPolicy = []
    optimisticCounts = []
    #Filling policy arrays and determining averages for each one-armed bandit
    for i in range(10):
        means.append(numpy.random.normal(0, 1))
        epsilonPolicy.append(0)
        epsilonCounts.append(0)
        epsilonMoving.append(0)
        optimisticCounts.append(3)
    for i in range(10):
        optimisticPolicy.append(max(means)+3)
    timeStep = 0
    #Runs 10000 iterations of the algorithms selecting one armed bandits and receiving rewards.
    while timeStep < 10000:
        #Calculates the updated stationary Epsilon-greedy policy
        if random.random() <= epsilonValue:
            selectedArm = random.randint(0, 9)
        else:
            selectedArm = selectArm(epsilonPolicy)
        epsilonPolicy[selectedArm] = (numpy.random.normal(means[selectedArm], 1) + epsilonCounts[selectedArm]*epsilonPolicy[selectedArm])/(epsilonCounts[selectedArm]+1)
        epsilonCounts[selectedArm] += 1
        #Calculates the updated optimistic policy
        selectedArm = selectArm(optimisticPolicy)
        optimisticPolicy[selectedArm] = (numpy.random.normal(means[selectedArm], 1) + optimisticCounts[selectedArm]*optimisticPolicy[selectedArm])/(optimisticCounts[selectedArm]+1)
        optimisticCounts[selectedArm] += 1
        #Calculates the updated moving Epsilon-Greedy policy
        if random.random() <= epsilonValue:
            selectedArm = random.randint(0, 9)
        else:
            selectedArm = selectArm(epsilonMoving)
        movingReward = numpy.random.normal(means[selectedArm], 1)
        epsilonMoving[selectedArm] = movingReward + 0.2*(epsilonMoving[selectedArm] - movingReward)
        timeStep += 1
        #Updates means
        for i in range(len(means)):
            means[i] = means[i] + numpy.random.normal(0, 0.000001)
    rewards = [averageReward(optimisticPolicy, means), averageReward(epsilonPolicy, means, epsilonValue), averageReward(epsilonMoving, means, epsilonValue)]
    return rewards

def singleIterationPart2b():
    #Setting up values to be used later
    #Would have been cleaner as a couple arrays of arrays but it shouldn't effect performance
    means = []
    epsilonPolicy = []
    epsilonCounts = []
    epsilonMoving = []
    epsilonValue = 0.02
    optimisticPolicy = []
    optimisticCounts = []
    #Filling policy arrays and determining averages for each one-armed bandit
    for i in range(10):
        means.append(numpy.random.normal(0, 1))
        epsilonPolicy.append(0)
        epsilonCounts.append(0)
        epsilonMoving.append(0)
        optimisticCounts.append(3)
    for i in range(10):
        optimisticPolicy.append(max(means)+3)
    timeStep = 0
    #Runs 10000 iterations of the algorithms selecting one armed bandits and receiving rewards.
    while timeStep < 10000:
        #Calculates the updated stationary Epsilon-greedy policy
        if random.random() <= epsilonValue:
            selectedArm = random.randint(0, 9)
        else:
            selectedArm = selectArm(epsilonPolicy)
        epsilonPolicy[selectedArm] = (numpy.random.normal(means[selectedArm], 1) + epsilonCounts[selectedArm]*epsilonPolicy[selectedArm])/(epsilonCounts[selectedArm]+1)
        epsilonCounts[selectedArm] += 1
        #Calculates the updated optimistic policy
        selectedArm = selectArm(optimisticPolicy)
        optimisticPolicy[selectedArm] = (numpy.random.normal(means[selectedArm], 1) + optimisticCounts[selectedArm]*optimisticPolicy[selectedArm])/(optimisticCounts[selectedArm]+1)
        optimisticCounts[selectedArm] += 1
        #Calculates the updated moving Epsilon-Greedy policy
        if random.random() <= epsilonValue:
            selectedArm = random.randint(0, 9)
        else:
            selectedArm = selectArm(epsilonMoving)
        movingReward = numpy.random.normal(means[selectedArm], 1)
        epsilonMoving[selectedArm] = movingReward + 0.2*(epsilonMoving[selectedArm] - movingReward)
        timeStep += 1
        #Updates means
        for i in range(len(means)):
            means[i] = 0.5*means[i] + numpy.random.normal(0, 0.0001)
    rewards = [averageReward(optimisticPolicy, means), averageReward(epsilonPolicy, means, epsilonValue), averageReward(epsilonMoving, means, epsilonValue)]
    return rewards

def singleIterationPart2c():
    #Setting up values to be used later
    #Would have been cleaner as a couple arrays of arrays but it shouldn't effect performance
    means = []
    epsilonPolicy = []
    epsilonCounts = []
    epsilonMoving = []
    epsilonValue = 0.02
    optimisticPolicy = []
    optimisticCounts = []
    #Filling policy arrays and determining averages for each one-armed bandit
    for i in range(10):
        means.append(numpy.random.normal(0, 1))
        epsilonPolicy.append(0)
        epsilonCounts.append(0)
        epsilonMoving.append(0)
        optimisticCounts.append(3)
    for i in range(10):
        optimisticPolicy.append(max(means)+3)
    timeStep = 0
    #Runs 10000 iterations of the algorithms selecting one armed bandits and receiving rewards.
    while timeStep < 10000:
        #Calculates the updated stationary Epsilon-greedy policy
        if random.random() <= epsilonValue:
            selectedArm = random.randint(0, 9)
        else:
            selectedArm = selectArm(epsilonPolicy)
        epsilonPolicy[selectedArm] = (numpy.random.normal(means[selectedArm], 1) + epsilonCounts[selectedArm]*epsilonPolicy[selectedArm])/(epsilonCounts[selectedArm]+1)
        epsilonCounts[selectedArm] += 1
        #Calculates the updated optimistic policy
        selectedArm = selectArm(optimisticPolicy)
        optimisticPolicy[selectedArm] = (numpy.random.normal(means[selectedArm], 1) + optimisticCounts[selectedArm]*optimisticPolicy[selectedArm])/(optimisticCounts[selectedArm]+1)
        optimisticCounts[selectedArm] += 1
        #Calculates the updated moving Epsilon-Greedy policy
        if random.random() <= epsilonValue:
            selectedArm = random.randint(0, 9)
        else:
            selectedArm = selectArm(epsilonMoving)
        movingReward = numpy.random.normal(means[selectedArm], 1)
        epsilonMoving[selectedArm] = movingReward + 0.9*(epsilonMoving[selectedArm] - movingReward)
        timeStep += 1
        #Updates means
        for i in range(len(means)):
            if random.random() <= 0.005:
                means = permute(means)
    rewards = [averageReward(optimisticPolicy, means), averageReward(epsilonPolicy, means, epsilonValue), averageReward(epsilonMoving, means, epsilonValue)]
    return rewards

#Creates and fills out the average reward matrix.
averages = [[], [], []]
#Runs a number of rounds equal to the value of the "iterations" variable
#Takes a few minutes at 1000 iterations, but not too long
iterations = 1000
for i in range(iterations):
    res = singleIterationPart2c()
    for j in range(len(res)):
        for k in range(3):
            averages[k].append(res[k])

#Plots average rewards for part 2.2
plt.boxplot(averages)
plt.plot()
plt.show()

#Creates and fills out the average reward matrix.
averages = [[], [], []]
#Runs a number of rounds equal to the value of the "iterations" variable
#Takes a few minutes at 1000 iterations, but not too long
for i in range(iterations):
    res = singleIterationPart2a()
    for j in range(len(res)):
        for k in range(3):
            averages[k].append(res[k])

#Plots average reward for part 2a
plt.boxplot(averages)
plt.plot()
plt.show()

#Creates and fills out the average reward matrix.
averages = [[], [], []]
#Runs a number of rounds equal to the value of the "iterations" variable
#Takes a few minutes at 1000 iterations, but not too long
for i in range(iterations):
    res = singleIterationPart2b()
    for j in range(len(res)):
        for k in range(3):
            averages[k].append(res[k])

#Plots average reward for part 2b
plt.boxplot(averages)
plt.plot()
plt.show()