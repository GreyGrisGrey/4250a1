import numpy
import random
from math import exp
import matplotlib.pyplot as plt

#Greedily selects a one-armed bandit given a policy
def selectArm(policy):
    selectedArm = random.randint(0, (policy.count(max(policy))-1))
    for i in range(len(policy)):
        if selectedArm == 0 and policy[i] == max(policy):
            return i
        elif policy[i] == max(policy):
            selectedArm -= 1

#Greedily selects a one-armed bandit given a policy, but for the gradient algorithm
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

#Given a policy and the index of the optimal one armed bandit, calculates the probability that the algorithm will make the right choice.
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
    #Lower values of epsilon seem more effective in the long run, as after the algorithm has found the optimal value it deviates less
    #Does make it find the optimal selection slower however
    #Setting alpha too low seems to cause the gradient algorithm to have some troubles finding the optimal selection
    #Setting it too high makes it jump around a lot, but if it isn't excessively high it finds the path eventually.
    #Theres a possible bug where the non-optimistic algorithm never finds the optimal selection on any runs, that may have just been a very unlucky attempt though as it only happened once in numerous trials

    #Setting up values to be used later
    #Would have been cleaner as a couple arrays of arrays but it shouldn't effect performance
    means = []
    nonOptimisticPolicy = []
    nonOptimisticCounts = []
    epsilonPolicy = []
    epsilonCounts = []
    epsilonValue = 0.02
    optimisticPolicy = []
    optimisticCounts = []
    gradiantPolicy = []
    gradiantRewardSum = 0
    gradiantCount = 1
    gradiantZeroCheck = True
    gradiantAlpha = 0.05
    #Filling policy arrays and determining averages for each one-armed bandit
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
    #Determing the optimal one-armed bandit
    for i in range(len(means)):
        if max(means) == means[i]:
            optimalChoice = i
    rewards = []
    probabilities = []
    timeStep = 0
    #Runs 10000 iterations of the algorithms selecting one armed bandits and receiving rewards.
    while timeStep < 10000:
        selectedArm, eSum = selectGradiant(gradiantPolicy)
        #Updates the average rewards and average probabilities for a given time step
        rewards.append([averageReward(nonOptimisticPolicy, means), averageReward(optimisticPolicy, means), averageReward(epsilonPolicy, means, epsilonValue)])
        probabilities.append([optimalProbability(optimalChoice, nonOptimisticPolicy), optimalProbability(optimalChoice, optimisticPolicy), optimalProbability(optimalChoice, epsilonPolicy, epsilonValue), optimalProbability(optimalChoice, gradiantPolicy, False, eSum)])
        #Calculates the updated non-optimistic policy
        selectedArm = selectArm(nonOptimisticPolicy)
        nonOptimisticPolicy[selectedArm] = (numpy.random.normal(means[selectedArm], 1) + nonOptimisticCounts[selectedArm]*nonOptimisticPolicy[selectedArm])/(nonOptimisticCounts[selectedArm]+1)
        nonOptimisticCounts[selectedArm] += 1
        #Calculates the updated Epsilon-greedy policy
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
        #Calculates the updated gradient policy
        selectedArm, eSum = selectGradiant(gradiantPolicy)
        gradiantReward = numpy.random.normal(means[selectedArm], 1)
        gradiantPolicy[selectedArm] = gradiantPolicy[selectedArm] + gradiantAlpha*(gradiantReward - (gradiantRewardSum/gradiantCount))*(1-(exp(gradiantPolicy[selectedArm])/eSum))
        for i in range(10):
            if i != selectedArm:
                gradiantPolicy[i] = gradiantPolicy[i] - gradiantAlpha*(gradiantReward - (gradiantRewardSum/gradiantCount))*(exp(gradiantPolicy[i])/eSum)
        gradiantRewardSum += gradiantReward
        #This is just to prevent a divide-by-zero error on the first iteration
        if gradiantZeroCheck:
            gradiantZeroCheck = False
        else:
            gradiantCount += 1
        timeStep += 1
    return rewards, probabilities


#Creates and fills out the probability and average reward matrices.
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
#Runs a number of rounds equal to the value of the "iterations" variable
#Takes a few minutes at 1000 iterations, but not too long
iterations = 1000
for i in range(iterations):
    res = singleIterationPart1()
    for j in range(len(res[0])):
        for k in range(3):
            averages[k][j] += res[0][j][k]
        for k in range(4):
            probabilities[k][j] += res[1][j][k]
    if i%10 == 0:
        print(i)
#Divides the average rewards and the probability of selecting by the number of iterations, to make the graphs look nicer
#I tried doing this in the above step, but it took forever.
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