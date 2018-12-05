import random
import math
from array import *
from BinaryGene import BinaryGene
from Rules import Rules
from Data import Data
from copy import deepcopy
import csv



# Performs one point crossover with random point selection
def crossover(population, size, len, data, cond_len):
    print("Crossover")
    for i in range(size):
        crossover = random.random()
        if i % 2 is 1 and crossover > 0.6:
            crossover_pt = random.randint(0, len - 1)
            #population[i].set_fitness(0)
            #population[i - 1].set_fitness(0)
            for j in range(len):
                if j >= crossover_pt:
                    parent_one = population[i - 1].get_gene()[j]
                    parent_two = population[i].get_gene()[j]
                    population[i - 1].update_gene(j, parent_two)
                    population[i].update_gene(j, parent_one)
            #fitness_function(population[i - 1], cond_len data)
            #fitness_function(population[i], cond_len, rulebase, data)
    return population


# Performs mutation with random probability for each gene bit flip
def mutation(population, size, len, data, cond_len):
    #print("Mutation")
    for i in range(size):
        population[i].set_fitness(0)
        for j in range(len):
            mutate = random.random()
            if mutate < 0.5:
                choice = random.uniform(-0.1, 0.1)
                #print(choice)
                population[i].update_gene(j, (choice))
    # for i in range(size):
    #     fitness_function(population[i], cond_len, rulebase, data)
    return population


def match(d, r):
    for i in range(len(d)):
        if d[i] is not r[i] and r[i] is not 2:
            return False
            break
    return True


# Tournament selection to determine new offspring for new generation pool
def tournament_selection(population, size):
    #print("Tournament Selection")
    offspring = []
    for i in range(size):
        parent_one = random.randrange(0, size)
        parent_two = random.randrange(0, size)
        if population[parent_one].get_fitness() > population[parent_two].get_fitness():
            p1 = deepcopy(population[parent_one])
            offspring.append(p1)
        else:
            p2 = deepcopy(population[parent_two])
            offspring.append(p2)
        population[i] = offspring[i]
    return population


# initialises the chromosomes for the start pool for first generation
def __init__chromosomes(size, len, cond_len):
    population = [BinaryGene() for i in range(size)]
    for i in range(size):
        population[i].set_fitness(0)
        for j in range(len):
            population[i].set_gene(random.uniform(0, 1))
    return population


# initilises the rule base list creating list size and rule size
def __init__rules(len, num_rule):
    rulebase = [Rules() for i in range(num_rule)]
    for i in range(num_rule):
        for j in range(len):
            rulebase[i].set_cond(0)
    return rulebase


# initilises the data set by reading in text file and creating list of all data
def __init__data(file, len, data_len):
    k = 0
    data = [Data() for i in range(data_len)]
    file_name = open(file, 'r', newline='')
    next(file_name)
    next(file_name)
    for line in file_name:
        line = line.strip('\n')
        line = line.strip('\r')
        temp = ""
        j = 0
        for i in line:
            if(i is not " "):
                temp = temp + i
                if j is 63:
                    data[k].classification = int(temp)
                    temp = ""
            else:
                data[k].set_var(float(temp))
                temp = ""
            j = j + 1
        #print(str(data[k].get_var()) + " " + str(data[k].classification))
        k = k + 1
    return data


# calcualtes mean, max and sum of current population
def fitness(population, size, max):
    sum = 0
    for i in range(size):
        if population[i].get_fitness() >= max:
            max = population[i].get_fitness()
        sum = sum + population[i].get_fitness()
    mean = sum / size
    print("MEAN: " + str(mean))
    print("MAX: " + str(max))
    return mean, max


# Fitness function to determine fitness of current candidate
def fitness_function(data, sig):
    fitness = 0
    #(data.get_classification())
    #print(sig)
    if data.get_classification() is 1 and sig >= 0.5:
        fitness = 1
    if data.get_classification() is 0 and sig < 0.5:
        fitness = 1
    #print("Fitness: " + str(fitness))
    return fitness


def train_network(train_data, genomes, i_n, h_n, o_n):
    input = []
    in_hid = []
    hid_out = []
    n_hid = [0,0,0,0,0]
    n_out = [0]
    temp = []
    bias = 1
    bias_hid = []
    bias_out = []
    k = 0
    j = 0

    for g in range(len(genomes)):
        tally = 0
        # print("---------")
        # print(genome.get_fitness())
        # print("----------")
        k = 0
        # single genome
        for j in range(len(genomes[g].get_gene())):
            # check gene is less than number of weights between input and hidden and create list in list of each weight from each input
            if j < (i_n * h_n):
                temp.append(genomes[g].get_gene()[j])
                k = k + 1
                if k is h_n:
                    in_hid.append(deepcopy(temp))
                    temp.clear()
                    k = 0
            # check gene is equal or greater than the number of input to hidden weights needed and create list of weights from hidden to output node
            if j >= i_n * h_n and len(hid_out) is not (h_n * o_n):
                hid_out.append(deepcopy(genomes[g].get_gene()[j]))
            # create list of bias to hidden weights
            if j >= (i_n * h_n) + (h_n * o_n):
                if len(bias_hid) < h_n:
                    bias_hid.append(genomes[g].get_gene()[j])
                else:
                    bias_out.append(genomes[g].get_gene()[j])
        # find the sum between the input and hidden nodes and save to hidden node list
        count = 0
        for d in range(len(train_data)):
            for i in range(len(train_data[d].get_var())):
                input.append(train_data[d].get_var()[i])
            for j in range(i_n):
                for k in range(h_n):
                    n_hid[k] = n_hid[k] + (input[j] * in_hid[j][k])
            for j in range(h_n):
                n_hid[j] = n_hid[j] + (bias * bias_hid[j])
            for j in range(h_n):
                n_hid[j] = sigmoid(n_hid[j])
            count = 0
            while (count < h_n):
                n_out[0] = n_out[0] + (n_hid[count] * hid_out[count])
                count = count + 1
            for j in range(o_n):
                n_out[j] = n_out[j] + (bias * bias_out[j])
            #print(n_out[0])
            tally = fitness_function(train_data[d], sigmoid(n_out[0]))
            input.clear()
            n_hid = [0, 0, 0, 0, 0]
            n_out = [0]
            genomes[g].set_fitness(tally)

        in_hid.clear()
        hid_out.clear()
        bias_out.clear()
        bias_hid.clear()


def test_network(test_data, i_n, h_n, o_n, population, best):
    input = []
    in_hid = []
    hid_out = []
    n_hid = [0, 0, 0, 0, 0]
    n_out = [0]
    temp = []
    bias = 1
    bias_hid = []
    bias_out = []

    for g in range(len(population)):
        tally = 0
        if population[g].get_fitness() is best:
            k = 0
            # single genome
            for j in range(len(population[g].get_gene())):
                # check gene is less than number of weights between input and hidden and create list in list of each weight from each input
                if j < (i_n * h_n):
                    temp.append(population[g].get_gene()[j])
                    k = k + 1
                    if k is h_n:
                        in_hid.append(deepcopy(temp))
                        temp.clear()
                        k = 0
                # check gene is equal or greater than the number of input to hidden weights needed and create list of weights from hidden to output node
                if j >= i_n * h_n and len(hid_out) is not (h_n * o_n):
                    hid_out.append(deepcopy(population[g].get_gene()[j]))
                # create list of bias to hidden weights
                if j >= (i_n * h_n) + (h_n * o_n):
                    if len(bias_hid) < h_n:
                        bias_hid.append(population[g].get_gene()[j])
                    else:
                        bias_out.append(population[g].get_gene()[j])
            # find the sum between the input and hidden nodes and save to hidden node list
            count = 0
            for data in test_data:
                for i in range(len(data.get_var())):
                    input.append(data.get_var()[i])
                for j in range(i_n):
                    for k in range(h_n):
                        n_hid[k] = deepcopy(n_hid[k] + (input[j] * in_hid[j][k]))
                for j in range(h_n):
                    n_hid[j] = deepcopy(n_hid[j] + (bias * bias_hid[j]))
                for j in range(h_n):
                    n_hid[j] = sigmoid(n_hid[j])
                count = 0
                while (count < h_n):
                    n_out[0] = deepcopy(n_out[0] + (n_hid[count] * hid_out[count]))
                    count = count + 1
                for j in range(o_n):
                    n_out[j] = deepcopy(n_out[j] + (bias * bias_out[j]))
                tally = tally + fitness_function(data, sigmoid(n_out[0]))
                input.clear()
                n_hid = [0, 0, 0, 0, 0]
                n_out = [0]
            print(tally)
            print((tally / len(test_data)) * 100)
        in_hid.clear()
        hid_out.clear()
        bias_out.clear()
        bias_hid.clear()


def sigmoid(w_sum):
    return 1 / (1 + math.exp(-w_sum))


# main program to run GA
def main():
    # ANN variables
    train = []
    test = []
    in_nodes = 7
    hid_nodes = 5
    out_nodes = 1


    #GA variables
    var = 100
    pop_size = var
    mean = []
    cond_len = 7
    data_len = 2000
    chromosome_len = (in_nodes * hid_nodes) + (hid_nodes * out_nodes) + (hid_nodes + out_nodes)
    generations = 5000

    data_set = __init__data("data3.txt", cond_len, data_len).copy()
    for i in range(data_len):
        if i % 2 is 0:
            test.append(deepcopy(data_set[i]))
        else:
            train.append(deepcopy(data_set[i]))
    population_obj = __init__chromosomes(pop_size, chromosome_len, cond_len).copy()

    best = 0
    temp_best = population_obj[0]

    for i in range(generations):

        print("GENERATION " + str(i + 1))
        population_obj = tournament_selection(population_obj, pop_size).copy()
        #population_obj = crossover(population_obj, pop_size, chromosome_len, data_set, cond_len).copy()
        population_obj = mutation(population_obj, pop_size, chromosome_len, data_set, cond_len).copy()
        train_network(train, population_obj, in_nodes, hid_nodes, out_nodes)
        # Maintain best fitness gene for each generation

        for g in range(len(population_obj)):
            if population_obj[g].get_fitness() >= best:
                best = population_obj[g].get_fitness()
                temp_best = deepcopy(population_obj[g])
        worst = best
        for g in range(len(population_obj)):
            if population_obj[g].get_fitness() < worst:
                worst = population_obj[g].get_fitness()
        for j in range(len(population_obj)):
            if population_obj[j].get_fitness() is worst:
                population_obj[j] = deepcopy(temp_best)
                break


        # # Maintain best fitness gene for each generation
        # worst = population_obj[0].get_fitness()
        # for gene in population_obj:
        #     if gene.get_fitness() >= worst:
        #         worst = deepcopy(gene.get_fitness())
        # for gene in population_obj:
        #     if gene.get_fitness() <= temp_best.get_fitness():
        #         temp_best = deepcopy(gene)
        # for j in range(len(population_obj)):
        #     if population_obj[j].get_fitness() is worst:
        #         population_obj[j] = deepcopy(temp_best)
        #         break

        val, val_2 = fitness(population_obj, pop_size, temp_best.get_fitness())


        if (i + 1) % 10 is 0:
            test_network(train, in_nodes, hid_nodes, out_nodes, population_obj, temp_best.get_fitness())

        mean.append(val)
        mean.append(val_2)
        for genome in population_obj:
            genome.fitness = 0

    with open('genetic_algorithm.csv', 'w', newline='') as new:
        writer = csv.writer(new)
        writer.writerow(["Generation", "Mean", "Best"])
        count = 0
        for i in range(generations * 2):
            if i % 2 is 1:
                count = count + 1
                temp = []
                temp.append(count)
                temp.append(mean[i - 1])
                temp.append(mean[i])
                writer.writerow(temp)
    print("made it")
    for genome in population_obj:
        if genome.get_fitness() is temp_best.get_fitness():
            print(genome.get_gene())

    test_network(train, in_nodes, hid_nodes, out_nodes, population_obj, val_2)
    test_network(test, in_nodes, hid_nodes, out_nodes, population_obj, val_2)


main()
