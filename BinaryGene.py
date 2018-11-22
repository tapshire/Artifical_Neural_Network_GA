class BinaryGene:

    def __init__(self):
        self.gene = [];
        self.fitness = 0;

    def set_gene(self, value):
        self.gene.append(value);

    def set_fitness(self, quality):
        self.fitness = self.fitness + quality;

    def get_gene(self):
        return self.gene;

    def get_fitness(self):
        return self.fitness;

    def update_gene(self, index, value):
        self.gene[index] = self.gene[index] + value