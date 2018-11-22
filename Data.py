class Data:

    def __init__(self):
        self.variables = []
        self.classification = 0

    def set_var(self, value):
        self.variables.append(float(value))

    def get_var(self):
        return self.variables

    def update_var(self, index, val):
        self.variables[index] = val

    def get_classification(self):
        return int(self.classification)