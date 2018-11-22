class Rules:

    def __init__(self):
        self.cond = []
        self.out = 0

    def set_cond(self, value):
        self.cond.append(value)

    def get_cond(self):
        return self.cond

    def update_cond(self, index, val):
        self.cond[index] = val

    def get_out(self):
        return self.out

