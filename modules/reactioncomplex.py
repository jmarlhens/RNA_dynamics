from pysb import Monomer


class ReactionComplex:

    def __init__(self, substrate: Monomer = None, product: Monomer = None, model=None):
        self.substrate = substrate
        self.product = product

        if model is None:
            raise Exception("model is None")

        self.model = model
        self.parameters = model.parameters
        pass
