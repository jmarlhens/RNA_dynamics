from pysb import Rule, Model

from modules.reactioncomplex import ReactionComplex
from modules.molecules import RNA


class Csy4Activity(ReactionComplex):
    def __init__(self, rna: RNA = None, product_rna_names: [str] = None, model: Model = None):
        products = [RNA.get_instance(sequence_name=seq_name, model=model) for seq_name in product_rna_names]

        super().__init__(substrate=rna, product=products, model=model)

        self.k_csy4 = self.parameters["k_csy4"]

        rules = []
        rule = Rule(f'RNA_Cleavage_{rna.name}_to_{"_and_".join([prod_rna.name for prod_rna in products])}',
                    rna(state="full") >> sum(
                        [prod_rna(state="full", sense=None, toehold=None) for prod_rna in products], None),
                    self.k_csy4)
        rules.append(rule)
        self.rules = rules
