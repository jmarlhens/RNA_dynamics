from pysb import Rule, Model, Expression
from modules.reactioncomplex import ReactionComplex
from modules.molecules import RNA, Protein

class Transcription(ReactionComplex):
    def __init__(self, sequence_name: str = None, model: Model = None):
        rna = RNA.get_instance(sequence_name=sequence_name, model=model)

        super().__init__(substrate=None, product=rna, model=model)

        self.k_tx = self.parameters["k_tx"]
        self.k_concentration = self.parameters["k_" + sequence_name + "_concentration"]
        Expression('k_tx_plasmid_' + sequence_name, self.k_concentration * self.k_tx)
        self.k_deg = self.parameters["k_rna_deg"]

        rules = []
        # Transcription rule: RNA is produced in the unbound state
        rule = Rule(
            f'transcription_{rna.name}',
            None >> rna(state="full", sense=None, toehold=None),
            model.expressions['k_tx_plasmid_' + sequence_name]
        )
        rules.append(rule)

        self.rules = rules

class Translation(ReactionComplex):
    def __init__(self, rna: RNA = None, prot_name: str = None, model: Model = None):
        sequence_name = rna.sequence_name if prot_name is None else prot_name
        protein = Protein.get_instance(sequence_name=sequence_name, model=model)

        super().__init__(substrate=rna, product=protein, model=model)

        self.k_tl = self.parameters["k_tl"]
        self.k_mat = self.parameters["k_mat"]
        self.k_deg = self.parameters["k_prot_deg"]

        rules = []

        # Translation rule: Translation occurs only if RNA is not bound at any site
        rule = Rule(
            f'translation_of_{rna.name}_to_{protein.name}',
            rna(state="full") >>
            rna(state="full") + protein(state="immature"),
            self.k_tl
        )
        rules.append(rule)

        self.rules = rules
