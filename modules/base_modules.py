from pysb import Rule, Model

from modules.reactioncomplex import ReactionComplex
from modules.molecules import RNA, Protein


class Transcription(ReactionComplex):
    def __init__(self, sequence_name: str = None, model:Model=None):
        rna = RNA.get_instance(sequence_name=sequence_name, model=model)

        super().__init__(substrate=None, product=rna, model=model)

        self.k_tx = self.parameters["k_tx"]
        self.k_deg = self.parameters["k_rna_deg"]



        rules = []
        # Transcription
        rule = Rule(f'transcription_{rna.name}',
                    None >> rna(state="full", sense=None, toehold=None),
                    self.k_tx)
        rules.append(rule)

        self.rules = rules




class Translation(ReactionComplex):
    def __init__(self, rna: RNA = None, prot_name:str=None, model:Model=None):
        sequence_name = rna.sequence_name if prot_name is None else prot_name
        protein = Protein.get_instance(sequence_name=rna.sequence_name, model=model)

        super().__init__(substrate=rna, product=protein, model=model)

        self.k_tl = self.parameters["k_tl"]
        self.k_mat = self.parameters["k_mat"]
        self.k_deg = self.parameters["k_prot_deg"]



        rules = []
        rule = Rule(f'translation_of_{rna.name}_to_{protein.name}',
                    rna(state="full") >> rna(state="full") + protein(state="immature"), self.k_tl)
        rules.append(rule)
        self.rules = rules

        pass
