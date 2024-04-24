from pysb import Rule, Model

from modules.molecules import RNA, Protein
from modules.reactioncomplex import ReactionComplex


class Toehold(ReactionComplex):
    def __init__(self, rna: RNA = None, translational_control: tuple = None, prot_name: str = None, model: Model = None):
        assert RNA is not None
        assert translational_control is not None

        sequence_name = rna.sequence_name if prot_name is None else prot_name
        protein = Protein.get_instance(sequence_name=sequence_name, model=model)

        super().__init__(substrate=rna, product=protein, model=model)

        self.toehold_name = translational_control[0]
        self.trigger_name = translational_control[1]

        # RNA.get_instance returns either the already existing instance or creates it
        trigger = RNA.get_instance(sequence_name=self.trigger_name, model=model)

        self.k_tl = self.parameters["k_tl"]
        self.k_tl_bound = self.parameters["k_tl_bound_toehold"]
        self.k_toehold_binding = self.parameters["k_trigger_binding"]
        self.k_toehold_unbinding = self.parameters["k_trigger_unbinding"]
        # self.k_tl_unbound = self.parameters["k_tl_unbound_toehold"]

        rules = []

        rule = Rule(f'TOEHOLD_{trigger.name}_binding_to_{rna.name}',
                    trigger(state="full", toehold=None) + rna(state="full", toehold=None) >>
                    trigger(state="full", toehold=1) % rna(state="full", toehold=1),
                    self.k_toehold_binding)
        rules.append(rule)
        rule = Rule(f'TOEHOLD_{trigger.name}_unbinding_from_{rna.name}',
                    trigger(state="full", toehold=1) % rna(state="full", toehold=1) >>
                    trigger(state="full", toehold=None) + rna(state="full", toehold=None),
                    self.k_toehold_unbinding)
        rules.append(rule)

        rule = Rule(f'TOEHOLD_unbound_translation_of_{rna.name}_to_{protein.name}',
                    rna(state="full", toehold=None) >> rna(state="full", toehold=None) + protein(state="immature"),
                    self.k_tl)
        rules.append(rule)

        rule = Rule(f'TOEHOLD_bound_translation_of_{rna.name}_to_{protein.name}',
                    trigger(state="full", toehold=1) % rna(state="full", toehold=1) >>
                    trigger(state="full", toehold=1) % rna(state="full", toehold=1) + protein(state="immature"),
                    self.k_tl_bound)
        rules.append(rule)

        self.rules = rules

        pass
