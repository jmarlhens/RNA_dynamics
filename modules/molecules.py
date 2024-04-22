from pysb import Monomer, Rule


class RNA(Monomer):
    def __init__(self, sequence_name, model):
        super().__init__(name="RNA_" + sequence_name, sites=["sense", "toehold", "state"],
                         site_states={"state": {"full", "partial", "init"}})

        self.sequence_name = sequence_name
        k_rna_deg = model.parameters["k_rna_deg"]
        rule = Rule(f'RNA_degradation_{self.name}', self() >> None, k_rna_deg)


class Protein(Monomer):
    def __init__(self, sequence_name, model):
        super().__init__(name="Protein_" + sequence_name, sites=["state"],
                         site_states={"state": {"mature", "immature"}})

        self.sequence_name = sequence_name

        k_prot_deg = model.parameters["k_prot_deg"]
        rule = Rule(f'Protein_degradation_{self.name}', self() >> None, k_prot_deg)
