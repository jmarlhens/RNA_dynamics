from pysb import Monomer, Rule, Model


class MyMonomer(Monomer):
    prefix = "MyMonomer_"

    @classmethod
    def sequence_name_to_name(cls, sequence_name: str) -> str:
        name = cls.prefix + sequence_name
        return name

    @classmethod
    def get_instance(cls, sequence_name: str, model: Model):
        name = cls.prefix + sequence_name

        species = model.monomers.keys()
        if name in species:
            monomer = model.monomers[name]
        else:
            monomer = cls(sequence_name, model)

        return monomer


class RNA(MyMonomer):
    prefix = "RNA_"

    def __init__(self, sequence_name: str, model: Model):
        name = RNA.sequence_name_to_name(sequence_name)
        # Separate binding sites for sense (STAR regulation) and toehold (Toehold regulation)
        super().__init__(name=name, sites=["b", "sense", "toehold", "state"],
                         site_states={
                             "state": {"full", "partial", "init"},
                         })

        self.sequence_name = sequence_name

        k_rna_deg = model.parameters["k_rna_deg"]

        # Add degradation rule for the RNA
        rule_name_degradation = f'RNA_degradation_{self.name}'
        degradation_rule = Rule(rule_name_degradation, self() >> None, k_rna_deg)

        # Add the degradation rule to the model
        model.add_component(degradation_rule)


class Protein(MyMonomer):
    prefix = "Protein_"

    def __init__(self, sequence_name: str, model: Model):
        name = Protein.sequence_name_to_name(sequence_name)
        super().__init__(name=name, sites=["state"],
                         site_states={"state": {"mature", "immature"}})

        self.sequence_name = sequence_name

        self.k_mat = model.parameters["k_mat"]
        self.k_prot_deg = model.parameters["k_prot_deg"]

        rule_name_maturation = f'maturation_{self.name}'
        rule_name_degradation = f'Protein_degradation_{self.name}'

        # Define maturation and degradation rules for the protein
        maturation_rule = Rule(rule_name_maturation, self(state="immature") >> self(state="mature"), self.k_mat)
        degradation_rule = Rule(rule_name_degradation, self() >> None, self.k_prot_deg)

        # Add rules to the model
        model.add_component(maturation_rule)
        model.add_component(degradation_rule)
