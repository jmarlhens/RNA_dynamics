from pysb import Monomer, Rule, Model, Parameter


class MyMonomer(Monomer):
    prefix = "MyMonomer_"

    @classmethod
    def sequence_name_to_name(cls, sequence_name: str) -> str:
        name = cls.prefix + sequence_name
        return name

    @classmethod
    def get_instance(
        cls, sequence_name: str, model: Model, kinetic_parameters: dict = None
    ) -> "MyMonomer":
        """
        Retrieve an instance of the monomer based on its sequence name.
        If the monomer does not exist in the model, create a new instance.
        :param sequence_name: The sequence name of the monomer.
        :param model: The model to check for existing monomers.
        :param kinetic_parameters: Kinetic parameters for the monomer, if needed.
        :return: An instance of the monomer.
        """
        name = cls.prefix + sequence_name

        species = model.monomers.keys()
        if name in species:
            monomer = model.monomers[name]
        else:
            monomer = cls(sequence_name, model, kinetic_parameters)

        return monomer


class RNA(MyMonomer):
    prefix = "RNA_"

    def __init__(
        self, sequence_name: str, model: Model, kinetic_parameters: dict = None
    ):
        name = RNA.sequence_name_to_name(sequence_name)
        # Separate binding sites for sense (STAR regulation) and toehold (Toehold regulation)
        super().__init__(
            name=name,
            sites=["b", "sense", "toehold", "state"],
            site_states={
                "state": {"full", "partial", "init"},
            },
        )

        self.sequence_name = sequence_name

        # Add degradation later
        # k_rna_deg = model.parameters["k_rna_deg"]
        #
        # # Add degradation rule for the RNA
        # rule_name_degradation = f'RNA_degradation_{self.name}'
        # degradation_rule = Rule(rule_name_degradation, self() >> None, k_rna_deg)
        #
        # # Add the degradation rule to the model
        # model.add_component(degradation_rule)


class Protein(MyMonomer):
    prefix = "Protein_"

    def __init__(self, sequence_name: str, model: Model, kinetic_parameters: dict):
        """
        Initialize a Protein monomer with maturation and degradation rules.

        :param sequence_name:
        :param model:
        :param kinetic_parameters:
        """
        name = Protein.sequence_name_to_name(sequence_name)
        super().__init__(
            name=name, sites=["state"], site_states={"state": {"mature", "immature"}}
        )

        self.sequence_name = sequence_name
        protein_parameters = ["k_mat", "k_prot_deg"]
        existing_parameters = set(model.parameters.keys())

        for param_name in protein_parameters:
            if param_name not in existing_parameters:
                Parameter(param_name, kinetic_parameters[param_name])

        rule_name_maturation = f"maturation_{self.name}"
        rule_name_degradation = f"Protein_degradation_{self.name}"

        # Define maturation and degradation rules for the protein
        maturation_rule = Rule(
            rule_name_maturation,
            self(state="immature") >> self(state="mature"),
            model.parameters["k_mat"],
        )
        degradation_rule = Rule(
            rule_name_degradation, self() >> None, model.parameters["k_prot_deg"]
        )

        # Add rules to the model
        model.add_component(maturation_rule)
        model.add_component(degradation_rule)
