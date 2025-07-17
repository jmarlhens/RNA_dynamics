from pysb import Rule, Model, Parameter
from circuits.modules.reactioncomplex import ReactionComplex
from circuits.modules.molecules import RNA
from circuits.modules.base_modules import KineticsType


class Csy4Activity(ReactionComplex):
    def __init__(
        self,
        rna: RNA = None,
        product_rna_names: [str] = None,
        model: Model = None,
        kinetic_parameters: dict = None,
        kinetics_type: KineticsType = KineticsType.MICHAELIS_MENTEN,
    ):
        # Retrieve RNA products based on provided names
        products = [
            RNA.get_instance(sequence_name=seq_name, model=model)
            for seq_name in product_rna_names
        ]

        super().__init__(substrate=rna, product=products, model=model)

        existing_parameters = set(model.parameters.keys())
        if "k_csy4" not in existing_parameters:
            Parameter("k_csy4", kinetic_parameters["k_csy4"])

        self.k_csy4 = self.parameters["k_csy4"]

        rules = []
        if kinetics_type == KineticsType.MICHAELIS_MENTEN:
            # Cleavage rule: RNA is cleaved into products, I removed the binding to sense/toehold. it shouldn;t influence the cleavage
            rule = Rule(
                f"RNA_Cleavage_{rna.name}_to_{'_and_'.join([prod_rna.name for prod_rna in products])}",
                rna(state="full")
                >> sum(
                    [
                        prod_rna(state="full", sense=None, toehold=None, b=None)
                        for prod_rna in products
                    ],
                    None,
                ),
                model.parameters["k_csy4"],
            )
        elif kinetics_type == KineticsType.MASS_ACTION:
            rule = Rule(
                f"RNA_Cleavage_{rna.name}_to_{'_and_'.join([prod_rna.name for prod_rna in products])}",
                rna(state="full")
                >> sum(
                    [
                        prod_rna(state="full", sense=None, toehold=None, b=None)
                        for prod_rna in products
                    ],
                    None,
                ),
                model.parameters["k_csy4"],
            )
        else:
            raise ValueError(f"Unknown kinetics type: {kinetics_type}")

        rules.append(rule)

        # Add the rules to the model
        self.rules = rules
