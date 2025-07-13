from pysb import Rule, Model
from circuits.modules.reactioncomplex import ReactionComplex
from circuits.modules.molecules import RNA


class Sequestration(ReactionComplex):
    def __init__(self, species1_name, species2_name, model: Model):
        """
        Sequestration reaction between two species involving binding and unbinding.

        :param species1_name: The name of the first species (e.g., RNA).
        :param species2_name: The name of the second species (e.g., RNA).
        :param model: The PySB model where the rules will be added.
        """
        # Retrieve instances of the species
        species1 = RNA.get_instance(sequence_name=species1_name, model=model)
        species2 = RNA.get_instance(sequence_name=species2_name, model=model)

        # Initialize parent class with the model only
        super().__init__(model=model)

        # Define binding and unbinding rates
        self.k_bind = model.parameters.get(
            "k_sequestration_bind", 1
        )  # Default rate for binding
        self.k_unbind = model.parameters.get(
            "k_sequestration_unbind", 0.1
        )  # Default rate for unbinding

        # Determine the site to use for binding based on the regulation type
        # dont like the way it is, needs to change later
        if "star" in species1.name.lower() or "star" in species2.name.lower():
            binding_site = "sense"
            rule_suffix = "_sense"
        elif "trigger" in species1.name.lower() or "trigger" in species2.name.lower():
            binding_site = "toehold"
            rule_suffix = "_toehold"
        else:
            raise ValueError(
                f"Neither sense nor toehold keyword found in species names: {species1_name}, {species2_name}"
            )

        # Define binding and unbinding rules using the selected site
        binding_rule = Rule(
            f"{species1.name}_binds_{species2.name}{rule_suffix}",
            species1(state="full", **{binding_site: None})
            + species2(state="full", **{binding_site: None})
            >> species1(state="full", **{binding_site: 1})
            % species2(state="full", **{binding_site: 1}),
            self.k_bind,
        )

        unbinding_rule = Rule(
            f"{species1.name}_unbinds_{species2.name}{rule_suffix}",
            species1(state="full", **{binding_site: 1})
            % species2(state="full", **{binding_site: 1})
            >> species1(state="full", **{binding_site: None})
            + species2(state="full", **{binding_site: None}),
            self.k_unbind,
        )

        # Add rules to the model
        model.add_component(binding_rule)
        model.add_component(unbinding_rule)
