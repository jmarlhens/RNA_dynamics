from pysb import Rule, Model
from modules.reactioncomplex import ReactionComplex
from modules.molecules import RNA

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
        self.k_bind = model.parameters.get("k_sequestration_bind", 1)  # Default rate for binding
        self.k_unbind = model.parameters.get("k_sequestration_unbind", 0.1)  # Default rate for unbinding

        # Define binding and unbinding rules using the unified `binding` site
        binding_rule = Rule(
            f'{species1.name}_binds_{species2.name}',
            species1(binding=None, state="full") + species2(binding=None, state="full") >>
            species1(binding=1, state="full") % species2(binding=1, state="full"),
            self.k_bind
        )

        unbinding_rule = Rule(
            f'{species1.name}_unbinds_{species2.name}',
            species1(binding=1, state="full") % species2(binding=1, state="full") >>
            species1(binding=None, state="full") + species2(binding=None, state="full"),
            self.k_unbind
        )

        # Add rules to the model
        model.add_component(binding_rule)
        model.add_component(unbinding_rule)

        # Debugging prints to ensure correct rule integration
        print(f"Added binding rule: {binding_rule}")
        print(f"Added unbinding rule: {unbinding_rule}")
