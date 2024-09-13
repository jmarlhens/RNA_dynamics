from pysb import Rule, Model
from modules.reactioncomplex import ReactionComplex
from modules.molecules import RNA, Protein


class Sequestration(ReactionComplex):
    def __init__(self, species1, species2, model: Model):
        """
        Sequestration reaction between two species involving binding and unbinding.

        :param species1: The first species (e.g., RNA, Protein).
        :param species2: The second species (e.g., RNA, Protein).
        :param model: The PySB model where the rules will be added.
        """
        super().__init__(substrate=species1, product=species2, model=model)

        # Define binding and unbinding rates
        self.k_bind = self.parameters.get("k_sequestration_bind", 1)  # Default rate for binding
        self.k_unbind = self.parameters.get("k_sequestration_unbind", 0.1)  # Default rate for unbinding

        # Define binding and unbinding rules
        binding_rule = Rule(
            f'{species1.name}_binds_{species2.name}',
            species1(sequestration="free") + species2(sequestration="free") >>
            species1(sequestration="bound") % species2(sequestration="bound"),
            self.k_bind
        )

        unbinding_rule = Rule(
            f'{species1.name}_unbinds_{species2.name}',
            species1(sequestration="bound") % species2(sequestration="bound") >>
            species1(sequestration="free") + species2(sequestration="free"),
            self.k_unbind
        )

        # Add rules to the model
        model.add_component(binding_rule)
        model.add_component(unbinding_rule)
