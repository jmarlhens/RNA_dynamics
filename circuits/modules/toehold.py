from pysb import Rule, Model, Monomer, Observable, Initial
from circuits.modules.molecules import RNA, Protein
from circuits.modules.reactioncomplex import ReactionComplex
from circuits.modules.base_modules import KineticsType


class Toehold(ReactionComplex):
    def __init__(
        self,
        rna: RNA = None,
        translational_control: tuple = None,
        prot_name: str = None,
        model: Model = None,
        kinetics_type: KineticsType = KineticsType.MICHAELIS_MENTEN,
    ):
        assert rna is not None
        assert translational_control is not None

        sequence_name = rna.sequence_name if prot_name is None else prot_name
        protein = Protein.get_instance(sequence_name=sequence_name, model=model)

        super().__init__(substrate=rna, product=protein, model=model)

        self.toehold_name = translational_control[0]
        self.trigger_name = translational_control[1]
        self.kinetics_type = kinetics_type

        # Retrieve or create the trigger RNA instance
        trigger = RNA.get_instance(sequence_name=self.trigger_name, model=model)

        # Choose which implementation to use based on kinetics_type
        if kinetics_type == KineticsType.MICHAELIS_MENTEN:
            self.rules = self._setup_michaelis_menten(rna, trigger, protein)
        elif kinetics_type == KineticsType.MASS_ACTION:
            self.rules = self._setup_mass_action(rna, trigger, protein, model)
        else:
            raise ValueError(f"Unknown kinetics type: {kinetics_type}")

    def binding_unbinding_rna(self, rna, trigger, rules):
        # Binding rule: Trigger RNA binds to Toehold RNA at the `toehold` site when both are unbound (toehold=None)
        binding_rule = Rule(
            f"TOEHOLD_{trigger.name}_binding_to_{rna.name}",
            trigger(state="full", toehold=None) + rna(state="full", toehold=None)
            >> trigger(state="full", toehold=1) % rna(state="full", toehold=1),
            self.k_toehold_binding,
        )
        rules.append(binding_rule)

        # Unbinding rule: Trigger RNA unbinds from Toehold RNA at the `toehold` site
        unbinding_rule = Rule(
            f"TOEHOLD_{trigger.name}_unbinding_from_{rna.name}",
            trigger(state="full", toehold=1) % rna(state="full", toehold=1)
            >> trigger(state="full", toehold=None) + rna(state="full", toehold=None),
            self.k_toehold_unbinding,
        )
        rules.append(unbinding_rule)

        return rules

    def _setup_michaelis_menten(self, rna, trigger, protein):
        """
        Set up the toehold switch with the original Michaelis-Menten style kinetics.
        This preserves the original implementation's behavior.
        """
        # Parameters
        self.k_tl_unbound = self.parameters["k_tl_unbound_toehold"]
        self.k_tl_bound = self.parameters["k_tl_bound_toehold"]
        self.k_toehold_binding = self.parameters["k_trigger_binding"]
        self.k_toehold_unbinding = self.parameters["k_trigger_unbinding"]
        rules = []

        # Binding and unbinding rules
        rules = self.binding_unbinding_rna(rna, trigger, rules)

        # Translation when Toehold RNA is unbound (toehold=None)
        unbound_translation_rule = Rule(
            f"TOEHOLD_unbound_translation_of_{rna.name}_to_{protein.name}",
            rna(state="full", toehold=None)
            >> rna(state="full", toehold=None) + protein(state="immature"),
            self.k_tl_unbound,
        )
        rules.append(unbound_translation_rule)

        # Translation when Toehold RNA is bound to the Trigger (toehold=1)
        bound_translation_rule = Rule(
            f"TOEHOLD_bound_translation_of_{rna.name}_to_{protein.name}",
            trigger(state="full", toehold=1) % rna(state="full", toehold=1)
            >> trigger(state="full", toehold=1) % rna(state="full", toehold=1)
            + protein(state="immature"),
            self.k_tl_bound,
        )
        rules.append(bound_translation_rule)

        return rules

    def _setup_mass_action(self, rna, trigger, protein, model):
        """
        Set up the toehold switch with mass action kinetics.
        This implements a more mechanistic model with explicit ribosome binding.
        """
        # Add ribosome monomer if it doesn't exist
        ribosome_name = "ribosome"
        ribosome = next((m for m in model.monomers if m.name == ribosome_name), None)
        if ribosome is None:
            # The way the ribosome rules are modeled are not very accurate at the moment, the code will run but the rules don't reflect the actual binding and unbinding of the ribosome. Print a warning.
            print(
                f"Warning: Using default ribosome monomer '{ribosome_name}' in the model. These rules are not accurate."
            )
            Monomer(
                ribosome_name, ["b", "state"], {"state": ["free", "bound", "active"]}
            )
            Initial(
                model.monomers[ribosome_name](b=None, state="free"),
                model.parameters["ribosome_0"],
            )

        # Parameters for mass action kinetics
        self.k_toehold_binding = self.parameters["k_trigger_binding"]
        self.k_toehold_unbinding = self.parameters["k_trigger_unbinding"]
        self.k_ribo_bind_unbound = self.parameters.get(
            "k_tl_bind_unbound", 0.1
        )  # Default if not provided
        self.k_ribo_bind_bound = self.parameters.get(
            "k_tl_bind_bound", 1.0
        )  # Default if not provided
        self.k_ribo_cat_unbound = self.parameters.get(
            "k_tl_cat_unbound", 0.1
        )  # Default if not provided
        self.k_ribo_cat_bound = self.parameters.get(
            "k_tl_cat_bound", 1.0
        )  # Default if not provided

        rules = []

        # Rule 1 and 2: Binding and unbinding of trigger RNA to toehold RNA
        rules = self.binding_unbinding_rna(rna, trigger, rules)

        # Rule 3: Ribosome binding to unbound RNA (inefficient)
        rule3 = Rule(
            f"TOEHOLD_ribosome_binding_unbound_{rna.name}",
            model.monomers[ribosome_name](b=None, state="free")
            + rna(state="full", toehold=None, b=None)
            >> model.monomers[ribosome_name](b=1, state="bound")
            % rna(state="full", toehold=None, b=1),
            self.k_ribo_bind_unbound,
        )
        rules.append(rule3)

        # Rule 4: Translation (protein production) from unbound RNA-ribosome complex (inefficient)
        rule4 = Rule(
            f"TOEHOLD_translation_unbound_{rna.name}_to_{protein.name}",
            model.monomers[ribosome_name](b=1, state="bound")
            % rna(state="full", toehold=None, b=1)
            >> model.monomers[ribosome_name](b=None, state="free")
            + rna(state="full", toehold=None, b=None)
            + protein(state="immature"),
            self.k_ribo_cat_unbound,
        )
        rules.append(rule4)

        # Rule 5: Ribosome binding to trigger-bound RNA (efficient)
        rule5 = Rule(
            f"TOEHOLD_ribosome_binding_bound_{rna.name}",
            model.monomers[ribosome_name](b=None, state="free")
            + (trigger(state="full", toehold=1) % rna(state="full", toehold=1, b=None))
            >> model.monomers[ribosome_name](b=2, state="active")
            % (trigger(state="full", toehold=1) % rna(state="full", toehold=1, b=2)),
            self.k_ribo_bind_bound,
        )
        rules.append(rule5)

        # Rule 6: Translation (protein production) from trigger-bound RNA-ribosome complex (efficient)
        rule6 = Rule(
            f"TOEHOLD_translation_bound_{rna.name}_to_{protein.name}",
            model.monomers[ribosome_name](b=2, state="active")
            % (trigger(state="full", toehold=1) % rna(state="full", toehold=1, b=2))
            >> model.monomers[ribosome_name](b=None, state="free")
            + (trigger(state="full", toehold=1) % rna(state="full", toehold=1, b=None))
            + protein(state="immature"),
            self.k_ribo_cat_bound,
        )
        rules.append(rule6)

        # Create observables
        Observable(
            f"obs_Trigger_{rna.name}_complex",
            trigger(state="full", toehold=1) % rna(state="full", toehold=1),
        )
        Observable(
            f"obs_Ribosome_unbound_{rna.name}_complex",
            model.monomers[ribosome_name](b=1, state="bound")
            % rna(state="full", toehold=None, b=1),
        )
        Observable(
            f"obs_Ribosome_Trigger_{rna.name}_complex",
            model.monomers[ribosome_name](b=2, state="active")
            % (trigger(state="full", toehold=1) % rna(state="full", toehold=1, b=2)),
        )

        return rules
