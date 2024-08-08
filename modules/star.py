from pysb import Rule, Model, Expression
from modules.molecules import RNA
from modules.reactioncomplex import ReactionComplex


class STAR(ReactionComplex):
    def __init__(self, sequence_name: str = None, transcriptional_control: tuple = None, model: Model = None):
        assert sequence_name is not None
        assert transcriptional_control is not None
        rna = RNA.get_instance(sequence_name=sequence_name, model=model)

        super().__init__(substrate=None, product=rna, model=model)

        self.sense_name = transcriptional_control[0]
        self.star_name = transcriptional_control[1]

        self.rna_name = rna.sequence_name
        self.regulator_name = self.star_name

        # RNA.get_instance returns either the already existing instance or creates the desired
        regulated = rna  # RNA.get_instance(sequence_name=self.rna_name, model=model)
        regulator = RNA.get_instance(sequence_name=self.regulator_name, model=model)

        self.k_init = self.parameters["k_tx_init"]
        self.k_concentration = self.parameters["k_" + sequence_name + "_concentration"]
        Expression('k_tx_plasmid_' + sequence_name, self.k_concentration * self.k_init)
        self.k_bind = self.parameters["k_star_bind"]
        self.k_unbind = self.parameters["k_star_unbind"]
        self.k_act = self.parameters["k_star_act"]
        self.k_act_reg = self.parameters["k_star_act_reg"]
        self.k_stop = self.parameters["k_star_stop"]
        self.k_stop_reg = self.parameters["k_star_stop_reg"]
        self.k_deg = self.parameters["k_rna_deg"]

        rules = []
        # ToDo Needs to be reworked significantly

        # Define the Rules for the RNA dynamics
        # Initiation of RNA transcription
        rule = Rule('STAR_RNA_transcription_initiation_%s' % (regulated.name),
                    None >> regulated(state='init', sense=None, toehold=None), model.expressions['k_tx_plasmid_' + sequence_name])
        rules.append(rule)

        # Binding of RNA regulator to the early transcript
        rule = Rule('STAR_RNA_regulator_binding_%s_%s' % (regulated.name, regulator.name),
                    regulated(sense=None) + regulator(state='full', sense=None)
                    >> regulated(sense=1) % regulator(state='full', sense=1), self.k_bind)
        rules.append(rule)

        # Unbinding of Full and Early RNA transcript and RNA regulator (slow reaction)
        rule = Rule('STAR_RNA_regulator_unbinding_full_%s_%s' % (regulated.name, regulator.name),
                    regulated(sense=1) % regulator(state='full', sense=1)
                    >> regulated(sense=None) + regulator(state='full', sense=None), self.k_unbind)
        rules.append(rule)

        # Activation of Transcription (RNA activator promotes transcription)
        rule = Rule('STAR_RNA_full_transcription_reg_%s_%s' % (regulated.name, regulator.name),
                    regulated(state='init', sense=1) % regulator(state='full', sense=1)
                    >> regulated(state='full', sense=1) % regulator(state='full', sense=1), self.k_act_reg)
        rules.append(rule)
        rule = Rule('STAR_RNA_full_transcription_%s' % (regulated.name),
                    regulated(state='init', sense=None) >> regulated(state='full', sense=None), self.k_act)
        rules.append(rule)

        # Stop of Transcription (RNA inhibitor stops transcription)
        rule = Rule('STAR_RNA_partial_transcription_reg_%s_%s' % (regulated.name, regulator.name),
                    regulated(state='init', sense=1) % regulator(state='full', sense=1)
                    >> regulated(state='partial', sense=1) % regulator(state='full', sense=1), self.k_stop_reg)
        rules.append(rule)
        rule = Rule('STAR_RNA_partial_transcription_%s' % (regulated.name),
                    regulated(state='init', sense=None) >> regulated(state='partial', sense=None),
                    self.k_stop)
        rules.append(rule)

        # Degradation of free RNA
        # ToDo Check whether bound regulator and regulated shall be degraded at the same rate
        # The degradation equation for RNAs is already included on creation
        # rule = Rule('RNA_degradation_%s' % (regulated.name), regulated() >> None, self.k_deg)
        # rules.append(rule)

        self.rules = rules
