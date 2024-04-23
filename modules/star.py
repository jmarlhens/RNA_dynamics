from pysb import Rule, Model
from modules.molecules import RNA
from modules.reactioncomplex import ReactionComplex


class Star(ReactionComplex):
    def __init__(self, rna: RNA = None, rna_regulator: RNA = None, model: Model = None):
        assert rna is not None
        assert rna_regulator is not None

        super().__init__(substrate=rna, product=None, model=model)

        self.rna_name = rna.sequence_name
        self.regulator_name = rna_regulator.sequence_name

        # RNA.get_instance returns either the already existing instance or creates it
        regulated = RNA.get_instance(sequence_name=self.rna_name, model=model)
        regulator = RNA.get_instance(sequence_name=self.regulator_name, model=model)

        self.k_init = self.parameters["k_init"]
        self.k_bind = self.parameters["k_bind"]
        self.k_unbind = self.parameters["k_unbind"]
        self.k_act = self.parameters["k_act"]
        self.k_act_reg = self.parameters["k_act_reg"]
        self.k_stop = self.parameters["k_stop"]
        self.k_stop_reg = self.parameters["k_stop_reg"]
        self.k_deg = self.parameters["k_deg"]

        rules = []

        # Define the Rules for the RNA dynamics
        # Initiation of RNA transcription
        rule = Rule('RNA_transcription_initiation_%s' % (regulated.name), None >> regulated(state='init', terminator=None, r=None), self.k_init)
        rules.append(rule)

        # Binding of RNA regulator to the early transcript
        rule = Rule('RNA_regulator_binding_%s_%s' % (regulated.name, regulator.name), regulated(terminator=None) + regulator(state='full', r=None) >> regulated(terminator=1)%regulator(state='full', r=1), self.k_bind)
        rules.append(rule)

        # Unbinding of Full and Early RNA transcript and RNA regulator (slow reaction)
        rule = Rule('RNA_regulator_unbinding_full_%s_%s' % (regulated.name, regulator.name), regulated(terminator=1)%regulator(state='full', r=1) >> regulated(terminator=None) + regulator(state='full', r=None), self.k_unbind)
        rules.append(rule)

        # Activation of Transcription (RNA activator promotes transcription)
        rule = Rule('RNA_full_transcription_reg_%s_%s' % (regulated.name, regulator.name), regulated(state='init', terminator=1)%regulator(state='full', r=1) >> regulated(state='full', terminator=1)%regulator(state='full', r=1), self.k_act_reg)
        rules.append(rule)
        rule = Rule('RNA_full_transcription_%s' % (regulated.name), regulated(state='init', terminator=None) >> regulated(state='full', terminator=None), self.k_act)
        rules.append(rule)

        # Stop of Transcription (RNA inhibitor stops transcription)
        rule = Rule('RNA_partial_transcription_reg_%s_%s' % (regulated.name, regulator.name), regulated(state='init', terminator=1)%regulator(state='full', r=1) >> regulated(state='partial', terminator=1)%regulator(state='full', r=1), self.k_stop_reg)
        rules.append(rule)
        rule = Rule('RNA_partial_transcription_%s' % (regulated.name), regulated(state='init', terminator=None) >> regulated(state='partial', terminator=None), self.k_stop)
        rules.append(rule)

        # Degradation of free RNA
        rule = Rule('RNA_degradation_%s' % (regulated.name), regulated() >> None, self.k_deg)
        rules.append(rule)

        self.rules = rules