from pysb import Parameter, Rule, Expression, Model
from pysb.util import alias_model_components

def RNA_regulatory_base_module(RNA_regulated, RNA_regulator, k_init, k_bind, k_unbind, k_act, k_act_reg, k_stop, k_stop_reg, k_deg):

    # Define the Rules for the RNA dynamics
    # Initiation of RNA transcription
    Rule('RNA_transcription_initiation_%s' % (RNA_regulated.name), None >> RNA_regulated(state='init', terminator=None, r=None), k_init)

    # Binding of RNA regulator to the early transcript
    Rule('RNA_regulator_binding_%s_%s' % (RNA_regulated.name, RNA_regulator.name), RNA_regulated(terminator=None) + RNA_regulator(state='full', r=None) >> RNA_regulated(terminator=1)%RNA_regulator(state='full', r=1), k_bind)
    # Unbinding of Full and Early RNA transcript and RNA regulator (slow reaction)
    Rule('RNA_regulator_unbinding_full_%s_%s' % (RNA_regulated.name, RNA_regulator.name), RNA_regulated(terminator=1)%RNA_regulator(state='full', r=1) >> RNA_regulated(terminator=None) + RNA_regulator(state='full', r=None), k_unbind)

    # Activation of Transcription (RNA activator promotes transcription):
    Rule('RNA_full_transcription_reg_%s_%s' % (RNA_regulated.name, RNA_regulator.name), RNA_regulated(state='init', terminator=1)%RNA_regulator(state='full', r=1) >> RNA_regulated(state='full', terminator=1)%RNA_regulator(state='full', r=1), k_act_reg)
    Rule('RNA_full_transcription_%s' % (RNA_regulated.name), RNA_regulated(state='init', terminator=None) >> RNA_regulated(state='full', terminator=None), k_act)

    # Stop of Transcription (RNA inhibitor stops transcription):
    Rule('RNA_partial_transcription_reg_%s_%s' % (RNA_regulated.name, RNA_regulator.name), RNA_regulated(state='init', terminator=1)%RNA_regulator(state='full', r=1) >> RNA_regulated(state='partial', terminator=1)%RNA_regulator(state='full', r=1), k_stop_reg)
    Rule('RNA_partial_transcription_%s' % (RNA_regulated.name), RNA_regulated(state='init', terminator=None) >> RNA_regulated(state='partial', terminator=None), k_stop)

    # Degradation of free RNA
    Rule('RNA_degradation_%s' % (RNA_regulated.name), RNA_regulated() >> None, k_deg)

def RNA_regulatory_base_parameters(model, RNA_regulated, RNA_regulator, parameters):
    # Define the Parameters for the kinetic rates
    Parameter('k_init_' + RNA_regulated.name, parameters['k_init']) # in mol . L^-1 . s^-1
    Parameter('k_bind_' + RNA_regulated.name + "_" + RNA_regulator.name, parameters['k_bind']) # in mol^-1 . L . s^-1
    Parameter('k_unbind_' + RNA_regulated.name + "_" + RNA_regulator.name, parameters['k_unbind']) # in s^-1
    Parameter('k_act_' + RNA_regulated.name, parameters['k_act']) # in s^-1
    Parameter('k_act_reg_' + RNA_regulated.name + "_" + RNA_regulator.name, parameters['k_act_reg']) # in s^-1
    Parameter('k_stop_' + RNA_regulated.name, parameters['k_stop']) # in s^-1
    Parameter('k_stop_reg_' + RNA_regulated.name + "_" + RNA_regulator.name, parameters['k_stop_reg']) # in s^-1
    Parameter('k_deg_' + RNA_regulated.name, parameters['k_deg']) # in s^-1

    # Rate constant conversion to molecule numbers
    Expression('k_init_' + RNA_regulated.name + '_molnum', model.parameters['k_init_' + RNA_regulated.name] * model.parameters['omega']) # in s^-1
    Expression('k_bind_' + RNA_regulated.name + "_" + RNA_regulator.name + '_molnum', model.parameters['k_bind_' + RNA_regulated.name + "_" + RNA_regulator.name] / model.parameters['omega']) # in s^-1
    Expression('k_unbind_' + RNA_regulated.name + "_" + RNA_regulator.name + '_molnum', model.parameters['k_unbind_' + RNA_regulated.name + "_" + RNA_regulator.name]) # in s^-1
    Expression('k_act_' + RNA_regulated.name + '_molnum', model.parameters['k_act_' + RNA_regulated.name]) # in s^-1
    Expression('k_act_reg_' + RNA_regulated.name + "_" + RNA_regulator.name + '_molnum', model.parameters['k_act_reg_' + RNA_regulated.name + "_" + RNA_regulator.name]) # in s^-1
    Expression('k_stop_' + RNA_regulated.name + '_molnum', model.parameters['k_stop_' + RNA_regulated.name]) # in s^-1
    Expression('k_stop_reg_' + RNA_regulated.name + "_" + RNA_regulator.name + '_molnum', model.parameters['k_stop_reg_' + RNA_regulated.name + "_" + RNA_regulator.name]) # in s^-1
    Expression('k_deg_' + RNA_regulated.name + '_molnum', model.parameters['k_deg_' + RNA_regulated.name]) # in s^-1

    alias_model_components()

def init_model(omega_val):
    # Create a PySB model
    model = Model()
    Parameter('omega', omega_val)  # in L
    alias_model_components()
    return model
