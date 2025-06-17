"""
Centralized configuration for synthetic circuit analyses.
Contains circuit condition parameters and data file paths.
"""

# Data file paths for each circuit
DATA_FILES = {
    "trigger_antitrigger": "../../data/data_parameter_estimation/trigger_antitrigger.csv",
    "sense_star_6": "../../data/data_parameter_estimation/sense_star.csv",
    "cascade": "../../data/data_parameter_estimation/cascade.csv",
    "toehold_trigger": "../../data/data_parameter_estimation/toehold_trigger.csv",
    "cffl_type_1": "../../data/data_parameter_estimation/c1_ffl_and.csv",
    "star_antistar_1": "../../data/data_parameter_estimation/star_antistar.csv",
}

# Circuit condition parameters
CIRCUIT_CONDITIONS = {
    "trigger_antitrigger": {
        "To3 3 nM + Tr3 5 nM": {
            "k_Toehold3_GFP_concentration": 3,
            "k_Trigger3_concentration": 5,
            "k_aTrigger3_concentration": 0,
        },
        # "To3 3 nM + Tr3 3 nM": {
        #     "k_Toehold3_GFP_concentration": 3,
        #     "k_Trigger3_concentration": 3,
        #     "k_aTrigger3_concentration": 0,
        # },
        "To3 3 nM + Tr3 5 nM + aTr3 5 nM": {
            "k_Toehold3_GFP_concentration": 3,
            "k_Trigger3_concentration": 5,
            "k_aTrigger3_concentration": 5,
        },
        "To3 3 nM + Tr3 5 nM + aTr3 10 nM": {
            "k_Toehold3_GFP_concentration": 3,
            "k_Trigger3_concentration": 5,
            "k_aTrigger3_concentration": 10,
        },
        "To3 3 nM + Tr3 5 nM + aTr3 15 nM": {
            "k_Toehold3_GFP_concentration": 3,
            "k_Trigger3_concentration": 5,
            "k_aTrigger3_concentration": 15,
        },
    },
    "star_antistar_1": {
        "Sense only": {
            "k_Star1_concentration": 0,
            "k_aStar1_concentration": 0,
            "k_Sense1_GFP_concentration": 3,
        },
        "Sense with 5 nM STAR": {
            "k_Star1_concentration": 5,
            "k_aStar1_concentration": 0,
            "k_Sense1_GFP_concentration": 3,
        },
        "Sense with 10 nM STAR": {
            "k_Star1_concentration": 10,
            "k_aStar1_concentration": 0,
            "k_Sense1_GFP_concentration": 3,
        },
        "Sense with 15 nM STAR": {
            "k_Star1_concentration": 15,
            "k_aStar1_concentration": 0,
            "k_Sense1_GFP_concentration": 3,
        },
        "Sense with 15 nM STAR and 15 nM antiSTAR": {
            "k_Star1_concentration": 15,
            "k_aStar1_concentration": 15,
            "k_Sense1_GFP_concentration": 3,
        },
        "Sense with 15 nM STAR and 10 nM antiSTAR": {
            "k_Star1_concentration": 15,
            "k_aStar1_concentration": 10,
            "k_Sense1_GFP_concentration": 3,
        },
        "Sense with 15 nM STAR and 5 nM antiSTAR": {
            "k_Star1_concentration": 15,
            "k_aStar1_concentration": 5,
            "k_Sense1_GFP_concentration": 3,
        },
    },
    "toehold_trigger": {
        "To3 5 + Tr3 5": {
            "k_Toehold3_GFP_concentration": 5,
            "k_Trigger3_concentration": 5,
        },
        "To3 5 + Tr3 4": {
            "k_Toehold3_GFP_concentration": 5,
            "k_Trigger3_concentration": 4,
        },
        "To3 5 + Tr3 3": {
            "k_Toehold3_GFP_concentration": 5,
            "k_Trigger3_concentration": 3,
        },
        "To3 5 + Tr3 2": {
            "k_Toehold3_GFP_concentration": 5,
            "k_Trigger3_concentration": 2,
        },
        "To3 5 + Tr3 1": {
            "k_Toehold3_GFP_concentration": 5,
            "k_Trigger3_concentration": 1,
        },
        "To3 5": {"k_Toehold3_GFP_concentration": 5, "k_Trigger3_concentration": 0},
    },
    "sense_star_6": {
        "Se6 5 nM + St6 15 nM": {
            "k_Sense6_GFP_concentration": 5,
            "k_Star6_concentration": 15,
        },
        "Se6 5 nM + St6 10 nM": {
            "k_Sense6_GFP_concentration": 5,
            "k_Star6_concentration": 10,
        },
        "Se6 5 nM + St6 5 nM": {
            "k_Sense6_GFP_concentration": 5,
            "k_Star6_concentration": 5,
        },
        "Se6 5 nM + St6 3 nM": {
            "k_Sense6_GFP_concentration": 5,
            "k_Star6_concentration": 3,
        },
        "Se6 5 nM + St6 0 nM": {
            "k_Sense6_GFP_concentration": 5,
            "k_Star6_concentration": 0,
        },
    },
    "cascade": {
        "To3 3 nM + Se6Tr3P 5 nM + St6 15 nM": {
            "k_Toehold3_GFP_concentration": 3,
            "k_Sense6_Trigger3_concentration": 5,
            "k_Star6_concentration": 15,
        },
        "To3 3 nM + Se6Tr3P 5 nM + St6 10 nM": {
            "k_Toehold3_GFP_concentration": 3,
            "k_Sense6_Trigger3_concentration": 5,
            "k_Star6_concentration": 10,
        },
        "To3 3 nM + Se6Tr3P 5 nM + St6 5 nM": {
            "k_Toehold3_GFP_concentration": 3,
            "k_Sense6_Trigger3_concentration": 5,
            "k_Star6_concentration": 5,
        },
        "To3 3 nM + Se6Tr3P 5 nM + St6 3 nM": {
            "k_Toehold3_GFP_concentration": 3,
            "k_Sense6_Trigger3_concentration": 5,
            "k_Star6_concentration": 3,
        },
        "To3 3 nM + Se6Tr3P 5 nM + St6 0 nM": {
            "k_Toehold3_GFP_concentration": 3,
            "k_Sense6_Trigger3_concentration": 5,
            "k_Star6_concentration": 0,
        },
    },
    "cffl_type_1": {
        "15 nM": {
            "k_Sense6_Trigger3_concentration": 5,
            "k_Star6_concentration": 15,
            "k_Sense6_Toehold3_GFP_concentration": 3,
        },
        "12 nM": {
            "k_Sense6_Trigger3_concentration": 5,
            "k_Star6_concentration": 12,
            "k_Sense6_Toehold3_GFP_concentration": 3,
        },
        "10 nM": {
            "k_Sense6_Trigger3_concentration": 5,
            "k_Star6_concentration": 10,
            "k_Sense6_Toehold3_GFP_concentration": 3,
        },
        "7 nM": {
            "k_Sense6_Trigger3_concentration": 5,
            "k_Star6_concentration": 7,
            "k_Sense6_Toehold3_GFP_concentration": 3,
        },
        "5 nM": {
            "k_Sense6_Trigger3_concentration": 5,
            "k_Star6_concentration": 5,
            "k_Sense6_Toehold3_GFP_concentration": 3,
        },
        "3 nM": {
            "k_Sense6_Trigger3_concentration": 5,
            "k_Star6_concentration": 3,
            "k_Sense6_Toehold3_GFP_concentration": 3,
        },
        "0 nM": {
            "k_Sense6_Trigger3_concentration": 5,
            "k_Star6_concentration": 0,
            "k_Sense6_Toehold3_GFP_concentration": 3,
        },
    },
}


# Helper function to get a circuit's condition parameters
def get_circuit_conditions(circuit_name):
    """Get the condition parameters for a specific circuit"""
    return CIRCUIT_CONDITIONS.get(circuit_name, {})


# Helper function to get a circuit's data file path
def get_data_file(circuit_name):
    """Get the data file path for a specific circuit"""
    return DATA_FILES.get(circuit_name)
