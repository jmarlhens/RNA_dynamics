import re
import sympy as sp
import numpy as np


# Function to substitute expressions in an ODE string
def substitute_expressions(ode_string, expression_dict):
    for expr_name, expr_definition in expression_dict.items():
        # Replace each expression placeholder with its definition
        # This uses a regular expression to find whole word matches only, to avoid partial replacements
        pattern = r'\b' + re.escape(expr_name) + r'\b'
        ode_string = re.sub(pattern, f'({expr_definition})', ode_string)
    return ode_string

def find_ODEs_from_Pysb_model(model):
    odes = model._odes
    if len(odes) == 0:
        raise Exception("Model has no odes. Please run ODE build_simulate_analyse before to instantiate odes.")
    #
    equations = []
    for i, eq in enumerate(odes):
        equations += ["d" + model.observables[i].name + f"/dt = {eq}"]

    #
    name_mapping = {}
    for i, species in enumerate(model.species):
        name_mapping[f'__s{i}'] = model.observables[i].name
    expression_dict = {expr.name: str(expr.expand_expr()) for expr in model.expressions}

    # Substitute expressions in the ODEs
    equations = [substitute_expressions(eq, expression_dict) for eq in equations]
    equations = [substitute_expressions(eq, name_mapping) for eq in equations]

    return equations

def simpy_equations_from_string(equations):
    # Extract observables as variables
    variables = sp.symbols([obs.name for obs in model.observables])

    # Extract parameters and their values
    parameters = {param.name: param.value for param in model.parameters[::-1]}

    # Dynamically create sympy symbols for parameters
    param_symbols = sp.symbols(list(parameters.keys()))

    all_symbols_dict = {**{str(var): var for var in variables}, **{str(param): param for param in param_symbols}}

    # Assuming 'equations' is your list of ODE strings
    equations_sym = []
    for eq_str in equations:
        pass
        # Split each equation string at '=',
        # taking the part after '=' as the expression to be solved for steady state
        ode_rhs = eq_str.split('=')[1]

        # Substitute parameter names with their values in the expression
        for param_name, param_value in parameters.items():
            ode_rhs = ode_rhs.replace(param_name, str(param_value))

        # Convert to a symbolic expression
        expr = sp.sympify(ode_rhs, locals=all_symbols_dict)
        equations_sym.append(expr - 0)  # Subtract 0 to imply '= 0' for steady state

    return equations_sym, variables

def simpy_equations_from_Pysb_model(model):
    equations = find_ODEs_from_Pysb_model(model)
    equations_sym, variables = simpy_equations_from_string(equations)
    return equations_sym, variables

def convert_to_latex(equations):
    # Convert the ODEs into LaTeX format with specified modifications
    formatted_equations = []
    for eq in equations:
        # Replace ** with ^ for exponentiation in LaTeX
        eq = eq.replace('**', '^')

        # Add subscript to variable names that have an underscore and subsequent characters
        eq = re.sub(r'(\w+)_(\w+)', r'\1_{\2}', eq)

        # remove "*1"
        eq = re.sub(r'\*(1)', '', eq)

        # Simplify multiplication by -1 by targeting the entire expression preceding it
        # Look for expressions enclosed in parentheses followed by *(-1) and prepend a negative sign
        eq = re.sub(r'\(([^)]+)\)\*\(\-1\)', r'-\(\1\)', eq)
        # Address any instances not enclosed by parentheses, though this case should be rare
        eq = re.sub(r'(\w+)\*\(\-1\)', r'-\1', eq)

        # Convert derivatives and general fractions to LaTeX fraction format
        # Handle derivatives first
        eq = re.sub(r'(d\w+_{\w+})/(d\w+)', r'\\frac{\1}{\2}', eq)
        # Handle more complex expressions
        eq = re.sub(r'(\([^)]+\))\/(\([^)]+\))', r'\\frac{\1}{\2}', eq)
        # Handle simple variable or constant fractions
        eq = re.sub(r'(\b\w+|\b\w+_{\w+})\/(\b\w+|\b\w+_{\w+})', r'\\frac{\1}{\2}', eq)

        # Align the equation using &=
        eq = eq.replace('=', '&=')

        formatted_equations.append(eq)

    # Combine all equations into one, separated by \\
    combined_equations = ' \\\\\n'.join(formatted_equations)
    return f'\\begin{{align*}}\n{combined_equations}\n\\end{{align*}}'

def write_to_file(latex_code, filename="odes.tex"):
    # Write the LaTeX-formatted equations to a file
    with open(filename, 'w') as file:
        file.write('\\documentclass{article}\n\\usepackage{amsmath}\n\\begin{document}\n')
        file.write(latex_code + '\n')
        file.write('\\end{document}')


if __name__ == '__main__':
    from pysb.examples.robertson import model
    # simulate the model
    tspan = np.linspace(0, 300, 3000)
    from pysb.simulator import ScipyOdeSimulator
    sim = ScipyOdeSimulator(model, tspan)
    simres = sim.run()


    equations = find_ODEs_from_Pysb_model(model)
    print(equations)

    # Convert and write to file
    latex_equations = convert_to_latex(equations)
    write_to_file(latex_equations)
