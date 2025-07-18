import argparse
from multiprocessing import Pool

from simulations_and_analysis.individual import individual_circuits_run_mcmc


def main(num_processes=1):
    circuits_to_fit = [
        # "constitutive sfGFP",                   # J
        # "sense_star_6",                       # J
        # "toehold_trigger",                    # J
        # "star_antistar_1",                    # J
        # "trigger_antitrigger",                # J
        "cascade",
        "cffl_type_1",
        "inhibited_incoherent_cascade",
        "inhibited_cascade",
        "or_gate_c1ffl",
        "iffl_1",
        "cffl_12",
    ]



    print(f"STARTING EXECUTION OF ALL CIRCUITS WITH {num_processes} PROCESSES")
    print("CIRCUITS:")
    print(circuits_to_fit)

    circuits_to_fit = list(map(lambda elem: [elem], circuits_to_fit))
    with Pool(num_processes) as p:
        p.map(individual_circuits_run_mcmc.main, circuits_to_fit)

    print("COMPLETED EXECUTION OF ALL CIRCUITS")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Individual Parallel Circuits Run MCMC',
        description='Model calibration of individual circuits')

    parser.add_argument('-n', '--num_processes', type=int, default=4)  # optional argument

    args = parser.parse_args()
    num_processes = args.num_processes


    main(num_processes=num_processes)