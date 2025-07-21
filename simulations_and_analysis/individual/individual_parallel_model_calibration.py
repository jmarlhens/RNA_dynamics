import argparse
import time
from multiprocessing import Pool

from simulations_and_analysis.individual import individual_circuits_run_mcmc


def main(num_processes=1, waiting_time=5):
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
        # p.map(individual_circuits_run_mcmc.main, circuits_to_fit)
        results = []
        for circuit_id in circuits_to_fit:
            res = p.apply_async(individual_circuits_run_mcmc.main, [circuit_id])
            print(f"SUBMITTED  {circuit_id} and now waiting {waiting_time} s prior to submission of next circuit")
            time.sleep(waiting_time)  # To make the cython execution disjoint
            results.append(res)
        p.close()
        print("SUBMITTED ALL CIRCUITS, WAITING FOR JOIN")
        do_wait = True
        completed_circuits = []
        while do_wait:
            time.sleep(60)
            do_wait = False
            print("CHECK ON PROCESS STATE")
            for iC, res in enumerate(results):
                circuit_id = circuits_to_fit[iC]
                # if circuit_id in completed_circuits:
                #    continue

                is_ready = res.ready()

                do_wait = do_wait or not is_ready
                if is_ready:
                    completed_circuits.append(circuit_id)
                    print(f"CIRCUIT {circuit_id} is READY and finished {'SUCCESSFULL' if res.successful() else 'WITH EXCEPTION'}.")

        p.join()
    print("COMPLETED EXECUTION OF ALL CIRCUITS")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Individual Parallel Circuits Run MCMC',
        description='Model calibration of individual circuits')

    parser.add_argument('-n', '--num_processes', type=int, default=7)  # optional argument

    args = parser.parse_args()
    num_processes = args.num_processes

    main(num_processes=num_processes)
