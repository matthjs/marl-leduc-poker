from cfr_experiment import Experiment2
from dream_experiment import Experiment
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def plot_exploitability(cfr_e, cfr_i, dream_e, dream_i, outfile='Comparison.png'):
    """
    Plots exploitability for CFR and DREAM experiments against Iterations trained
    with 95% confidence intervals across multiple runs.

    Parameters:
    - cfr_e, dream_e: list of lists of exploitability scores per run
    - cfr_i, dream_i: list of lists of iterations per run
    - outfile: filename to save the plot (will save as a combined figure)
    """

    def mean_ci(data):
        """Compute mean and 95% CI for each timestep/node, safe for small samples."""
        data = np.array(data)
        mean = np.mean(data, axis=0)
        
        if len(data) < 2:
            # Not enough runs for a confidence interval
            ci = np.zeros_like(mean)
        else:
            sem = stats.sem(data, axis=0, nan_policy='omit')
            ci = sem * stats.t.ppf(0.975, len(data) - 1)
        return mean, ci

    # Compute mean and CI for iterations
    cfr_e_iters_mean, cfr_e_iters_ci = mean_ci(cfr_e)
    cfr_i_mean, _ = mean_ci(cfr_i)
    dream_e_iters_mean, dream_e_iters_ci = mean_ci(dream_e)
    dream_i_mean, _ = mean_ci(dream_i)

    # --- Plot: Exploitability vs Iterations Trained ---
    plt.figure(figsize=(10, 5))
    plt.fill_between(cfr_i_mean, cfr_e_iters_mean - cfr_e_iters_ci, cfr_e_iters_mean + cfr_e_iters_ci, color='blue', alpha=0.2)
    plt.plot(cfr_i_mean, cfr_e_iters_mean, label='CFR', color='blue', marker='o')
    
    plt.fill_between(dream_i_mean, dream_e_iters_mean - dream_e_iters_ci, dream_e_iters_mean + dream_e_iters_ci, color='orange', alpha=0.2)
    plt.plot(dream_i_mean, dream_e_iters_mean, label='DREAM', color='orange', marker='s')
    
    plt.xlabel('Iterations Trained')
    plt.ylabel('Exploitability')
    plt.title('Exploitability vs Iterations Trained')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Iterations_' + outfile)
    plt.show()


if __name__ == "__main__":
    cfr_e = []
    cfr_i = []
    dream_e = []
    dream_i = []
    runs = 10

    for run in range(runs):
        experiment_cfr = Experiment2()
        experiment_dream = Experiment()
        cfr_iters, cfr_exp = experiment_cfr.run()
        dream_iters, dream_exp = experiment_dream.run()
        cfr_e.append(cfr_exp)
        cfr_i.append(cfr_iters)
        dream_e.append(dream_exp)
        dream_i.append(dream_iters)
        plot_exploitability(cfr_e, cfr_i, dream_e, dream_i, outfile=f"Exploitability_Comparison_4000i_{run+1}Runs.png")
    
    
