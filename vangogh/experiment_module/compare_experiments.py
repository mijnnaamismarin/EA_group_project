from matplotlib import pyplot as plt
from scipy.stats import wilcoxon, mannwhitneyu, ttest_ind


def compare_experiments(experiment_1, experiment_2, significance_level, plot=True, test="mannwhitneyu"):
    """
    Compare two time series using a pairwise test.

    Args:
        experiment_1 (ndarray): First time series.
        experiment_2 (ndarray): Second time series.
        significance_level (float): Significance level for the statistical test.
        plot (bool): Whether to plot the comparison.
        test (str): Statistical test.

    Returns:
        str: The result of the hypothesis test, either to 'Reject' or 'Retain'.

    """

    if test == "wilcoxon":  # Paired independent observations
        test_result = wilcoxon(experiment_1, experiment_2)
    elif test == "mannwhitneyu":  # Independent tests
        test_result = mannwhitneyu(experiment_1, experiment_2)
    else:
        test_result = ttest_ind(experiment_1, experiment_2)  # Normal distribution assumption

    if plot:
        plot_compare_convergence(experiment_1, experiment_2)

    # The null hypothesis is a statement or assumption that suggests there is
    # no significant difference or relationship between the variables being compared.
    #
    # Reject: Significant difference.
    # Retain: Not enough evidence to support an alternative hypothesis

    if test_result.pvalue < significance_level:
        return test_result.pvalue, 'Null hypothesis rejected'
    else:
        return test_result.pvalue, 'Null hypothesis retained'


def plot_compare_convergence(experiment_1, experiment_2):
    """
    Plot the convergence of 2 series.

    Args:
        experiment_1 (ndarray): First time series.
        experiment_2 (ndarray): Second time series.

    """

    generations = list(range(len(experiment_1)))

    plt.plot(generations, experiment_1)
    plt.plot(generations, experiment_2)
    plt.xlabel('Generations')

    plt.title(f'Convergence Plot Comparing Two Alternatives ')
    plt.ylabel('Fitness Score')
    plt.legend(['Hypothesis Experiment', 'Control Experiment'])
    plt.show()
