import numpy as np

from vangogh.population import Population


def select(population, selection_size, selection_name='tournament_4'):
    if 'tournament' in selection_name:
        tournament_size = int(selection_name.split('_')[-1])
        return tournament_select(population, selection_size, tournament_size)
    elif "fiscis" == selection_name:
        return fiscis_select(population)
    elif "fiscis_umda" == selection_name:
        return fiscis_umda_select(population)    
    else:
        raise ValueError('Invalid selection name:', selection_name)


def one_tournament_round(population, tournament_size, return_winner_index=False):
    rand_perm = np.random.permutation(len(population.fitnesses))
    competing_fitnesses = population.fitnesses[rand_perm[:tournament_size]]
    winning_index = rand_perm[np.argmin(competing_fitnesses)]
    if return_winner_index:
        return winning_index
    else:
        return {
            'genotype': population.genes[winning_index, :],
            'fitness': population.fitnesses[winning_index],
        }


def tournament_select(population, selection_size, tournament_size=4):
    genotype_length = population.genes.shape[1]
    selected = Population(selection_size, genotype_length, "N/A")

    n = len(population.fitnesses)
    num_selected_per_iteration = n // tournament_size
    num_parses = selection_size // num_selected_per_iteration

    for i in range(num_parses):
        # shuffle
        population.shuffle()

        winning_indices = np.argmin(population.fitnesses.squeeze().reshape((-1, tournament_size)),
                                    axis=1)
        winning_indices += np.arange(0, n, tournament_size)

        selected.genes[i * num_selected_per_iteration:(i + 1) * num_selected_per_iteration,
        :] = population.genes[winning_indices, :]
        selected.fitnesses[i * num_selected_per_iteration:(i + 1) * num_selected_per_iteration] = \
        population.fitnesses[winning_indices]

    return selected
    
def fiscis_select(population):
    maxfit = np.max(population.fitnesses)
    minfit = np.min(population.fitnesses)
    if maxfit == minfit:
        probability_survival = np.ones_like(population.fitnesses)
    else:
        probability_survival = (maxfit - population.fitnesses) / (maxfit - minfit)
    mask_survival = np.random.uniform(0, 1, population.fitnesses.shape)
    mask_survival = mask_survival < probability_survival

    #select by population
    N_alive = np.sum(mask_survival)
    genotype_length = population.genes.shape[1]
    selected = Population(N_alive, genotype_length, "N/A")
    selected.genes = population.genes[mask_survival]
    selected.fitnesses = population.fitnesses[mask_survival]

    return selected

#####################
# Function to estimate probabilities of the selected population
def estimate_probabilities(selected_population):
    population_size, genotype_length = selected_population.genes.shape
    probabilities = []
    gene_values_list = []

    for i in range(genotype_length):
        gene_values, counts = np.unique(selected_population.genes[:, i], return_counts=True)
        probabilities.append(counts / population_size)
        gene_values_list.append(gene_values)

    return probabilities, gene_values_list

# Function to sample new population based on probabilities
def sample_new_population(probabilities, gene_values_list, population_size):
    genotype_length = len(probabilities)
    new_population = Population(population_size, genotype_length, "RANDOM")

    for i in range(genotype_length):
        sampled_genes = np.random.choice(gene_values_list[i], size=population_size, p=probabilities[i])
        new_population.genes[:, i] = sampled_genes 
        new_population.genes[:, i] = np.round(new_population.genes[:, i]).astype(int)
    return new_population

#####################


def fiscis_umda_select(population):     #fiscis, umda hybrid, should only work classical, p+o doesn't work.(to make it work it needs additional parameters for p+o situation)
    N_original, genotype_length = population.genes.shape
    selected = Population(N_original, genotype_length, "N/A")

    # make population by fiscis
    pop_fiscis = fiscis_select(population)
    N_fiscis = pop_fiscis.genes.shape[0]

    # make the rest by umda
    probabilities, gene_values = estimate_probabilities(population)
    pop_umda = sample_new_population(probabilities, gene_values, N_original-N_fiscis)

    selected.genes[:N_fiscis] = pop_fiscis.genes
    selected.genes[N_fiscis:] = pop_umda.genes
    selected.fitnesses[:N_fiscis] = pop_fiscis.fitnesses
    selected.fitnesses[N_fiscis:] = pop_umda.fitnesses
    print(f"N_fiscis:{N_fiscis}")
    return selected
