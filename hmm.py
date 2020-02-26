import random
import argparse
import codecs
from collections import defaultdict
import numpy as np
from numpy import random
import os


def helper_read(filename, values_of_arc, values_of_arc_prov):
    for line in map(lambda line: line.split(), codecs.open(filename, 'r', 'utf8')):
        from_s = line[0]
        to_s = line[1]
        if len(line) != 3:
            prob = None
            values_of_arc_prov = False
        else:
            if len(line) < 2:
                continue
            prob = float(line[2])
            values_of_arc[from_s][to_s] = prob
    return values_of_arc, values_of_arc_prov


def read_arcs(filename):
    values_of_arc = defaultdict(lambda: defaultdict(float))
    values_of_arc_prov = True
    values_of_arc, values_of_arc_prov = helper_read(filename, values_of_arc, values_of_arc_prov)
    return values_of_arc, values_of_arc_prov


def helper_func(countdict, total):
    return {item: val / total for item, val in countdict.items()}


def normalize(countdict):
    total = float(sum(countdict.values()))
    values = helper_func(countdict, total)
    return values


def write_arcs(dictionary_arct, filename):
    output_write = codecs.open(filename, 'w', 'utf8')
    output_write = helper_write(dictionary_arct, output_write)
    output_write.close()


def random_gen(m):
    return np.random.random()


def helper_values_from_distribution(values_cumul, d, cumulative):
    for values in d:
        cumulative += d[values]
        if values_cumul < cumulative:
            return values


def take_values_from_distribution(d):
    values_cumul = random_gen(20)
    cumulative = 0
    value = helper_values_from_distribution(values_cumul, d, cumulative)
    return value


def helper_write(dictionary_arct, output_write1):
    # Write the values to output dictionary
    for fn in dictionary_arct:
        for en in dictionary_arct[fn]:
            if dictionary_arct[fn][en] != 0:
                output_write1.write(fn + ' ' + en + ' ' + str(dictionary_arct[fn][en]) + '\n')
    return output_write1


# Observation class
class Observation:
    def __init__(self, states_of_sequence, sequence_of_outputs):
        self.states_of_sequence = states_of_sequence
        self.sequence_of_outputs = sequence_of_outputs

    def __str__(self):
        return ' '.join(self.states_of_sequence) + '\n' + ' '.join(self.sequence_of_outputs) + '\n'

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.sequence_of_outputs)


# Load the observations file
def load_observations(filename):
    total_corpus_lines = [line.split() for line in codecs.open(filename, 'r', 'utf8').readlines()]
    k = 1 + len(total_corpus_lines)
    if k % 2 != 1:
        total_corpus_lines[:len(total_corpus_lines) - 1]
    return [Observation(total_corpus_lines[i], total_corpus_lines[i + 1]) for i in range(0, len(total_corpus_lines), 2)]


# hmm model
class HMM:
    def __init__(self, transitions=None, emissions=None):
        self.transitions = transitions
        self.emissions = emissions
        if self.emissions:
            self.states = self.emissions.keys()

    def load(self, basename):
        # TODO: fill in for section a
        self.transitions, transmission_values_provided = read_arcs(basename + '.trans')
        self.emissions, emission_values_provided = read_arcs(basename + '.emit')
        self.states = self.emissions.keys()

        if not transmission_values_provided:
            print
            'No transition probabilities is given: initialized randomly.'
            self.init_transitions_random()
        if not emission_values_provided:
            print
            'No Emission probabilities is given: initialized randomly.'
            self.init_emissions_random()

    def init_transitions_random(self):
        for from_state in self.transitions:
            random_probs = np.random.random(len(self.transitions[from_state]))
            total = sum(random_probs)
            for to_index, to_state in enumerate(self.transitions[from_state]):
                self.transitions[from_state][to_state] = random_probs[to_index] / total

    def init_emissions_random(self):
        for state in self.emissions:
            random_probs = np.random.random(len(self.emissions[state]))
            total = sum(random_probs)
            for symi, sym in enumerate(self.emissions[state]):
                self.emissions[state][sym] = random_probs[symi] / total

    def dump(self, basename):
        write_arcs(self.transitions, basename + '.trans')
        write_arcs(self.emissions, basename + '.emit')

    def helper_generate(self, n, states, outputs):
        for i in range(n):
            if i == 0:
                state = take_values_from_distribution(self.transitions['#'])
            else:
                state = take_values_from_distribution(self.transitions[state])
            states.append(state)
            symbol = take_values_from_distribution(self.emissions[state])
            outputs.append(symbol)
        return states, outputs

    def generate(self, n):
        states = []
        outputs = []
        states, outputs = HMM.helper_generate(self, n, states, outputs)
        return Observation(states, outputs)

    def helper_viterbi(self, observation, cost_of_values_viterbi, backwards_of_algo_vb):

        for oi, obs in enumerate(observation.sequence_of_outputs):
            for si, state in enumerate(self.states):
                if oi == 0:
                    cost_of_values_viterbi[si, oi] = self.transitions['#'][state] * self.emissions[state][obs]
                else:
                    best_costs = {}
                    for pi, prevstate in enumerate(self.states):
                        best_costs[pi] = cost_of_values_viterbi[pi, oi - 1] * self.transitions[prevstate][state]
                    best_state, best_cost = max(best_costs.items(), key=lambda (state, cost): cost)
                    cost_of_values_viterbi[si, oi] = best_cost * self.emissions[state][obs]
                    backwards_of_algo_vb[si, oi] = best_state
        return cost_of_values_viterbi, best_costs, best_cost, best_state, backwards_of_algo_vb


    # Viterbi algorithm for maximum probable sequence
    def viterbi(self, observation):
        cost_of_values_viterbi = np.zeros((len(self.states), len(observation)))
        backwards_of_algo_vb = np.zeros((len(self.states), len(observation)), dtype=int)
        paths_mpp_vb = []
        cost_of_values_viterbi, best_costs, best_cost, best_state, backwards_of_algo_vb = HMM.helper_viterbi(self,
                                                                                                             observation,
                                                                                                             cost_of_values_viterbi,
                                                                                                             backwards_of_algo_vb)
        oi = len(observation) - 1
        best_state = np.argmax(cost_of_values_viterbi[:, oi])
        paths_mpp_vb.append(self.states[best_state])
        while oi > 0:
            best_state = backwards_of_algo_vb[best_state, oi]
            paths_mpp_vb.append(self.states[best_state])
            oi -= 1
        observation.states_of_sequence = paths_mpp_vb[::-1]

    def helper_forward(self, observation, matrix_forward_values):
        for oi, obs in enumerate(observation.sequence_of_outputs):
            for si, state in enumerate(self.states):
                if oi != 0:
                    for pi, prevstate in enumerate(self.states):
                        matrix_forward_values[si, oi] += matrix_forward_values[pi, oi - 1] * \
                                                         self.transitions[prevstate][state]
                    matrix_forward_values[si, oi] *= self.emissions[state][obs]
                else:
                    matrix_forward_values[si, oi] = self.transitions['#'][state] * self.emissions[state][obs]
        return matrix_forward_values

    def forward(self, observation):
        matrix_forward_values = np.zeros((len(self.states), len(observation)))
        matrix_values = HMM.helper_forward(self, observation, matrix_forward_values)
        return matrix_values

    def probability_forward1ability(self, observation):
        matrix_forward_values = self.forward(observation)
        return sum(matrix_forward_values[:, len(observation) - 1])

    def backward(self, observation):
        matrix_backward_values = np.zeros((len(self.states), len(observation)))
        for si, state in enumerate(self.states):
            matrix_backward_values[si, len(observation) - 1] = 1

        for oi in range(len(observation) - 2, -1, -1):
            for si, state in enumerate(self.states):
                for ni, nextstate in enumerate(self.states):
                    matrix_backward_values[si, oi] += matrix_backward_values[ni, oi + 1] * self.transitions[state][
                        nextstate] * self.emissions[nextstate][observation.sequence_of_outputs[oi + 1]]

        return matrix_backward_values

    def backward_probability(self, observation):
        matrix_backward_values = self.backward(observation)
        backprob = 0.0
        for si, state in enumerate(self.states):
            backprob += self.transitions['#'][state] * self.emissions[state][observation.sequence_of_outputs[0]] * \
                        matrix_backward_values[si, 0]
        return backprob

    def step_2_em_algo(self, counts_emitted, counts_transmitted, lock_for_emitted, lock_for_transmitted):
        if not lock_for_transmitted:
            for from_state in counts_transmitted:
                self.transitions[from_state] = normalize(counts_transmitted[from_state])

        if not lock_for_emitted:
            for state in counts_emitted:
                self.emissions[state] = normalize(counts_emitted[state])

    def unsupervised_learning_likelihood(self, total_available_corpus, values_to_cross_cv=0.0002,
                                         lock_for_emitted=False, lock_for_transmitted=False, number_of_converges=0):
        # Unsupervised learning
        emission_nest_probability = None
        log_likeli_best_values = -np.inf
        transmission_best_model = None
        old_ll = -np.inf
        log_likelihood = -1e300
        for i in range(number_of_converges + 1):
            if i > 0:

                if not lock_for_transmitted:
                    self.init_transitions_random()
                if not lock_for_emitted:
                    self.init_emissions_random()

            while log_likelihood - old_ll > values_to_cross_cv:
                old_ll = log_likelihood
                log_likelihood, counts_emitted, counts_transmitted = self.learning_step_expectation(
                    total_available_corpus)
                self.step_2_em_algo(counts_emitted, counts_transmitted, lock_for_emitted, lock_for_transmitted)
                print" Log likelihood values is  :" + str(log_likelihood)

            print'values reached convergence point'

            if log_likelihood > log_likeli_best_values:
                log_likeli_best_values = log_likelihood
                transmission_best_model = self.transitions
                emission_nest_probability = self.emissions

        self.__init__(transmission_best_model, emission_nest_probability)
        return log_likeli_best_values

    def learn_supervised(self, total_available_corpus, lock_for_emitted=False, lock_for_transmitted=False):
        counts_transmitted = defaultdict(lambda: defaultdict(int))
        counts_emitted = defaultdict(lambda: defaultdict(int))
        for observation in total_available_corpus:
            for oi in range(len(observation)):
                if oi != 0:
                    counts_transmitted['#'][observation.states_of_sequence[oi]] = counts_transmitted['#'][observation.states_of_sequence[oi]]+1
                else:
                    counts_transmitted[observation.states_of_sequence[oi - 1]][observation.states_of_sequence[oi]] += 1
                counts_emitted[observation.states_of_sequence[oi]][observation.sequence_of_outputs[oi]] += 1
        self.step_2_em_algo(counts_emitted, counts_transmitted, lock_for_emitted, lock_for_transmitted)





    def learning_step_expectation(self, total_available_corpus):
        log_likelihood = 0.0
        counts_emitted = defaultdict(lambda: defaultdict(float))
        counts_transmitted = defaultdict(lambda: defaultdict(float))

        for observation in total_available_corpus:
            matrix_forward_values = self.forward(observation)
            matrix_backward_values = self.backward(observation)
            probability_forward1 = sum(matrix_forward_values[:, len(observation) - 1])
            for oi, obs in enumerate(observation.sequence_of_outputs):
                prob_state = {}
                for si, state in enumerate(self.states):
                    counts_emitted[state][obs] += matrix_forward_values[si, oi] * matrix_backward_values[
                        si, oi] / probability_forward1
                prob_1_2 = {}
                for si, state in enumerate(self.states):
                    if oi != 0:
			for pi, prevstate in enumerate(self.states):
                 	     counts_transmitted[prevstate][state] += matrix_forward_values[pi, oi - 1] * \
                                                                    self.transitions[prevstate][state] * \
                                                                    self.emissions[state][obs] * matrix_backward_values[
                                                                        si, oi] /probability_forward1

                    else:
			 constm= matrix_backward_values[si, oi]/probability_forward1
                         counts_transmitted['#'][state] += matrix_forward_values[si, oi] *constm

            log_likelihood += np.log2(probability_forward1)
        return log_likelihood, counts_emitted, counts_transmitted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('paramfile', type=str, help='basename of the HMM parameter file')
    parser.add_argument('function',
                        type=str,
                        choices=['g', 'v', 'f', 'b', 'sup', 'unsup'],
                        help='random generation (g), best state sequence (v), forward probability of observations (f), backward probability of observations (b), supervised learning (sup), or unsupervised learning (unsup)?')
    parser.add_argument('obsfile', type=str, help='file with list of observations')
    parser.add_argument('--values_to_cross_cv', type=float, default=0.1, help='values_to_cross_cv threshold for EM')
    parser.add_argument('--number_of_converges', type=int, default=0,
                        help='number of random number_of_converges for EM')
    parser.add_argument('--lock_for_emitted', type=bool, default=False,
                        help='should the emission parameters be frozen during EM training?')
    parser.add_argument('--lock_for_transmitted', type=bool, default=False,
                        help='should the transition parameters be frozen during EM training?')
    args = parser.parse_args()
    model = HMM()
    model.load(args.paramfile)
    if args.function == 'v':
        total_available_corpus = load_observations(args.obsfile)
        outputfile = os.path.splitext(args.obsfile)[0] + '.tagged.obs'

        with codecs.open(outputfile, 'w', 'utf8') as o:
            for observation in total_available_corpus:
                model.viterbi(observation)
                o.write(str(observation))
    elif args.function == 'f':
        total_available_corpus = load_observations(args.obsfile)
        outputfile = os.path.splitext(args.obsfile)[0] + '.forwardprob'
        with open(outputfile, 'w') as o:
            for observation in total_available_corpus:
                o.write(str(model.probability_forward1ability(observation)) + '\n')
    elif args.function == 'b':
        total_available_corpus = load_observations(args.obsfile)
        outputfile = os.path.splitext(args.obsfile)[0] + '.backwardprob'
        with open(outputfile, 'w') as o:
            for observation in total_available_corpus:
                o.write(str(model.backward_probability(observation)) + '\n')
    elif args.function == 'sup':
        total_available_corpus = load_observations(args.obsfile)
        model.learn_supervised(total_available_corpus, args.lock_for_emitted, args.lock_for_transmitted)
        corpusbase = os.path.splitext(os.path.basename(args.obsfile))[0]
        model.dump(args.paramfile + '.' + corpusbase + '.trained')
    elif args.function == 'unsup':
        total_available_corpus = load_observations(args.obsfile)
        log_likelihood = model.unsupervised_learning_likelihood(total_available_corpus, args.values_to_cross_cv,
                                                                args.lock_for_emitted, args.lock_for_transmitted,
                                                                args.number_of_converges)
        print
        "The final model's log likelihood is", log_likelihood
        corpusbase = os.path.splitext(os.path.basename(args.obsfile))[0]
        model.dump(args.paramfile + '.' + corpusbase + '.trained')
    elif args.function == 'g':
        with codecs.open(args.obsfile, 'w', 'utf8') as o:
            for _ in range(20):
                o.write(str(model.generate(random.randint(1, 15))))


if __name__ == '__main__':
    main()







