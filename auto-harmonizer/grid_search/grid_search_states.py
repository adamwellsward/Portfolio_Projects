from abc_utils import *

import pandas as pd
from hmmlearn import hmm

import numpy as np
from sklearn.metrics import accuracy_score

from itertools import product
from matplotlib import pyplot as plt
from tqdm import tqdm
from tqdm import auto
from itertools import product

def load_song_subset(train_set, train_lengths, indices):
    """
    Create a subset of the train set and return both it and the lengths of the songs in the train set.
    """
    # get the cumulative lengths of the songs to index into the set
    end_positions = np.cumsum(train_lengths)
    positions = np.insert(end_positions, 0, np.array([0]))

    # loop through the indices and splice out the needed songs
    songs = []
    for i in indices:
        song = train_set.iloc[positions[i] : positions[i+1]]
        songs.append(song)

    # return songs and lengths
    return pd.concat(songs), train_lengths.iloc[indices]

def ffill_obs(melody_obs: np.ndarray, unique_obs: dict) -> np.ndarray:
    # make a smaller array out of the unique observations
    possible_obs = set(unique_obs.flatten())

    df_melody_obs = pd.Series(melody_obs)
    df_melody_obs[~df_melody_obs.isin(possible_obs)] = np.nan

    # fill forward first to fill all the holes
    df_melody_obs.ffill(inplace=True)

    # then fill backward to catch the case where
    # the beginning is empty
    df_melody_obs.bfill(inplace=True)

    return df_melody_obs.values.flatten()

def chord_accuracy(full_pred: np.array, true_states: np.array, num_chords: int=None, num_notes: int=None):
    '''
    Given the predicted matrix of states, compute the misclassification rate compared with the true_observations.
    Could be edited in the future to also compute the accuracy of our predicted note sequence.
    '''
    # check to make sure these are specified correctly
    if num_chords is None:
        raise ValueError("num_chords must be specified")
    if num_notes is None:
        raise ValueError("num_notes must be specified")
    
    # obtain the actual predicted chords 
    pred_chords = full_pred[:, num_chords-1]
    true_chords = true_states[:len(pred_chords), num_chords-1]

    # obtain the accuracy
    chord_acc = accuracy_score(true_chords, pred_chords)

    return chord_acc

def fit_model(train_set: pd.DataFrame, train_lengths: pd.Series, num_chords: int=1, num_notes: int=0, subset: bool=False, indices=None, lam: int=None, trans_prior: float=0.1, emissions_prior: float=0.1):    
    """ 
    Takes in the train set and parameters for the state space and returns the trained model, along with all of the dictionaries needed to decode the model as a tuple.

    To train on a smaller subset of the full train set, use the subset argument and pass in the indices needed. Uses the load_song_subset function.
    """
    # check if we want to do a subset of the full train set; if so, perform it
    if subset:
        # check that indices are specified; raise and error if not
        if indices is None:
            raise ValueError("Indices must be specified if subset=True")
        train_set, _ = load_song_subset(train_set, train_lengths, indices)

    # obtain the states and observations from the songs
    true_states, true_observations = dataframe_to_states(train_set, num_chords, num_notes)
    
    # create the transition matrices for the model
    transition_matrix, emission_probs, unique_states, unique_obs, states_to_index, observation_to_index = states_to_transition(true_states, true_observations, lam, trans_prior, emissions_prior)

    # now initialize the model and set the matrices for it
    model = hmm.CategoricalHMM(n_components=transition_matrix.shape[0], init_params='')
    model.transmat_ = transition_matrix.T
    model.emissionprob_ = emission_probs.T

    starting_state = -np.ones(unique_states.shape[1])
    starting_state_index = states_to_index[tuple(starting_state)]

    start_probs = np.zeros(transition_matrix.shape[0])
    start_probs[starting_state_index] = 1
    
    model.startprob_ = start_probs

    # return the model,  the dictionaries
    return model, (unique_states, unique_obs, states_to_index, observation_to_index)

def predict_states(model: hmm.CategoricalHMM, all_dicts: tuple, observation: np.ndarray, song_lengths: list):
    """
    Uses the model to decode an observation. The all_dicts tuple should contain the model dictionaries returned from fit_model
    Returns the predicted states.
    """
    # unpack the tuple to get what we need
    unique_states, unique_obs, _, observation_to_index = all_dicts

    # perform a forward fill on the observation in case there are any values in it that we have never seen before
    observation = ffill_obs(observation, unique_obs)
    
    # get the indices of the observation
    observation_indices = np.array([int(observation_to_index[(o,)]) for o in observation])

    # get the predicted state indices
    _, pred_indices = model.decode(observation_indices.reshape(-1, 1), lengths=song_lengths)

    # use the unique_states dictionary to take the indices to the actual states
    pred_states = unique_states[pred_indices, :]

    # return the predicted states
    return pred_states

def get_prediction(model, all_dicts, val_set: pd.DataFrame, val_lengths: pd.Series, num_chords: int=1, num_notes: int=0, subset: bool=False, indices: list=None, do_print: bool=True):
    """
    Perform a full prediction using a trained model. 
    
    Parameters
    ----------
    model (hmm.CategoricalHMM): The trained model 
    all_dicts (tuple): Contains the arrays and dictionaires needed to transform the prediction into readable output 
    val_set (pd.DataFrame): The data to predict on 
    val_lengths (pd.Series): The lengths of the songs in the data
    num_chords (int): Number of chords in the state space; defaults to 1
    num_notes (int): Number of notes in the state space; defaults to 0
    subset (bool): Whether or not to take a subset of the val_set; defaults to False, 
    indices (list): The indices for the subset, only used if subset=True 
    do_print (bool): Whether or not to print the true and predicted states side by side, defaults to True

    Returns: the predicted states, the true states, and the accuracy
    """
    if subset:
        val_set, val_lengths = load_song_subset(val_set, val_lengths, indices)
    
    true_states, new_song_obs = dataframe_to_states(val_set, num_chords, num_notes)

    # get the predicted states (chop off the first element of the songs because it added a 0)
    pred_states = predict_states(model, all_dicts, new_song_obs[1:], val_lengths.values.flatten().tolist())

    
    # print the results, then return the results and the accuracy
    if do_print:
        print("Pred\t\tTrue")
        cumul = np.cumsum(val_lengths.values)
        for i in range(len(pred_states)):
            if i in set(cumul):
                print("----- New Song -----")
            print(f"{pred_states[i]}\t\t{true_states[i]}")

    # get the accuracy
    accuracy = chord_accuracy(pred_states, true_states, num_chords, num_notes)
    print("Accuracy:", accuracy)

    return pred_states, true_states, accuracy



if __name__ == '__main__':
    train_df, train_len_series, train_indicies_series, val_df, val_len_series, val_indicies_series = load_datasets()

    n_chords_range = [1, 2]
    n_melody_range = [0, 1]
    transmat_prior = [5, 20, 50, 200]
    emission_prior = [5, 20, 50, 200]

    parameter_range = list(product(n_chords_range, n_melody_range))

    validation_accuracy_sets = []

    for n_chords, n_melody in tqdm(parameter_range, desc='Grid-searching over possible state configurations'):
        validation_accuracies = []
        plt.figure()
        print(f"Number of Chords: {n_chords}", f"Number of Melody Notes: {n_melody}")
        
        for t_prior, e_prior in product(transmat_prior, emission_prior):
            # Fit a model with these parameters
            model, all_dicts = fit_model(train_df, train_len_series, n_chords, n_melody, trans_prior=t_prior, emissions_prior=e_prior)
            # Validate
            pred_states, true_states, accuracy = get_prediction(model, all_dicts, val_df, val_len_series, n_chords, n_melody, do_print=False)
            print(f'Accuracy with T={t_prior}, E={e_prior}:	{accuracy}')
            validation_accuracies.append(accuracy)
        
        val_accuracy_matrix = np.array(validation_accuracies).reshape(len(transmat_prior), len(emission_prior))
        plt.imshow(val_accuracy_matrix, aspect='auto')

        plt.xticks(ticks=np.arange(len(emission_prior)), labels=emission_prior)
        plt.yticks(ticks=np.arange(len(transmat_prior)), labels=transmat_prior)
        plt.xlabel('Emission Prior')
        plt.ylabel('Transition Prior')
        plt.colorbar(label='Accuracy')

        plt.savefig(f'plots/gridsearch_results_n_chords_{n_chords}_n_melody_{n_melody}.png')

        validation_accuracy_sets.append(validation_accuracies)
        # TODO: put into dataframe: trans prior, emission prior, num chords, num melody