from abc_utils import *

import pandas as pd
from hmmlearn import hmm

import numpy as np
from sklearn.metrics import accuracy_score

from itertools import product
from matplotlib import pyplot as plt
from tqdm import tqdm
from tqdm import auto

def load_song_subset(train_set, train_lengths, indices):
    """
    Create a subset of the train"""
    end_positions = np.cumsum(train_lengths)
    positions = np.insert(end_positions, 0, np.array([0]))
    songs = []
    for i in indices:
        song = train_set.iloc[positions[i] : positions[i+1]]
        songs.append(song)
    # return songs and lengths
    return pd.concat(songs)

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

    # # obtain the actual predicted notes from the state
    # if num_notes != 0: 
    #     pred_notes = full_pred[:, -1]
    #     true_notes = true_states[:len(pred_notes), -1]

    #     # obtain the accuracy
    #     note_acc = accuracy_score(true_notes, pred_notes)
    # else:
    #     note_acc = None
    return chord_acc

def fit_model(train_set: pd.DataFrame, train_lengths: pd.Series, num_chords: int=1, num_notes: int=0, subset: bool=False, indices=None):
    """ 
    Takes in the train set and parameters for the state space and returns the trained model, along with all of the dictionaries needed to decode the model as a tuple.

    To train on a smaller subset of the full train set, use the subset argument and pass in the indices needed. Uses the load_song_subset function.
    """
    # check if we want to do a subset of the full train set; if so, perform it
    if subset:
        # check that indices are specified; raise and error if not
        if indices is None:
            raise ValueError("Indices must be specified if subset=True")
        train_set = load_song_subset(train_set, train_lengths, indices)

    # obtain the states and observations from the songs
    true_states, true_observations = dataframe_to_states(train_set, num_chords, num_notes)

    # create the transition matrices for the model
    transition_matrix, emission_probs, unique_states, unique_obs, states_to_index, observation_to_index = states_to_transition(true_states, true_observations)

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

def predict_states(model: hmm.CategoricalHMM, all_dicts: tuple, observation: np.ndarray):
    """
    Uses the model to decode an observation. The all_dicts tuple should contain the model dictionaries returned from fit_model
    Returns the predicted states.
    """
    # unpack the tuple to get what we need
    unique_states, unique_obs, _, observation_to_index = all_dicts

    # perform a forward fill on the observation in case there are any values in it that we have never seen before
    print(accuracy_score(observation, ffill_obs(observation, unique_obs)))
    
    observation = ffill_obs(observation, unique_obs)
    # get the indices of the observation
    observation_indices = np.array([int(observation_to_index[(o,)]) for o in observation])

    # get the predicted state indices
    probability, pred_indices = model.decode(observation_indices.reshape(-1, 1))

    # use the unique_states dictionary to take the indices to the actual states
    pred_states = unique_states[pred_indices, :]

    # return the predicted states
    return pred_states

def get_prediction(model, all_dicts, val_set: pd.DataFrame, val_lengths: pd.Series, num_chords: int=1, num_notes: int=0, subset: bool=False, indices=None, do_print: bool=True):
    if subset:
        val_set = load_song_subset(val_set, val_lengths, indices)
    
    true_states, new_song_obs = dataframe_to_states(val_set, num_chords, num_notes)

    # get the predicted states
    pred_states = predict_states(model, all_dicts, new_song_obs)

    # print the results, then return the results and the accuracy
    if do_print:
        print("Pred\t\tTrue")
        for p, t in zip(pred_states, true_states):
            print(p, t)
        print()
        print()

    # get the accuracy
    accuracy = chord_accuracy(pred_states, true_states, num_chords, num_notes)
    print("Accuracy:", accuracy)

    return pred_states, true_states, accuracy



if __name__ == '__main__':
    train_df, train_len_series, train_indicies_series, val_df, val_len_series, val_indicies_series = load_datasets()
    og_dataset = OG_Dataset(
        full_dataset_dir='curated_datasets'
    )
    train_songs, val_songs, _ = og_dataset.get_abc_texts_from_indicies(
        range(0, 1000),
        range(0, 3)
    )

    train_len_series = train_len_series.iloc[:1000]
    val_len_series = train_len_series.iloc[:3]

    train_df = pd.concat([abc_to_dataframe(song) for song in tqdm(train_songs['output'], desc='Loading songs')])
    val_df = pd.concat([abc_to_dataframe(song) for song in val_songs['output']])

    n_chords_range = [1, 2, 3]
    n_melody_range = [0, 1, 2, 3]

    parameter_range = list(product(n_chords_range, n_melody_range))
    validation_accuracies = []

    for n_chords, n_melody in tqdm(parameter_range, desc='Grid-searching over possible state configurations'):
        # Fit a model with these parameters
        model, all_dicts = fit_model(train_df, train_len_series, n_chords, n_melody)
        # Validate

        pred_states, true_states, accuracy = get_prediction(model, all_dicts, val_df, val_len_series, n_chords, n_melody, do_print=True)
        print(f'Accuracy on validation set with {n_chords} chords and {n_melody} melody notes:\t{accuracy}')
        validation_accuracies.append(accuracy)

    val_accuracy_matrix = np.array(validation_accuracies).reshape(len(n_chords_range), len(n_melody_range))
    plt.imshow(val_accuracy_matrix)

    plt.xticks(ticks=np.arange(len(n_melody_range)), labels=n_melody_range)
    plt.yticks(ticks=np.arange(len(n_chords_range)), labels=n_chords_range)
    plt.xlabel('Number of chords')
    plt.ylabel('Number of melody notes')
    plt.colorbar(label='Accuracy')

    plt.savefig('gridsearch_results.png')