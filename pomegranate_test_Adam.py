# This is one block/function
def ffill_obs(melody_obs: np.ndarray, unique_obs: dict) -> np.ndarray:
    # make a smaller array out of the unique observations
    possible_obs = list(set(unique_obs.flatten()))

    df_melody_obs = pd.Series(melody_obs)
    df_melody_obs[~df_melody_obs.isin(possible_obs)] = np.nan

    # fill forward first to fill all the holes
    df_melody_obs.ffill(inplace=True)

    # then fill backward to catch the case where
    # the beginning is empty
    df_melody_obs.bfill(inplace=True)

    return df_melody_obs.values.flatten()

# This is one block
songs, song_lengths = load_songs([total_songs + 10001])
val_states, val_observations = dataframe_to_states(
    songs, 
    NUM_CHORDS, 
    NUM_NOTES
)

val_observations = ffill_obs(val_observations, unique_obs)
observation_indices = np.array([int(observation_to_index[(o,)]) for o in val_observations]) 
likelihood, pred_states = model.decode(observation_indices.reshape(-1, 1))
states = unique_states[pred_states, :]
print(np.hstack((val_states, states)))
chord_accuracy(states, val_states)