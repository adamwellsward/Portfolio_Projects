import music21 as m21
import pandas as pd
import numpy as np
import re
import os
import typing
from tqdm import tqdm
from typing import Union

ROMAN_NUMERAL_MAP = {
    "-": 0, 

    "I": 1, "#I": 2, "bII": 2, "II": 3, "#II": 4, "bIII": 4, "III": 5, 
    "#III": 6, "bIV": 6, "IV": 6, "#IV": 7, "bV": 7, "V": 8, "#V": 9, 
    "bVI": 9, "VI": 10, "#VI": 11, "bVII": 11, "VII": 12, "bI": 12,

    "i": 13, "#i": 14, "bii": 14, "ii": 15, "#ii": 16, "biii": 16, "iii": 17, 
    "#iii": 18, "biv": 18, "iv": 18, "#iv": 19, "bv": 19, "v": 20, "#v": 21, 
    "bvi": 21, "vi": 22, "#vi": 23, "bvii": 23, "vii": 24, "bi": 24
}

CHORD_INT_MAP = {v: k for k, v in ROMAN_NUMERAL_MAP.items()}
    
def abc_to_dataframe(abc_text: str, 
                     chords_style='simple_numerals',
                     include_mode: bool=False):
    if chords_style != 'simple_numerals':
        raise Exception('Other chord styles have not been implemented.')

    # Prepare to collect chords and notes as `states`
    states_keys = ['measure', 'beat', 'chord', 'melody']
    states = {key: [] for key in states_keys}

    def _append_line(
            measure: int, 
            beat: float, 
            chord: m21.harmony.ChordSymbol, 
            melody: Union[m21.note.Note, m21.note.Rest],
            key: m21.key.Key,
            include_mode: bool = False
        ):
        """
        Append a new state to the dictionary `states`.

        The "melody" added to the dictionary will be 0 if `melody` is a rest,
        and between 1 and 139 if it is a note.

        The "chord" added to the dictionary will be 0 if there is no chord, and
        between 1 and 24 otherwise, as determined by ROMAN_NUMERAL_MAP.
        """
        chord_repr = chord.romanNumeral.romanNumeral if chord else '-'
        
        # If rest, use a -1 (MIDI mins at 0)
        if not melody:
            melody_repr = 0
        elif isinstance(melody, m21.note.Rest):
            melody_repr = 0
        elif isinstance(melody, m21.note.Note):
            # TODO: add more modes.
            mode_shift = 0  
            if include_mode:
                mode = key.mode
                if mode == 'major':
                    mode_shift = 0
                elif mode == 'minor':
                    mode_shift = 9 # Shift up to avoid negative values (rest is 0)
                else:
                    raise Exception(f'Mode is {mode}. Not major or minor')
            key_shift = (key.tonic.midi % 12) + mode_shift
            melody_repr = melody.pitch.midi + key_shift + 1 # Add 1 to make room so that 0 can represent a rest
        else:
            raise Exception('Melody should be a rest or a note')
            
        states['measure'].append(measure)
        states['beat'].append(beat)
        states['chord'].append(ROMAN_NUMERAL_MAP[chord_repr])
        states['melody'].append(melody_repr) 

    # Variables to collect the pieces of a new "state" before appending it to the `states` dictionary
    current_key = None
    current_melody = None
    current_chord = None
    current_measure = None
    current_beat = None
    
    def not_meaningful(element):
        """ Check if we want to bother processing an element. """
        return isinstance(
            element,
            (
                m21.spanner.RepeatBracket,
                m21.bar.Repeat
            )
        )
    
    # Iterate through part, collecting states
    part = m21.converter.parse(abc_text, format='abc').parts[0]
    for index, element in enumerate(part.flatten()):
        if not_meaningful(element): 
            continue
            
        this_beat = element.beat
        this_measure = element.measureNumber

        # The first time an object has measure and beat, use that to initialize current_beat and current_measure
        if current_beat is None and current_measure is None and this_beat is not None and this_measure is not None:
            current_beat = this_beat
            current_measure = this_measure

        # If the time in the music has changed, append the old stuff
        if (this_measure != current_measure or this_beat != current_beat) and (current_chord is not None or current_melody is not None): 
            # Append the last new state
            _append_line(
                current_measure,
                current_beat,
                current_chord,
                current_melody,
                current_key
            )
            # Update current measure
            current_measure = this_measure
            current_beat = this_beat
        
        
        if isinstance(element, m21.key.Key):
            current_key = element

        elif isinstance(element, m21.harmony.ChordSymbol):
            # Make sure the chord is relative to the current key
            if current_key is None:
                raise Exception('Current key unknown')
            
            element.key = current_key
            current_chord = element
            
        elif isinstance(element, (m21.note.Note, m21.note.Rest)):
            if current_key is None:
                raise Exception('Current key unknown')
            
            element.key = current_key
            current_melody = element
    
    return pd.DataFrame(states).astype({'melody': int})

def dataframe_to_states(song_df: pd.DataFrame, chords_per_state: int, melody_per_state: int):
    """
    Convert a dataframe created by `abc_to_dataframe` to a numpy matrix of ints
    representing the state over time, where the state can encode the "history"
    of the chord and melody to some degree determined by `chords_per_state` and
    `melody_per_state`.
    """
    if melody_per_state == 0 and chords_per_state == 0:
        raise Exception('Cannot return empty state array')
    elif melody_per_state < 0 or chords_per_state < 0:
        raise Exception('Cannot process negative numbers of chords or melody notes')
        
    n_rows = len(song_df) + 1  # +1 allows initial row to be empty
    
    # Initialize chord and melody states, handling cases where they might be 0
    chord_states = np.empty((n_rows, chords_per_state), dtype=int) if chords_per_state > 0 else np.empty((n_rows, 0), dtype=int)
    melody_states = np.empty((n_rows, melody_per_state), dtype=int) if melody_per_state > 0 else np.empty((n_rows, 0), dtype=int)

    if chords_per_state > 0:
        chord_states[0, :] = 0  # Start with zeros (rests)
    if melody_per_state > 0:
        melody_states[0, :] = 0  # Start with zeros (rests)

    for i, (_, row) in enumerate(song_df.iterrows()):
        if chords_per_state > 0:
            chord_states[i+1, 0:-1] = chord_states[i, 1:]  # Shift left
            chord_states[i+1, -1] = row['chord']  # Append new chord
        
        if melody_per_state > 0:
            melody_states[i+1, 0:-1] = melody_states[i, 1:]  # Shift left
            melody_states[i+1, -1] = row['melody']  # Append new melody
    
    # Concatenate only if both exist, otherwise return the non-empty one
    if chords_per_state > 0 and melody_per_state > 0:
        return np.hstack([chord_states, melody_states])[1:]  # Drop first row
    elif chords_per_state > 0:
        return chord_states[1:]
    elif melody_per_state > 0:
        return melody_states[1:]


def dataset_to_abc(dataset_abc_text: str, label, reference_number):
    """ 
    Given a poorly formatted string of ABC from the dataset, reformats metadata
    so that ABC readers can parse it well.
    """
    # Remove extra whitespace
    fixed_text = dataset_abc_text.strip()
    # Remove task tag
    fixed_text = re.sub(r'^%%\w+\s*', '', fixed_text)
    # Split metadata from the tune
    metadata, tune = fixed_text.split('|', 1)
    # Add newlines to metadata only where needed (after the first occurrence of each metadata key)
    metadata = re.sub(r"(\S:\S+)(?=\s)(?!\n)", r"\1\n", metadata)
    
    # Add reference number and title (only add if they don't exist already)
    if not re.search(r'^\s*X:', metadata):
        metadata = f'X:{reference_number}\n' + metadata
    if not re.search(r'^\s*T:', metadata):
        metadata = f'T:{label} {reference_number}\n' + metadata
    
    ## Fix key changes throughout tune
    ## TODO: still not processing key changes in abc_to_dataframe
    #key_regex = re.compile(r'\[K:([^\]]{1,2})\]')
    #tune = key_regex.sub(r'\n[K:\1]\n', tune)

    fixed_text = metadata + '|' + tune
    return fixed_text

def load_dataset_df(*, train: bool = True, local: bool = True) -> pd.DataFrame:
    """ 
    Loads the melodyhub dataset from HuggingFace as a pandas DataFrame. Must pip
    install `fsspec` and `huggingface_hub`.

    Keyword parameters:

    - train (bool): True for the train set, False for the validation set

    - local (bool): True to use the data in `./melodyhub`, False to download directly
    from HuggingFace 
    """
    splits = {'train': 'train.jsonl', 'validation': 'validation.jsonl'}
    # Select dataset based on `train` parameter
    split = 'train' if train else 'validation'
    # Set path based on `local` parameter
    path = "./melodyhub/" if local else "hf://datasets/sander-wood/melodyhub/"
    dataset = pd.read_json(
        path + splits[split],
        lines=True
    )
    return dataset

def load_harmonization_train_test(local=True):
    """ 
    Loads both train and test (validation) melodyhub datasets, returning only
    the lines for the 'harmonization' task. Puts the ABC input column into
    standard ABC format, with appropriate newlines.
    """
    train_set = load_dataset_df(train=True, local=local)
    test_set = load_dataset_df(train=False, local=local)

    train_set = train_set[train_set['task'] == 'harmonization']
    test_set = test_set[test_set['task'] == 'harmonization']

    for label, dataset in [('Train tune', train_set), ('Test tune', test_set)]:
        # Format tune metadata correctly
        dataset['input'] = dataset.apply(
            lambda row: dataset_to_abc(row['input'], label, row.name), axis=1
        )
        dataset['output'] = dataset.apply(
            lambda row: dataset_to_abc(row['output'], label, row.name), axis=1
        )
        
    return train_set, test_set

if __name__ == '__main__':
    train_set, _ = load_harmonization_train_test()

    bad_songs = []
    for i, row in tqdm(list(train_set.iterrows())):
        try:
            abc_to_dataframe(row['output'])
        except:
            bad_songs.append(row)
            print(row['output'])

    df = abc_to_dataframe(train_set.iloc[12]['output'])
    with open('bad_songs.txt', 'a') as file:
        file.write(str([song.index for song in bad_songs]))
        for song in bad_songs:
            file.write(song['output'])

    print(dataframe_to_states(df, 3, 2))
    print()