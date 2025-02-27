import music21 as m21
import pandas as pd
import re
import os
import typing
from typing import Union

def abc_to_dataframe(abc_text: str, 
                     chords_style='simple_numerals',
                     include_mode: bool=False):
    if chords_style != 'simple_numerals':
        raise Exception('Other chord styles have not been implemented.')
    
    part = m21.converter.parse(abc_text, format='abc').parts[0]

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
        chord_repr = chord.romanNumeral.romanNumeral if chord else '-'
        
        # If rest, use an hyphen
        if not melody:
            melody_repr = '-'
        elif isinstance(melody, m21.note.Rest):
            melody_repr = '-'
        elif isinstance(melody, m21.note.Note):
            # TODO: add more modes
            mode_shift = 0  
            if include_mode:
                mode = key.mode
                if mode == 'major':
                    mode_shift = 0
                elif mode == 'minor':
                    mode_shift = -3
                else:
                    raise Exception(f'Mode is {mode}. Not major or minor')
            key_shift = (key.tonic.midi % 12) + mode_shift
            melody_repr = melody.pitch.midi - key_shift
        else:
            raise Exception('Melody should be a rest or a note')
            
        states['measure'].append(measure)
        states['beat'].append(beat)
        states['chord'].append(chord_repr)
        states['melody'].append(melody_repr)

    current_key = None
    current_melody = None
    current_chord = None
    current_measure = None
    current_beat = None
    
    def not_meaningful(element):
        return isinstance(
            element,
            (
                m21.spanner.RepeatBracket,
                m21.bar.Repeat
            )
        )


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
            
        elif isinstance(element, m21.note.Note):
            if current_key is None:
                raise Exception('Current key unknown')
            
            element.key = current_key
            current_melody = element
    
    return pd.DataFrame(states)

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
    
    # Fix key changes throughout tune
    # TODO: add new lines as in 3.9 of   https://trillian.mit.edu/~jc/music/doc/ABC.html  

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

    abc_text = train_set.iloc[6]['output']
    df = abc_to_dataframe(abc_text)
    print()