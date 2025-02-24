import music21 as m21
import pandas as pd
import re
import os
import typing

def abc_to_dataframe(abc_text: str, 
                     chords_style='simple_numerals',
                     include_mode: bool=False):
    if chords_style != 'simple_numerals':
        raise Exception('Other chord styles have not been implemented.')

    # Prepare to collect chords and notes as `states`
    states_keys = ['measure', 'beat', 'chord', 'melody']
    states = {key: [] for key in states_keys}

    current_key = None
    current_melody = None
    current_chord = None

    def _append_line(
            measure: int, 
            beat: float, 
            chord: m21.harmony.ChordSymbol, 
            melody: Union[m21.note.Note, m21.note.Rest],
            key: m21.key.Key,
            include_mode: bool
        ):
        states['measure'].append(measure)
        states['beat'].append(beat)
        states['chord']: chord.romanNumeral.romanNumeral
        
        # If rest, use an hyphen
        if isinstance(melody, m21.note.Rest):
            states['melody'] = '-'
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
            states['melody'] = note.midi - key_shift 

    for index, element in enumerate(part.flatten()):
        if isinstance(element, m21.key.Key):
            current_key = element
        if isinstance(element, m21.harmony.ChordSymbol):
            # Make sure the chord is relative to the current key
            if current_key is None:
                raise Exception('Current key unknown')
            
            element.key = current_key
            current_chord = element
            
            states.append(pd.Series({
                'measure': element.measureNumber,
                'beat': element.beat,
                'chord': element.romanNumeral.romanNumeral
            })) 
            # TODO: use _append_line() to add states wheneverf the note or chord changes.

    
    return pd.DataFrame(states)

def abc_to_states(abc_text: str, chords_per_state: int):
    abc_score = m21.converter.parse(abc_text, format='abc')
    # The score should only have one part, which contains notes, chords, etc.
    part = abc_score.parts[0]

    # Flatten the part to iterate through everything in it.
    # TODO: See also m21.Stream.notes, which is an iterator only containing notes, chords, etc. No rests.
    # TODO: It looks like there's also a way to get timestamps in seconds. See Stream.seconds
    # TODO: use element.quarterLength to "standardize" meter
    for element in part.flatten().notesAndRests:
        if isinstance(element, m21.chord.Chord):
            measure = element.measureNumber
            beat = element.quarterLength
            print(f"\nChord: {element}\n\tMeasure {measure}\n\tDuration {beat}")
        elif isinstance(element, m21.note.Note):
            measure = element.measureNumber
            beat = element.quarterLength
            print(f"\nNote:  {element}\n\tMeasure {measure}\n\tDuration {beat}")
        elif isinstance(element, m21.note.Rest):
            measure = element.measureNumber
            beat = element.quarterLength
            print(f"\nRest:  {element}\n\tMeasure {measure}\n\tDuration {beat}")

    # Prepare to collect states
    states = []
    current_key = None

    if chords_per_state != 1:
        raise Exception('Not implemented')
    for index, element in enumerate(part.flatten()):
        if isinstance(element, m21.key.Key):
            current_key = element
        if isinstance(element, m21.harmony.ChordSymbol):
            # Make sure the chord is relative to the current key
            if current_key is None:
                raise Exception('Current key unknown')
            element.key = current_key
            states.append(pd.Series({
                'measure': element.measureNumber,
                'beat': element.beat,
                'chord': element.romanNumeral.romanNumeral
            }))
    
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

    abc_texts = [train_set.sample(1)['output'].item() for _ in range(100)]
    for abc_text in abc_texts:
        abc_parsed = abc_to_states(abc_text, 1)
        print(abc_parsed)
        print()