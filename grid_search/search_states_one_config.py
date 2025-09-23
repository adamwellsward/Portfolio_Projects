from grid_search_states import *
import numpy as np
import argparse

def job_with_params(n_chords, n_melody, t_prior, e_prior):
    print(f'Training model with\nn_chords: {n_chords}\nn_melody: {n_melody}\nt_prior:  {t_prior}\ne_prior:  {e_prior}')
    print()
    train_df, train_len_series, train_indicies_series, val_df, val_len_series, val_indicies_series = load_datasets()
    model, all_dicts = fit_model(train_df, train_len_series, n_chords, n_melody, trans_prior=t_prior, emissions_prior=e_prior)

    # Validate
    pred_states, true_states, accuracy = get_prediction(model, all_dicts, val_df, val_len_series, n_chords, n_melody, do_print=False)

    filepath = f'grid_search/{n_chords}_{n_melody}_{t_prior}_{e_prior}.txt'
    with open(filepath, 'w') as f:
        # Write job parameters
        f.write(f'Accuracy: {accuracy}\n\n')
        f.write(f'n_chords: {n_chords}\n')
        f.write(f'n_melody: {n_melody}\n')
        f.write(f't_prior: {t_prior}\n')
        f.write(f'e_prior: {e_prior}\n')

        # Write a sample of the predictions of the model
        n_sample = 3000
        f.write('\n')
        f.write(f'~ First {n_sample} predicted states ~\n')
        f.write('Predicted\tTrue\n')
        for pred, true in zip(pred_states[:n_sample], true_states[:n_sample]):
            f.write(f'{pred}\t{true}\n')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run grid search with specified parameters.')
    parser.add_argument('--n_chords', type=int, required=True, help='Number of chord states')
    parser.add_argument('--n_melody', type=int, required=True, help='Number of melody states')
    parser.add_argument('--t_prior', type=float, required=True, help='Transition prior')
    parser.add_argument('--e_prior', type=float, required=True, help='Emission prior')
    
    args = parser.parse_args()
    job_with_params(args.n_chords, args.n_melody, args.t_prior, args.e_prior)
