"""Generalization Test"""

# So its better to prevent train_loop from becoming this huge mess
# and instead pass in a callback
# a lens would be even better but there you go, we dont have that
import os
import pickle
import csv

def save_it_csv(params, fname):
    f = open(fname, 'w')
    writer = csv.writer(f)
    for key, value in params.items():
        writer.writerow([str(key), str(value)])
    f.close()


def pickle_it(data, savedir):
    with open(savedir, 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    f.close()

def save_callback(fetch_data,
                  feed_dict,
                  i: int,
                  compress=False,
                  pfx = '',
                  **kwargs):
    save = 'save' in kwargs and kwargs['save'] is True
    if  save:
        sess = kwargs['sess']
        savedir = kwargs['savedir']
        print("Saving")
        saver = kwargs['saver']
        stat_sfx = "%sit_%s_fetch.pickle" % (pfx, i)
        stats_path = os.path.join(savedir, stat_sfx)
        pickle_it(fetch_data, stats_path)

        # Params
        params_sfx = "%sit_%s_params" % (pfx, i)
        path = os.path.join(savedir, params_sfx)
        saver.save(sess, path)


def save_every_n(fetch_data, feed_dict, i, save_every=100, **kwargs):
    state = kwargs['state']
    if 'all_loss' in state:
        for k, v in fetch_data['losses'].items():
            state['all_loss'][k] += [v]
    else:
        state['all_loss'] = {}
        for k, v in fetch_data['losses'].items():
            state['all_loss'][k] = [v]


    if i % save_every == 0:
        save_callback(fetch_data, feed_dict, i, **kwargs)

import sys
import math
def nan_cancel(fetch_data,
              feed_dict,
              i: int,
              **kwargs):
    """Cancel on NaN in loss"""
    if 'loss' in fetch_data and math.isnan(fetch_data['loss']):
        print("NaNs found")
        sys.exit()

def save_everything_last(fetch_data,
                         feed_dict,
                         i: int,
                         **kwargs):
    """Saves everything on the last iteration"""
    num_iterations = kwargs['num_iterations']
    save = 'save' in kwargs and kwargs['save'] is True
    if save and i == (num_iterations - 1):
        savedir = kwargs['savedir']
        path = os.path.join(savedir, "state.pickle")
        pickle_it(kwargs['state'], path)
        save_callback(fetch_data, feed_dict, i, pfx='last_', **kwargs)


def save_options(fetch_data, feed_dict, i: int, **kwargs):
    """Save the options"""
    save = 'save' in kwargs and kwargs['save'] is True
    if save and i == 0:
        savedir = kwargs['savedir']
        options_dir = "options"
        valid_types = [list, str, float, int]
        to_save_options = {}
        for k, v in kwargs.items():
            if any((isinstance(v, typ) for typ in valid_types)):
                to_save_options[k] = v
        stats_path = os.path.join(savedir, options_dir)
        pickle_it(to_save_options, "%s.pickle" % stats_path)
        save_it_csv(to_save_options, "%s.csv" % stats_path)
