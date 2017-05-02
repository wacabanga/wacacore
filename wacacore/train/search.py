import subprocess
import os
from wacacore.train.hyper import rand_product


def stringify(k, v):
    """Turn a key value into command line argument"""
    if v is True:
        return "--%s" % k
    elif v is False:
        return ""
    else:
        return "--%s=%s" % (k, v)


def make_batch_string(options):
    """Turn options into a string that can be passed on command line"""
    batch_string = [stringify(k, v) for k, v in options.items()]
    return batch_string


def run_sbatch(options, file_path, bash_script='run.sh'):
    """Execute sbatch with options"""
    run_str = ['sbatch', bash_script, file_path] + make_batch_string(options)
    print(run_str)
    subprocess.call(run_str)


def rand_hyper_search(options, file_path, var_options_keys, nsamples, prefix,
                      nrepeats):
    """Randomized hyper parameter search"""
    rand_product(lambda options: run_sbatch(options, file_path),
                 options, var_options_keys, nsamples, prefix, nrepeats)
