"""Hyper Parameter Search"""
from wacacore.util.misc import extract, dict_prod
import numpy as np

def rand_product(run_me, options, var_option_keys, prefix='', nrepeats=1):
    """Train parametric inverse and vanilla neural network with different
    amounts of data and see the test_error
    Args:
        run_me: function to call, should execute test and save stuff
        Options: Options to be passed into run_me
        var_option_keys: Set of keys, where options['keys'] is a sequence
            and we will vary over cartesian product of all the keys

    """
    _options = {}
    _options.update(options)
    var_options = extract(var_option_keys, options)

    for i in range(nrepeats):
        var_options_prod = list(dict_prod(var_options))
        the_time = time.time()
        for j, prod in enumerate(var_options_prod):
            dirname = "%s_%s_%s_%s" % (prefix, str(the_time), i, j)
            _options['dirname'] = dirname
            _options.update(prod)
            run_me(_options)

def test_everything(run_me, options, var_option_keys, prefix='', nrepeats=1):
    """Train parametric inverse and vanilla neural network with different
    amounts of data and see the test_error
    Args:
        run_me: function to call, should execute test and save stuff
        Options: Options to be passed into run_me
        var_option_keys: Set of keys, where options['keys'] is a sequence
            and we will vary over cartesian product of all the keys

    """
    _options = {}
    _options.update(options)
    var_options = extract(var_option_keys, options)

    for i in range(nrepeats):
        var_options_prod = dict_prod(var_options)
        the_time = time.time()
        for j, prod in enumerate(var_options_prod):
            dirname = "%s_%s_%s_%s" % (prefix, str(the_time), i, j)
            _options['dirname'] = dirname
            _options.update(prod)
            run_me(_options)
