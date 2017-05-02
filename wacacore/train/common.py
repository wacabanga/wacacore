"""Common functions for training"""
from wacacore.util.io import mk_dir
from wacacore.util.misc import inn
import tensorflow as tf
from tensorflow import Graph, Tensor, Session
from typing import List, Generator, Callable, Sequence
import numpy as np
import pdb

def do_load(options):
    return 'load' in options and options['load'] and 'params_file' in options

def do_save(options):
    return inn(options, 'save', 'dirname', 'datadir')

def prep_load(sess, saver, params_file):
    saver.restore(sess, params_file)

def prep_save(dirname, datadir):
    return mk_dir(dirname=dirname, datadir=datadir)

def gen_fetch(sess: Session,
              debug=False,
              **kwargs):
    init = tf.global_variables_initializer()
    sess.run(init)
    fetch = {}
    if debug:
        fetch['check'] = tf.add_check_numerics_ops()

    return fetch


def variable_summaries(losses):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        for loss_name, loss_tensor in losses.items():
            tf.summary.scalar(loss_name, loss_tensor)
    return tf.summary.merge_all()


def setup_file_writers(summaries_dir, sess):
    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    train_writer = tf.summary.FileWriter(summaries_dir, sess.graph)
    return [train_writer]
    # test_writer = tf.summary.FileWriter(summaries_dir + '/test')

def updates(loss: Tensor, var_list, options):
    """Generate an update tensor which when executed will perform an
    optimization step
    Args:
        loss: a loss tensor to be minimized
    """
    with tf.name_scope("optimization"):
        if options['update'] == 'momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate=options['learning_rate'],
                                                   momentum=options['momentum'])
        elif options['update'] == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=options['learning_rate'])
        elif options['update'] == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=options['learning_rate'])
        else:
            assert False, "Unknown loss minimizer"
        update_step = optimizer.minimize(loss, var_list=var_list)
        return optimizer, update_step

def gen_feed_dict(generators, remove_update=False):
    """Generate fetoggle ed dict for optimization step"""
    feed_dict = {}
    for gen in generators:
        sub_feed_dict = next(gen)
        feed_dict.update(sub_feed_dict)

    if remove_update is True:
        feed_dict.pop('update_step', None)
    return feed_dict

# @profile
def train_loop(sess: Session,
               loss_updates: Sequence[Tensor],
               fetch,
               train_generators: Sequence[Generator],
               test_generators: Sequence[Generator],
               loss_ratios: Sequence[int]=None,
               test_every=100,
               num_iterations=100000,
               callbacks=None,
               **kwargs):
    """Perform training by `num_iterations` optimizaiton steps
    Args:
        sess: Tensorflow session
        loss_updates: gradient update tensors
        fetch: Dictionary/List whose leaves are tensors to record every it.
        train_generators: a sequence of train_generators. A generator should return
            a dict {tensor: value}.  The union of all the dicts is passed as
            feed_dict in the gradient steps.
        test_generators: Like generators but for test_data
        loss_ratios: Different weights for different loss terms
        num_iterations: number of iterations to run
        test_every: evaluate test data set test_every iterations
        num_iterations: number of iterations
        callbacks: functions to be called with result from fetch
    """
    #pdb.set_trace()
    # Default 1 for loss_ratios and normalize
    loss_ratios = [1 for i in range(len(loss_updates))] if loss_ratios is None else loss_ratios
    loss_ratios = loss_ratios / np.sum(loss_ratios)
    callbacks = [] if callbacks is None else callbacks

    # Prepare dict to be passed to callbacks
    callback_dict = {}
    callback_dict.update(kwargs)
    callback_dict.update({'sess': sess})
    state = {}

    # Main loop
    for i in range(num_iterations):
        # Generate input
        curr_fetch = {}
        curr_fetch.update(fetch)
        curr_fetch["update_loss"] = np.random.choice(loss_updates, p=loss_ratios)
        feed_dict = gen_feed_dict(train_generators)
        fetch_res = sess.run(curr_fetch, feed_dict=feed_dict)
        print("fetch_res")
        print(fetch_res)

        # Evaluate on test data every test_every iterations
        if test_generators is not None and (i % test_every == 0 or i == num_iterations - 1):
            test_feed_dict = gen_feed_dict(test_generators, True)
            test_fetch_res = sess.run(fetch, feed_dict=test_feed_dict)
            fetch_res['test_fetch_res'] = test_fetch_res
            if 'loss' in test_fetch_res:
                print("Test Loss", test_fetch_res['loss'])
            if 'losses' in test_fetch_res:
                print("Test Losses", test_fetch_res['losses'])

        # Do all call backs
        for cb in callbacks:
            cb(fetch_res, feed_dict, i, num_iterations=num_iterations, state=state, **callback_dict)
        print("Iteration: ", i)
        if 'loss' in fetch_res:
            print(fetch_res['loss'])
        if 'losses' in fetch_res:
            print(fetch_res['losses'])
