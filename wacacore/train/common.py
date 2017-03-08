from typing import List, Generator, Callable
import tensorflow as tf
from tensorflow import Graph, Tensor, Session
import os

def layer_width(i, o, n, p):
    """Compute the layer width for a desired number of parameters
    Args:
        i: Length of input
        o: Length of output
        p: Desired number of parameters
        n: Number of layers
    Returns:
        Size of inner layers"""
    b = i + 1 + o + n
    a = n
    c = o - p
    inner = np.sqrt(b*b - 4*a*c)
    return (-b + inner)/(2*a), (-b - inner)/(2*a)


def prep_save(sess: Session, save: bool, dirname: str, params_file: str, load: bool):
    save_params = {}
    if save or load:
        saver = tf.train.Saver()
    if save is True:
        save_dir = mk_dir(dirname=dirname)
        save_params['save_dir'] = save_dir
        save_params['saver'] = saver = tf.train.Saver()
    if load is True:
        saver.restore(sess, params_file)
    return save_params


def gen_fetch(sess: Session,
              debug=False,
              **kwargs):
    init = tf.global_variables_initializer()
    sess.run(init)

    fetch = {}
    if debug:
        fetch['check'] = tf.add_check_numerics_ops()

    return fetch


def gen_update_step(loss: Tensor) -> Tensor:
    with tf.name_scope('optimization'):
        # optimizer = tf.train.MomentumOptimizer(0.001,
        #                                        momentum=0.1)
        optimizer = tf.train.AdamOptimizer(0.01)
        update_step = optimizer.minimize(loss)
        return update_step


def get_updates(loss: Tensor, options):
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
        update_step = optimizer.minimize(loss)
        return optimizer, update_step


def train_loop(sess: Session,
               loss_updates: Sequence[Tensor],
               fetch,
               generators: Sequence[Generator],
               test_generators,
               loss_ratios: Sequence[int]=None,
               test_every=100,
               num_iterations=100000,
               callbacks=[],
               **kwargs):
    """Perform training
    Args:
        sess: Tensorflow session
        loss_updates: gradient update tensors:
        test_generators: a sequence of generators. A generator should return
            a dict {tensor: value}.  The union of all the dicts is passed as
            feed_dict in the gradient steps.
        loss_ratios:
        num_iterations: number of iterations to run
        test_every: evaluate test data set test_every iterations
        num_iterations: number of iterations
        callbacks: functions to be called with result from fetch
    """
    # Default 1 for loss_ratios and normalize
    loss_ratios = [1 for i in range(len(loss_updates))] if loss_ratios is None else loss_ratios
    loss_ratios = loss_ratios / np.sum(loss_ratios)

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
        feed_dict = {}
        for gen in generators:
            sub_feed_dict = next(gen)
            feed_dict.update(sub_feed_dict)
        # Optimizeation Step
        fetch_res = sess.run(curr_fetch, feed_dict=feed_dict)

        # Evaluate on test data every test_every iterations
        if test_generators is not None and (i % test_every == 0 or i == num_iterations - 1):
            test_feed_dict = {}
            for gen in test_generators:
                sub_feed_dict = next(gen)
                test_feed_dict.update(sub_feed_dict)
            test_feed_dict = {k: v for k, v in test_feed_dict.items() if k != "update_step"}
            test_fetch_res = sess.run(fetch, feed_dict=test_feed_dict)
            fetch_res['test_fetch_res'] = test_fetch_res
            print("Test Loss", test_fetch_res['loss'])

        # Do all call backs
        for cb in callbacks:
            cb(fetch_res, feed_dict, i, num_iterations=num_iterations, state=state, **callback_dict)
        print("Iteration: ", i, " Loss: ", fetch_res['loss'])
        print("Iteration: ", i, " Losses: ", fetch_res['losses'])
