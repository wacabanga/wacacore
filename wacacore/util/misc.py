"""Miscellaneious Utils"""
import itertools
import string
import random
from functools import reduce
from typing import Dict, Sequence, Any, TypeVar
T = TypeVar('T')


def rand_string(n):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(n))


def extract(keys: Sequence, dict: Dict):
    """Restrict dict to keys in `keys`"""
    return {key: dict[key] for key in keys}


def pull(dict: Dict, *keys):
    """Restrict dict to keys in `keys`"""
    return {key: dict[key] for key in keys}


def getn(dict: Dict, *keys):
    return (dict[k] for k in keys)


def inn(seq, *keys):
    return all((k in seq for k in keys))


def print_one_per_line(xs:Sequence):
    """Simple printing of one element of xs per line"""
    for x in xs:
        print(x)


def same(xs) -> bool:
    """All elements in xs are the same?"""
    if len(xs) == 0:
        return True
    else:
        x1 = xs[0]
        for xn in xs:
            if xn != x1:
                return False

    return True


def product(xs):
    return reduce((lambda x, y: x * y), xs)


def dict_prod(d: Dict[Any, Sequence]):
    """
    Cartesian product for dictionaries
    Args:
        dict: A dictionary where all keys are iterable
    Returns:
        iterable over dict
    """
    keys = d.keys()
    it = itertools.product(*d.values())
    return (dict(zip(keys, tup)) for tup in it)


def flat_dict(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat_dict(d), prefix="%s_" % k)
        elif isinstance(v, list):
            for i, v2 in enumerate(v):
                out['%s%s_%s' % (prefix, k, i)] = v2
        else:
            out['%s%s' % (prefix, k)] = v
    return out

def stringy_dict(d):
    out = ""
    for (key, val) in d.items():
        if val is not None and val is not '':
            out = out + "%s_%s__" % (str(key), str(val))
    return out


def pos_in_seq(x: T, ys: Sequence[T]) -> int:
    """Return the index of a value in a list"""
    for i, y in enumerate(ys):
        if x == y:
            return i
    assert False, "element not in list"

def complement(indices: Sequence, shape: Sequence) -> Sequence:
    bools = np.zeros(shape)
    output = []
    for index in indices:
        bools[tuple(index)] = 1
    for index, value in np.ndenumerate(bools):
        if value == 0:
            output.append(index)
    return np.squeeze(np.array(output))


def complement_bool(indices: np.ndarray, shape: Sequence) -> Sequence:
    if len(indices.shape) > 2:
        indices = indices.reshape(-1, indices.shape[-1])
    bools = np.zeros(shape)
    for index in indices:
        bools[tuple(index)] = 1
    return bools
