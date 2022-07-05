from typing import List, Tuple

import pandas as pd
import scipy.spatial


def _validate_euler_mode(mode: str) -> Tuple[str, List[str]]:

    _mode = mode.lower()

    names = (
        {'x', 'y', 'z'},
        {'roll', 'pitch', 'yaw'},
        {'alpha', 'beta', 'gamma'},
        {'α', 'β', 'γ'},
        {'psi', 'theta', 'phi'},
        {'ψ', 'θ', 'φ'},
        )

    separators = (
        ' ',
        '-',
        '_',
        )

    for s in separators:
        if s in _mode:
            mode_list = _mode.split(s)
            break
    else:
        if not set(_mode).issubset({'x', 'y', 'z'}):
            raise ValueError(f'Modes other than xyz (such as xyz, xyx, zxz) must '
                             f'separated with one of " ", "-" or "_".  Mode '
                             f'{mode} is not a valid euler angle mode.')
        mode_list = list(_mode)

    if not (
            # must have three rotations
            len(mode_list) == 3
            and (
                    # must have three unique rotations
                    len(set(mode_list)) == 3

                    # must have two unique rotations, and the first and last
                    # rotations must be about the same axes
                    or (len(set(mode_list)) == 2 and mode_list[0] == mode_list[2])
                )
            ):
        raise ValueError(f'Euler modes must have 3 elements '
                         f'and must have either three unique elements (Tait-Bryan angles) '
                         f'or must have two unique elements in the pattern x-y-x '
                         f'(proper Euler angles).  Mode {mode} does not fit this format')

    for n in names:
        if set(mode_list).issubset(n):
            new_mode = ''.join([a for m in mode_list for a, b in zip(names[0], n) if m == b])
            return new_mode, mode_list
    else:
        raise ValueError('Mode uses an unknown naming convention, valid naming '
                         'conventions are: x-y-z, roll-pitch-yaw, '
                         'alpha-beta-gamma, α-β-γ, psi-theta-phi, ψ-θ-φ')


def quaternion_to_euler(df: pd.DataFrame, mode: str = 'x-y-z') -> pd.DataFrame:
    """
    Convert quaternion data in the dataframe ``df`` to euler angles.  This can
    be done with either intrinsic or extrinsic rotations, determined
    automatically based on ``mode``.

    :param df:  The input quaternions to convert.  Must have columns labelled
                'X', 'Y', 'Z', and 'W'.
    :param mode: The order of the axes to rotate.  The default is intrinsic
                 rotation about x-y-z.
    :return:  A dataframe with the euler-angles of the quaternion data.

    .. seealso::
        - `SciPy's documentation on converting into euler angles <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_euler.html>`_
        - `Wikipedia's article on Euler angles <https://en.wikipedia.org/wiki/Euler_angles>`_
    """

    valid_mode, columns = _validate_euler_mode(mode)

    r = scipy.spatial.transform.Rotation.from_quat(df[['X', 'Y', 'Z', 'W']])

    return pd.DataFrame(r.as_euler(valid_mode), index=df.index, columns=columns)
