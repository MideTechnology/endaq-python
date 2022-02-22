import pytest

import numpy as np
import pandas as pd

from endaq.calc import rotation


@pytest.mark.parametrize(
        'quat, euler',
        [
            ((0., 0., 0., 1.),  (0.,    0.,    0.)),
            ((0., 0., 0., -1.), (0.,    0.,    0.)),
            ((1., 0., 0., 0.),  (np.pi, 0.,    0.)),
            ((0., 1., 0., 0.),  (np.pi, 0.,    np.pi)),
            ((0., 0., 1., 0.),  (0.,    0.,    np.pi)),
            ]
        )
def test_quat_to_euler_data(quat, euler):
    df = pd.DataFrame([quat], index=[0], columns=['X', 'Y', 'Z', 'W'])
    target = pd.DataFrame([euler], index=[0], columns=['x', 'y', 'z'])

    pd.testing.assert_frame_equal(rotation.quaternion_to_euler(df), target)


@pytest.mark.parametrize(
        'mode, columns, raises',
        [
            ('xyz', ['x', 'y', 'z'], None),
            ('roll-pitch-yaw', ['roll', 'pitch', 'yaw'], None),
            ('snap-crackle-pop', ['snap', 'crackle', 'pop'], pytest.raises(ValueError)),
            ]
        )
def test_quat_to_euler_modes(mode, columns, raises):
    df = pd.DataFrame([(0., 0., 0., 1.)], index=[0], columns=['X', 'Y', 'Z', 'W'])
    target = pd.DataFrame([(0.,    0.,    0.)], index=[0], columns=columns)

    if raises is None:
        pd.testing.assert_frame_equal(rotation.quaternion_to_euler(df, mode=mode), target)
    else:
        with raises:
            pd.testing.assert_frame_equal(rotation.quaternion_to_euler(df, mode=mode), target)