import pytest
import numpy as np
import sympy as sp

from endaq.batch import quat


@pytest.fixture()
def p():
    return np.array(
        [
            sp.symbols("w_p:6", real=True),
            sp.symbols("x_p:6", real=True),
            sp.symbols("y_p:6", real=True),
            sp.symbols("z_p:6", real=True),
        ]
    ).T.reshape(2, 3, 4)


@pytest.fixture()
def q():
    return np.array(
        [
            sp.symbols("w_q:6", real=True),
            sp.symbols("x_q:6", real=True),
            sp.symbols("y_q:6", real=True),
            sp.symbols("z_q:6", real=True),
        ]
    ).T.reshape(2, 3, 4)


def test_quat_conj(q):
    calc_result = quat.quat_conj(q)

    for i_nd in np.ndindex(q.shape[:-1]):
        assert sp.Quaternion(*calc_result[i_nd]) == sp.conjugate(
            sp.Quaternion(*q[i_nd])
        )


def test_quat_inv(q):
    calc_result = quat.quat_inv(q)

    for i_nd in np.ndindex(q.shape[:-1]):
        assert sp.Quaternion(*calc_result[i_nd]) == sp.Quaternion(*q[i_nd]) ** -1


def test_quat_mul(p, q):
    calc_result = quat.quat_mul(p, q)

    for i_nd in np.ndindex(q.shape[:-1]):
        assert sp.Quaternion(*calc_result[i_nd]) == sp.Quaternion(
            *p[i_nd]
        ) * sp.Quaternion(*q[i_nd])


def test_quat_div(p, q):
    calc_result = quat.quat_div(p, q)

    for i_nd in np.ndindex(q.shape[:-1]):
        assert (
            sp.Quaternion(*calc_result[i_nd])
            # faster to check p*(q**-1) than p/q due to symbolic representations
            == sp.Quaternion(*p[i_nd]) * sp.Quaternion(*q[i_nd]) ** -1
        )
