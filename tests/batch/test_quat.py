import pytest
import numpy as np
import sympy as sp
import quaternion

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


def test_quat_to_angvel():
    # Note - the math for calculating angular velocity requires calculating a
    # derivative of each quaternion component; `quat_to_angvel` uses
    # `np.gradient(edge_order=2)`, while `quaternion.angular_velocity` uses
    # differentiated cubic splines. These two methods are identical when
    # calculating over three quaternion data points.`
    q = np.array(
        [
            [0.29998779, 0.33111572, 0.21844482],
            [-0.58532715, -0.6151123, -0.59637451],
            [-0.66094971, -0.62219238, -0.64056396],
            [-0.36132812, -0.35339355, -0.43157959],
        ]
    ).T

    calc_result = quat.quat_to_angvel(q, 2)
    expt_result = quaternion.angular_velocity(
        quaternion.as_quat_array(q), 2 * np.arange(0, q.shape[0])
    )

    np.testing.assert_allclose(calc_result, expt_result)
