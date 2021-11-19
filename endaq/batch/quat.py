import numpy as np


def as_quat_array(q, force_copy=False, qaxis=-1):
    """Formats a quaternion array."""
    q = (np.array if force_copy else np.asarray)(q)

    if not (-q.ndim <= qaxis < q.ndim):
        raise ValueError("invalid quaternion axis")

    if q.ndim == 0 or q.shape[qaxis] != 4:
        raise ValueError(
            "improper shape for quaternion data;"
            " specified axis should contain WXYZ coordinates"
        )

    return q


def quat_mul(q1, q2, qaxis=-1):
    """Multiply two quaternion arrays."""
    q1, q2 = as_quat_array(q1, qaxis=qaxis), as_quat_array(q2, qaxis=qaxis)
    q1, q2 = np.moveaxis(q1, qaxis, -1), np.moveaxis(q2, qaxis, -1)

    result = np.empty(np.broadcast(q1, q2).shape, dtype=np.result_type(q1, q2))

    result[..., 0] = np.sum(q1 * q2 * np.array([1, -1, -1, -1]), axis=-1)
    result[..., 1] = np.sum(
        q1 * q2[..., [1, 0, 3, 2]] * np.array([1, 1, 1, -1]), axis=-1
    )
    result[..., 2] = np.sum(
        q1 * q2[..., [2, 3, 0, 1]] * np.array([1, -1, 1, 1]), axis=-1
    )
    result[..., 3] = np.sum(
        q1 * q2[..., [3, 2, 1, 0]] * np.array([1, 1, -1, 1]), axis=-1
    )

    return np.moveaxis(result, -1, qaxis)


def quat_conj(q, qaxis=-1):
    """Conjugate an array of quaternion."""
    q = as_quat_array(q, force_copy=True, qaxis=qaxis)

    q = np.moveaxis(q, qaxis, 0)
    q[1:] *= -1
    return np.moveaxis(q, 0, qaxis)


def quat_inv(q, qaxis=-1):
    """Invert an array of quaternions."""
    return quat_conj(q, qaxis=qaxis) / np.sum(q ** 2, axis=qaxis, keepdims=True)


def quat_div(q1, q2, qaxis=-1):
    """Divide two quaternion arrays."""
    return quat_mul(q1, quat_inv(q2, qaxis=qaxis), qaxis=qaxis)


def quat_to_angvel(q, *dt, qaxis=-1):
    """
    Calculate the angular velocity for an array of orientation quaternions.

    :param q: quaternion array; requires q.shape[-1] == 4
    :param dt: the time corresponing to each quaternion sample
    :return: the angular velocity
    """
    q = np.moveaxis(q, qaxis, -1)
    q_prime = np.gradient(q, *dt, axis=range(q.ndim - 1), edge_order=2)
    ang_vel = 2 * quat_div(q_prime, q)[..., 1:]

    return np.moveaxis(ang_vel, -1, qaxis)
