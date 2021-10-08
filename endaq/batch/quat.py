import numpy as np


def as_quat_array(q, force_copy=False):
    """Formats a quaternion array."""
    q = (np.array if force_copy else np.asarray)(q)

    if q.ndim == 0 or q.shape[-1] != 4:
        raise ValueError(
            "improper shape for quaternion data; should have wxyz in last axis"
        )

    return q


def quat_mul(q1, q2):
    """Multiply two quaternion arrays."""
    q1, q2 = as_quat_array(q1), as_quat_array(q2)

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

    return result


def quat_conj(q):
    """Conjugate an array of quaternion."""
    q = as_quat_array(q, force_copy=True)
    q[..., 1:] *= -1

    return q


def quat_inv(q):
    """Invert an array of quaternions."""
    return quat_conj(q) / np.sum(q ** 2, axis=-1, keepdims=True)


def quat_div(q1, q2):
    """Divide two quaternion arrays."""
    return quat_mul(q1, quat_inv(q2))


def quat_to_angvel(q, *dt):
    """
    Calculate the angular velocity for an array of orientation quaternions.

    :param q: quaternion array; requires q.shape[-1] == 4
    :param dt: the time corresponing to each quaternion sample
    :return: the angular velocity
    """
    q_prime = np.gradient(q, *dt, axis=range(q.ndim - 1), edge_order=2)
    return 2 * quat_div(q_prime, q)[..., 1:]
