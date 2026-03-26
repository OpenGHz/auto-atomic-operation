"""
为了保证基于欧拉角控制的旋转平稳性，首先，需要确保初始欧拉角是通过特定的四元数转换得到的，否则由于欧拉角的多解性，会导致转换到四元数再转换回欧拉角时出现不一致的情况。其次，在进行旋转插值（如SLERP）时，直接对四元数进行插值，而不是对欧拉角进行插值，以避免欧拉角的奇异性和不连续性问题。最后，验证每一步的结果，确保四元数和欧拉角之间的一致性，以保证旋转的平稳性和正确性。
"""

from auto_atom.utils.transformations import (
    quaternion_from_euler,
    euler_from_quaternion,
    quaternion_multiply,
    quaternion_slerp,
)
from pprint import pprint
import numpy as np


np.set_printoptions(precision=6, suppress=True)


def quat_equivalent(q1, q2, tol=1e-6):
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    q1 /= np.linalg.norm(q1)
    q2 /= np.linalg.norm(q2)
    dot = np.dot(q1, q2)
    return abs(dot) > 1 - tol


init_euler = [-1.5709, -0.0000, 1.5708]
init_euler_quat = quaternion_from_euler(*init_euler)
base_quat = [0.500024, 0.500024, -0.499976, -0.499976]
print("init_euler:", init_euler)
print("init_euler_quat:", init_euler_quat)
print("base_quat:", base_quat)
assert quat_equivalent(init_euler_quat, base_quat), "The quaternions should be equal."
base_euler = euler_from_quaternion(base_quat)
assert np.allclose(base_euler, init_euler), (
    "Bad initial Euler angles: {} != {}.".format(base_euler, init_euler)
)
rela_euler = [0.0, 0.0, 1.5707963267948966]
rela_quat = quaternion_from_euler(*rela_euler)
target_quat = quaternion_multiply(rela_quat, base_quat)
slerp_quats = [quaternion_slerp(base_quat, target_quat, 0.1 * i) for i in range(0, 11)]
slerp_eulers = [np.array(euler_from_quaternion(q)) for q in slerp_quats]

pprint(
    {
        "base_quat": base_quat,
        "base_euler": base_euler,
        "rela_euler": rela_euler,
        "rela_quat": rela_quat,
        "target_quat": target_quat,
        # "slerp_quats": slerp_quats,
        "slerp_eulers": slerp_eulers,
    },
    sort_dicts=False,
)
