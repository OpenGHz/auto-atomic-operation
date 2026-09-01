# https://www.sciencedirect.com/science/article/pii/S1524070300905289
# https://www.sciencedirect.com/science/article/pii/S0094114X24002519
from collections.abc import Mapping
from typing import Any, List, Optional

import numpy as np

np.set_printoptions(precision=4, suppress=True)


_DEFAULT_A = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.062)
_DEFAULT_ALPHA = (
    0.0,
    np.pi / 2,
    np.pi / 2,
    np.pi / 2,
    np.pi / 2,
    np.pi / 2,
    np.pi / 2,
    np.pi / 2,
)
_DEFAULT_D = (0.0, 0.16452, 0.0, 0.249, 0.0, 0.329, 0.0, 0.0, 0.0)
_DEFAULT_THETA = (
    0.0,
    0.0,
    -13 * np.pi / 18,
    0.0,
    -2 * np.pi / 9,
    -np.pi / 2,
    np.pi / 2,
    np.pi,
    np.pi,
)
_DEFAULT_JOINT_LIMITS = (
    (-170 * np.pi / 180, 170 * np.pi / 180),
    (-150 * np.pi / 180, 50 * np.pi / 180),
    (-170 * np.pi / 180, 170 * np.pi / 180),
    (-140 * np.pi / 180, 10 * np.pi / 180),
    (-170 * np.pi / 180, 170 * np.pi / 180),
    (-45 * np.pi / 180, 45 * np.pi / 180),
    (-90 * np.pi / 180, 70 * np.pi / 180),
)


class KDL_7DOF:
    NEAR_ZERO = 1e-5

    @staticmethod
    def normalize_angle(angle: float, min_limit: float, max_limit: float) -> float:
        """
        Normalize angle to be within joint limits by adding/subtracting 2*pi.
        Returns the angle if it can be normalized, or the original angle if not.
        """
        TWO_PI = 2 * np.pi
        # Try adding 2*pi
        if angle < min_limit:
            normalized = angle + TWO_PI
            if min_limit <= normalized <= max_limit:
                return normalized
        # Try subtracting 2*pi
        if angle > max_limit:
            normalized = angle - TWO_PI
            if min_limit <= normalized <= max_limit:
                return normalized
        return angle

    def __init__(self, kinematics: Optional[Mapping[str, Any]] = None):
        """Create the analytical solver with optional robot-specific parameters.

        ``kinematics`` may override any of the DH arrays (``a``, ``alpha``,
        ``d`` and ``theta``), the seven-joint ``joint_limits`` matrix, or
        ``max_retry``.  The defaults are the original P7 V3 parameters, so a
        caller can reuse this solver for a kinematically compatible arm while
        declaring only the values that differ in its Hydra configuration.
        """
        params = dict(kinematics or {})
        self.a = self._vector_param("a", params.get("a", _DEFAULT_A), 8)
        self.alpha = self._vector_param("alpha", params.get("alpha", _DEFAULT_ALPHA), 8)
        self.d = self._vector_param("d", params.get("d", _DEFAULT_D), 9)
        self.theta = self._vector_param("theta", params.get("theta", _DEFAULT_THETA), 9)

        joint_limits = np.asarray(
            params.get("joint_limits", _DEFAULT_JOINT_LIMITS), dtype=np.float64
        )
        if joint_limits.shape != (7, 2) or not np.all(np.isfinite(joint_limits)):
            raise ValueError("joint_limits must be a finite 7x2 array")
        if np.any(joint_limits[:, 0] >= joint_limits[:, 1]):
            raise ValueError(
                "joint_limits lower bounds must be smaller than upper bounds"
            )
        self.JOINT_LIMITS = joint_limits

        self.max_retry = int(params.get("max_retry", 15))
        if self.max_retry < 1:
            raise ValueError("max_retry must be at least 1")

        self.R_a = np.eye(3)
        self.t_a = np.array([0, 0, self.d[3]])
        self.A = np.eye(4)
        self.A[:3, :3] = self.R_a
        self.A[:3, 3] = self.t_a
        self.t_a_squared = self.t_a @ self.t_a

        self.R_b = np.eye(3)
        self.t_b = np.array([0, -self.d[5], 0])
        self.B = np.eye(4)
        self.B[:3, :3] = self.R_b
        self.B[:3, 3] = self.t_b
        self.t_b_squared = self.t_b @ self.t_b

        r_3_align = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        R1_init = (
            self.dh_rotation(1, 0 - self.theta[1])
            @ self.dh_rotation(2, 0 - self.theta[2])
            @ self.dh_rotation(3, 0 - self.theta[3])
            @ r_3_align
        )
        # R1_init = self.dh_rotation(1, 0) @ self.dh_rotation(2, 0) @ self.dh_rotation(3, 0)
        T1_init = np.eye(4)
        T1_init[:3, :3] = R1_init
        self.elbow_init = T1_init @ self.A

        self.inv_ee = np.linalg.inv(self.dh_transform(8, 0))

        # self.t_a_norm = np.sqrt(self.a[3]**2 + self.d[3]**2) d[3]
        # self.phi4 = np.arctan2(self.a[3], self.d[3])
        # print(f"k4: {self.t_a_norm}, phi4: {self.phi4}")

        # Link inertial parameters (placeholder values)
        # Index 1..7 corresponds to link after joint 1..7
        # Index 0 is unused (base frame), index 8 is the tool/end-effector link
        self.link_masses = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.link_coms = [np.zeros(3) for _ in range(9)]
        self.link_inertias = [np.zeros((3, 3))] + [0.01 * np.eye(3) for _ in range(8)]

        # TCP transform (identity by default = no offset)
        self.T_tcp = np.eye(4)
        self.inv_tcp = np.eye(4)

        # Gravity vector in base frame
        self.gravity = np.array([0.0, 0.0, -9.81])

    @staticmethod
    def _vector_param(name: str, value: Any, size: int) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64)
        if result.shape != (size,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must be a finite vector with {size} values")
        return result.copy()

    def load_config_from_urdf(self, config_path: str):
        """
        Load robot configuration (inertial parameters, TCP transform) from URDF file specified in config.

        Args:
            config_path: Path to JSON config file with urdf_path and link_names
        """
        from urdf_parser import (
            load_robot_config,
            parse_urdf_inertial,
            parse_tcp_transform,
            _rpy_to_rotation_matrix,
        )
        import os

        config = load_robot_config(config_path)
        urdf_path = config["urdf_path"]
        if not os.path.isabs(urdf_path):
            urdf_path = os.path.join(os.path.dirname(config_path), urdf_path)

        eef_config = config.get("eef")
        masses, coms, inertias = parse_urdf_inertial(
            urdf_path, config["link_names"], eef_config
        )

        # Update with index 0 unused (base)
        self.link_masses = [0.0] + masses
        self.link_coms = [np.zeros(3)] + coms
        self.link_inertias = [np.zeros((3, 3))] + inertias

        # Parse TCP transform
        tcp_config = config.get("tcp")
        tool_link = config.get("flange_link", "tool")

        if tcp_config is not None:
            if "xyz" in tcp_config and "xyzw" in tcp_config:
                # Explicit TCP from position + quaternion (x, y, z, w)
                xyz = np.array(tcp_config["xyz"])
                qx, qy, qz, qw = tcp_config["xyzw"]
                # Quaternion to rotation matrix
                R = np.array(
                    [
                        [
                            1 - 2 * (qy * qy + qz * qz),
                            2 * (qx * qy - qz * qw),
                            2 * (qx * qz + qy * qw),
                        ],
                        [
                            2 * (qx * qy + qz * qw),
                            1 - 2 * (qx * qx + qz * qz),
                            2 * (qy * qz - qx * qw),
                        ],
                        [
                            2 * (qx * qz - qy * qw),
                            2 * (qy * qz + qx * qw),
                            1 - 2 * (qx * qx + qy * qy),
                        ],
                    ]
                )
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = xyz
                self.T_tcp = T
                self.inv_tcp = np.linalg.inv(T)
            elif "xyz" in tcp_config and "rpy" in tcp_config:
                # Explicit TCP from position + roll-pitch-yaw
                xyz = np.array(tcp_config["xyz"])
                rpy = tcp_config["rpy"]
                R = _rpy_to_rotation_matrix(rpy[0], rpy[1], rpy[2])
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = xyz
                self.T_tcp = T
                self.inv_tcp = np.linalg.inv(T)
            elif "link" in tcp_config:
                # TCP from URDF link name
                T = parse_tcp_transform(urdf_path, tool_link, tcp_config["link"])
                if T is not None:
                    self.T_tcp = T
                    self.inv_tcp = np.linalg.inv(T)
        else:
            # Auto-detect: look for end_link as child of tool link
            T = parse_tcp_transform(urdf_path, tool_link, "end_link")
            if T is not None:
                self.T_tcp = T
                self.inv_tcp = np.linalg.inv(T)

    def dh_transform(self, joint_index: int, angle: float) -> np.ndarray:
        """
        Compute the Denavit-Hartenberg transformation matrix for a given joint.
        """
        a_i = self.a[joint_index - 1]
        alpha_i = self.alpha[joint_index - 1]
        d_i = self.d[joint_index]
        theta_i = self.theta[joint_index] + angle

        ct = np.cos(theta_i)
        st = np.sin(theta_i)
        ca = np.cos(alpha_i)
        sa = np.sin(alpha_i)

        ret = np.array(
            [
                [ct, -st, 0, a_i],
                [st * ca, ct * ca, -sa, -d_i * sa],
                [st * sa, ct * sa, ca, d_i * ca],
                [0, 0, 0, 1],
            ]
        )
        # print(f"DH Transform for joint {joint_index} with angle {angle}:\n", ret)
        return ret

    def dh_rotation(self, joint_index: int, angle: float) -> np.ndarray:
        """
        Compute the rotation part of the Denavit-Hartenberg transformation matrix for a given joint.
        """
        alpha_i = self.alpha[joint_index - 1]
        theta_i = self.theta[joint_index] + angle

        ct = np.cos(theta_i)
        st = np.sin(theta_i)
        ca = np.cos(alpha_i)
        sa = np.sin(alpha_i)

        R = np.array([[ct, -st, 0], [st * ca, ct * ca, -sa], [st * sa, ct * sa, ca]])
        # print(f"DH Rotation for joint {joint_index} with angle {angle}:\n", R)
        return R

    def fk(
        self, joint_angles: list[7], frame_indices=None, use_tcp: bool = True
    ) -> np.ndarray:
        """
        Forward kinematics for end-effector or multiple frames.

        Args:
            joint_angles: 7 joint angles
            frame_indices: Optional list of frame indices (0-8). If None, returns end-effector pose.
            use_tcp: If True (default), compose T_tcp after frame 8 for end-effector pose.

        Returns:
            If frame_indices is None: 4x4 end-effector transformation matrix
            If frame_indices is list: List of 4x4 transformation matrices for requested frames
        """
        if frame_indices is None:
            T = np.eye(4)
            for i in range(7):
                T = T @ self.dh_transform(i + 1, joint_angles[i])
            T = T @ self.dh_transform(8, 0)
            if use_tcp:
                T = T @ self.T_tcp
            return T

        # Compute all frames up to max requested
        max_idx = max(frame_indices)
        frames = [np.eye(4)]
        for i in range(min(7, max_idx)):
            frames.append(frames[-1] @ self.dh_transform(i + 1, joint_angles[i]))
        if max_idx >= 8:
            for i in range(len(frames), 8):
                frames.append(frames[-1] @ self.dh_transform(i + 1, joint_angles[i]))
            frames.append(frames[-1] @ self.dh_transform(8, 0))

        return [frames[i] for i in frame_indices]

    def _elbow_fk(self, joint_angles: list[7]) -> np.ndarray:
        R1 = (
            self.dh_rotation(1, joint_angles[0])
            @ self.dh_rotation(2, joint_angles[1])
            @ self.dh_rotation(3, joint_angles[2])
        )
        # print(f"R1:\n{R1}")
        T1 = np.eye(4)
        T1[:3, :3] = R1
        T_elbow = T1 @ self.A
        return T_elbow

    def ik(
        self,
        target_pose: np.ndarray,
        reference_angles: list[float] = None,
        reference_elbow_pose: np.ndarray = None,
        use_tcp: bool = True,
    ) -> List[np.ndarray]:
        # Strip TCP from target if needed
        if use_tcp:
            target_pose = target_pose @ self.inv_tcp

        def base_to_shoulder(pose: np.ndarray) -> np.ndarray:
            pose = pose.copy()
            pose[:3, 3] -= np.array([0, 0, self.d[1]])
            return pose

        def R_y(theta):
            return self.dh_rotation(4, theta)

        def T_y(theta):
            T = np.eye(4)
            T[:3, :3] = R_y(theta)
            return T

        if reference_elbow_pose is not None:
            reference_elbow_pose = base_to_shoulder(reference_elbow_pose)

        if reference_angles is not None and reference_elbow_pose is None:
            reference_elbow_pose = self._elbow_fk(reference_angles)

        # print(f"reference_elbow_pose:\n{reference_elbow_pose}")

        P_wrist = target_pose @ self.inv_ee
        G = base_to_shoulder(P_wrist)
        t_g = G[:3, 3]
        t_g_squared = t_g @ t_g
        t_g_norm = np.sqrt(t_g_squared)
        # print(f"t_g: {t_g}, t_g_norm: {t_g_norm}")

        ret = []

        # solving for theta4
        # a3 * s4 + (-d3) * c4 = (tg^T * tg - t_a^T * t_a - t_b^T * t_b) / 2 * d5
        c4 = (self.t_a_squared + self.t_b_squared - t_g_squared) / (
            2 * self.d[5] * self.d[3]
        )
        if abs(c4) > 1:
            print("failed, -1")
            return []
        # print(f"c4: {c4}")
        positive_s4 = np.sqrt(1 - c4**2)
        for sig_s4 in (1, -1):
            s4 = sig_s4 * positive_s4
            theta4 = np.arctan2(s4, c4) - self.theta[4]
            # Normalize theta4 to be within joint limits
            theta4 = self.normalize_angle(
                theta4, self.JOINT_LIMITS[3, 0], self.JOINT_LIMITS[3, 1]
            )
            # print(f"theta4: {theta4}")
            if not (self.JOINT_LIMITS[3, 0] <= theta4 <= self.JOINT_LIMITS[3, 1]):
                print("failed, -2")
                continue

            # characterizing the extra degree of freedom
            axis_n = t_g / t_g_norm
            # print(f"axis_n: {axis_n}")
            if reference_elbow_pose is not None:
                vector_e = reference_elbow_pose[:3, 3]
                axis_a = vector_e / np.linalg.norm(vector_e)
                vector_u = axis_a - (axis_a @ axis_n) * axis_n
            else:
                # use stereographic SEW angle definition, refer to the second paper for details
                e_t = np.array([0, 0, -1])
                e_r = np.array([0, 1, 0])
                k_rt = np.cross(axis_n - e_t, e_r)
                vector_u = np.cross(k_rt, axis_n)

            axis_u = vector_u / np.linalg.norm(vector_u)
            # print(f"axis_u: {axis_u}")
            axis_v = np.cross(axis_n, axis_u)
            # print(f"axis_v: {axis_v}")

            cos_alpha = (self.t_a_squared + t_g_squared - self.t_b_squared) / (
                2 * self.d[3] * t_g_norm
            )
            vector_C = self.d[3] * cos_alpha * axis_n
            radius = self.d[3] * np.sqrt(1 - cos_alpha**2)
            # print(f"vector_C: {vector_C}, radius: {radius}")

            def elbow_position(psi: float) -> np.ndarray:
                return vector_C + radius * (np.cos(psi) * axis_u + np.sin(psi) * axis_v)

            for retry in range(self.max_retry):
                psi = 2 * np.pi / self.max_retry * np.ceil(retry / 2.0)
                if retry % 2 == 1:
                    psi = -psi
                # print(f"retry: {retry}, psi: {psi}")
                t_elbow = elbow_position(psi)
                t_elbow_norm = np.linalg.norm(t_elbow)
                # print(f"P_elbow: {t_elbow}")

                T_e_init = self.elbow_init @ T_y(theta4)
                # print(f"T_e_init:\n{T_e_init}")
                T_w_init = T_e_init @ self.B
                # print(f"T_w_init:\n{T_w_init}")
                e_init = T_e_init[:3, 3]
                w_init = T_w_init[:3, 3]
                # print(f"e_init: {e_init}, w_init: {w_init}")
                x_init = e_init / np.linalg.norm(e_init)
                y_init = (w_init - (w_init @ x_init) * x_init) / np.linalg.norm(
                    w_init - (w_init @ x_init) * x_init
                )
                z_init = np.cross(x_init, y_init)
                # print(f"x_init: {x_init}, y_init: {y_init}, z_init: {z_init}")

                system_init = np.eye(4)
                system_init[:3, :3] = np.column_stack((x_init, y_init, z_init))
                # print(f"system_init:\n{system_init}")

                x_target = t_elbow / t_elbow_norm
                y_target = (t_g - (t_g @ x_target) * x_target) / np.linalg.norm(
                    t_g - (t_g @ x_target) * x_target
                )
                z_target = np.cross(x_target, y_target)
                # print(f"x_target: {x_target}, y_target: {y_target}, z_target: {z_target}")

                system_target = np.eye(4)
                system_target[:3, :3] = np.column_stack((x_target, y_target, z_target))
                # print(f"system_target:\n{system_target}")

                R1 = system_target @ system_init.T
                # print(f"R1:\n{R1}")
                R2 = np.linalg.inv(R1 @ self.A @ T_y(theta4) @ self.B) @ G
                # print(f"R2:\n{R2}")

                # ret = np.array([])
                c2 = -R1[2, 2]
                if abs(c2) > 1 + self.NEAR_ZERO:
                    print("failed, -3")
                    continue
                if abs(c2) > 1:
                    c2 = np.clip(c2, -1, 1)
                positive_s2 = np.sqrt(1 - c2**2)
                # print("c2, s2", c2, positive_s2)

                # when s2 = 0, c2 = 1
                # P =
                # [ c1c3-s1s3, -c1s3-s1c3, 0, (c1c3-s1s3)a3 ] c13, -s13, 0, c13*a3
                # [ s1c3+c1s3, -s1s3+c1c3, 0, (s1c3+c1s3)a3 ] s13, c13, 0, s13*a3
                # [ 0, 0, 1, d3 ]
                # [ 0 0 0 1]

                first_three_joints_solutions = []

                if abs(positive_s2) < self.NEAR_ZERO:
                    # print("near zero 2")
                    theta2 = -self.theta[2]
                    theta13 = np.arctan2(R1[1, 0], R1[0, 0])

                    if reference_angles is not None:
                        ref_theta1 = reference_angles[0] + self.theta[1]
                        ref_theta3 = reference_angles[2] + self.theta[3]
                    else:
                        ref_theta1 = 0 + self.theta[1]
                        ref_theta3 = 0 + self.theta[3]

                    delta_angle = (theta13) - (ref_theta1 + ref_theta3)
                    theta1 = ref_theta1 + delta_angle / 2 - self.theta[1]
                    theta3 = ref_theta3 + delta_angle / 2 - self.theta[3]

                    if np.all(
                        np.array([theta1, theta2, theta3]) <= self.JOINT_LIMITS[:3, 1]
                    ) and np.all(
                        np.array([theta1, theta2, theta3]) >= self.JOINT_LIMITS[:3, 0]
                    ):
                        first_three_joints_solutions.append([theta1, theta2, theta3])

                else:
                    for i in (1, -1):
                        s2 = i * positive_s2
                        # print(f"s2: {s2}")

                        theta2 = np.arctan2(s2, c2) - self.theta[2]
                        theta1 = np.arctan2(R1[1, 2] * i, R1[0, 2] * i) - self.theta[1]
                        theta3 = np.arctan2(-R1[2, 1] * i, R1[2, 0] * i) - self.theta[3]

                        # Normalize angles to be within joint limits
                        theta1 = self.normalize_angle(
                            theta1, self.JOINT_LIMITS[0, 0], self.JOINT_LIMITS[0, 1]
                        )
                        theta2 = self.normalize_angle(
                            theta2, self.JOINT_LIMITS[1, 0], self.JOINT_LIMITS[1, 1]
                        )
                        theta3 = self.normalize_angle(
                            theta3, self.JOINT_LIMITS[2, 0], self.JOINT_LIMITS[2, 1]
                        )

                        if np.all(
                            np.array([theta1, theta2, theta3])
                            <= self.JOINT_LIMITS[:3, 1]
                        ) and np.all(
                            np.array([theta1, theta2, theta3])
                            >= self.JOINT_LIMITS[:3, 0]
                        ):
                            first_three_joints_solutions.append(
                                [theta1, theta2, theta3]
                            )
                        # else :
                        #     print("failed: theta1, theta2, theta3", theta1, theta2, theta3)

                if not first_three_joints_solutions:
                    print("failed, -4")
                    continue
                c6 = R2[1, 2]
                # angle6: [-pi/4, pi/4] + theta6 = [pi/4, 3pi/4], then sin6 is always non-negative
                s6 = np.sqrt(1 - c6**2)
                # print(f"s6: {s6}, c6: {c6}")
                # print(f"theta5: {theta5}")
                # print(f"theta6: {theta6}")
                # print(f"theta7: {theta7}")
                # print()
                theta6 = np.arctan2(s6, c6) - self.theta[6]
                theta5 = np.arctan2(R2[2, 2] / s6, R2[0, 2] / s6) - self.theta[5]
                theta7 = np.arctan2(R2[1, 1] / s6, -R2[1, 0] / s6) - self.theta[7]

                # Normalize angles to be within joint limits
                theta5 = self.normalize_angle(
                    theta5, self.JOINT_LIMITS[4, 0], self.JOINT_LIMITS[4, 1]
                )
                theta6 = self.normalize_angle(
                    theta6, self.JOINT_LIMITS[5, 0], self.JOINT_LIMITS[5, 1]
                )
                theta7 = self.normalize_angle(
                    theta7, self.JOINT_LIMITS[6, 0], self.JOINT_LIMITS[6, 1]
                )

                if np.all(
                    np.array([theta5, theta6, theta7]) <= self.JOINT_LIMITS[4:, 1]
                ) and np.all(
                    np.array([theta5, theta6, theta7]) >= self.JOINT_LIMITS[4:, 0]
                ):
                    for theta1, theta2, theta3 in first_three_joints_solutions:
                        ret.append(
                            np.array(
                                [theta1, theta2, theta3, theta4, theta5, theta6, theta7]
                            )
                        )
                if not ret:
                    print("failed, -5")
                    continue
                break

        # sort by distance to reference angles if provided
        if reference_angles is not None:
            ret.sort(
                key=lambda angles: np.linalg.norm(
                    np.array(angles) - np.array(reference_angles)
                )
            )
        return ret

    def fk2(self, joint_angles: list[7], use_tcp: bool = True) -> np.ndarray:
        """
        Alternative forward kinematics using elbow decomposition.
        """
        R2 = (
            self.dh_rotation(5, joint_angles[4])
            @ self.dh_rotation(6, joint_angles[5])
            @ self.dh_rotation(7, joint_angles[6])
        )
        T2 = np.eye(4)
        T2[:3, :3] = R2
        Ry = self.dh_rotation(4, joint_angles[3])
        Ty = np.eye(4)
        Ty[:3, :3] = Ry
        T_base_to_shoulder = np.eye(4)
        T_base_to_shoulder[:3, 3] = np.array([0, 0, self.d[1]])
        T_wrist_to_ee = self.dh_transform(8, 0)
        elbow_pose = self._elbow_fk(joint_angles) @ Ty
        T = T_base_to_shoulder @ elbow_pose @ self.B @ T2 @ T_wrist_to_ee
        if use_tcp:
            T = T @ self.T_tcp
        return T

    def geometric_jacobian(
        self, joint_angles: list, use_tcp: bool = True
    ) -> np.ndarray:
        """
        Compute the 6x7 geometric Jacobian in the base frame.

        Args:
            joint_angles: 7 joint angles
            use_tcp: If True (default), reference the TCP position instead of tool flange.

        Returns:
            J: np.ndarray of shape (6, 7).
                Rows 0-2: linear velocity Jacobian (J_v)
                Rows 3-5: angular velocity Jacobian (J_w)
        """
        frames = self.fk(joint_angles, frame_indices=list(range(9)))
        if use_tcp:
            o_n = (frames[8] @ self.T_tcp)[:3, 3]
        else:
            o_n = frames[8][:3, 3]

        J = np.zeros((6, 7))
        for i in range(7):
            z_i = frames[i + 1][:3, 2]
            o_i = frames[i + 1][:3, 3]
            J[:3, i] = np.cross(z_i, o_n - o_i)
            J[3:, i] = z_i
        return J

    def inverse_dynamics(
        self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray, gravity: np.ndarray = None
    ) -> np.ndarray:
        """
        Compute inverse dynamics using the Recursive Newton-Euler Algorithm.

        tau = M(q) @ qdd + C(q, qd) @ qd + G(q)

        Args:
            q: Joint positions (7,)
            qd: Joint velocities (7,)
            qdd: Joint accelerations (7,)
            gravity: Optional gravity vector (3,). If None, uses self.gravity

        Returns:
            tau: Joint torques (7,)
        """
        n = 7
        z_hat = np.array([0.0, 0.0, 1.0])

        # Compute individual DH transforms for joints 1..7 and the fixed frame 8
        R_local = [None] * (n + 2)
        p_local = [None] * (n + 2)

        for i in range(1, n + 1):
            T_i = self.dh_transform(i, q[i - 1])
            R_local[i] = T_i[:3, :3]
            p_local[i] = T_i[:3, 3]

        T_8 = self.dh_transform(8, 0)
        R_local[n + 1] = T_8[:3, :3]
        p_local[n + 1] = T_8[:3, 3]

        # Forward recursion: propagate velocities and accelerations
        omega = [np.zeros(3) for _ in range(n + 2)]
        alpha = [np.zeros(3) for _ in range(n + 2)]
        a_e = [np.zeros(3) for _ in range(n + 2)]
        a_c = [np.zeros(3) for _ in range(n + 2)]

        # Base frame initial conditions
        a_e[0] = -(gravity if gravity is not None else self.gravity)

        for i in range(1, n + 1):
            Rt = R_local[i].T  # transforms vectors from frame i-1 to frame i

            omega[i] = Rt @ omega[i - 1] + qd[i - 1] * z_hat
            alpha[i] = (
                Rt @ alpha[i - 1]
                + np.cross(Rt @ omega[i - 1], qd[i - 1] * z_hat)
                + qdd[i - 1] * z_hat
            )
            a_e[i] = Rt @ (
                np.cross(alpha[i - 1], p_local[i])
                + np.cross(omega[i - 1], np.cross(omega[i - 1], p_local[i]))
                + a_e[i - 1]
            )

            r_i = self.link_coms[i]
            a_c[i] = (
                np.cross(alpha[i], r_i)
                + np.cross(omega[i], np.cross(omega[i], r_i))
                + a_e[i]
            )

        # Forward propagation to tool frame (frame 8, fixed joint: qd=0, qdd=0)
        Rt8 = R_local[n + 1].T
        omega[n + 1] = Rt8 @ omega[n]
        alpha[n + 1] = Rt8 @ alpha[n]
        a_e[n + 1] = Rt8 @ (
            np.cross(alpha[n], p_local[n + 1])
            + np.cross(omega[n], np.cross(omega[n], p_local[n + 1]))
            + a_e[n]
        )
        r_tool = self.link_coms[n + 1]
        a_c[n + 1] = (
            np.cross(alpha[n + 1], r_tool)
            + np.cross(omega[n + 1], np.cross(omega[n + 1], r_tool))
            + a_e[n + 1]
        )

        # Backward recursion: compute forces and torques
        # Initialize at the tool link (index n+1 = 8)
        m_tool = self.link_masses[n + 1]
        I_tool = self.link_inertias[n + 1]
        f = [np.zeros(3) for _ in range(n + 2)]
        nn = [np.zeros(3) for _ in range(n + 2)]
        f[n + 1] = m_tool * a_c[n + 1]
        nn[n + 1] = (
            I_tool @ alpha[n + 1]
            + np.cross(omega[n + 1], I_tool @ omega[n + 1])
            + np.cross(r_tool, m_tool * a_c[n + 1])
        )
        tau = np.zeros(n)

        for i in range(n, 0, -1):
            m_i = self.link_masses[i]
            I_i = self.link_inertias[i]
            r_i = self.link_coms[i]

            # R_local[i+1] transforms vectors from frame i+1 to frame i
            R_ip1 = R_local[i + 1]

            f[i] = m_i * a_c[i] + R_ip1 @ f[i + 1]

            nn[i] = (
                I_i @ alpha[i]
                + np.cross(omega[i], I_i @ omega[i])
                + np.cross(r_i, m_i * a_c[i])
                + R_ip1 @ nn[i + 1]
                + np.cross(p_local[i + 1], R_ip1 @ f[i + 1])
            )

            tau[i - 1] = nn[i] @ z_hat

        return tau

    def gravity_vector(self, q: np.ndarray, gravity: np.ndarray = None) -> np.ndarray:
        """
        Compute the gravity torque vector G(q).

        Args:
            q: Joint positions (7,)
            gravity: Optional gravity vector (3,). If None, uses self.gravity

        Returns:
            G: np.ndarray of shape (7,)
        """
        return self.inverse_dynamics(q, np.zeros(7), np.zeros(7), gravity)

    def gravity_compensation(self, q: np.ndarray) -> np.ndarray:
        """
        Compute gravity compensation torques (alias for gravity_vector).

        Returns:
            tau_gravity: np.ndarray of shape (7,)
        """
        return self.gravity_vector(q)

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """
        Compute the 7x7 mass/inertia matrix M(q).

        Returns:
            M: np.ndarray of shape (7, 7), symmetric positive definite.
        """
        n = 7
        qd = np.zeros(n)
        g = self.inverse_dynamics(q, qd, np.zeros(n))

        M = np.zeros((n, n))
        for i in range(n):
            ei = np.zeros(n)
            ei[i] = 1.0
            M[:, i] = self.inverse_dynamics(q, qd, ei) - g

        M = 0.5 * (M + M.T)
        return M

    def coriolis_torques(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        """
        Compute C(q, qd) @ qd directly (efficient, 2 RNEA calls).

        Returns:
            c: np.ndarray of shape (7,)
        """
        return self.inverse_dynamics(q, qd, np.zeros(7)) - self.gravity_vector(q)

    def coriolis_matrix(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        """
        Compute the 7x7 Coriolis/centrifugal matrix C(q, qd)
        using Christoffel symbols with numerical differentiation of M(q).

        Satisfies: M_dot - 2C is skew-symmetric.

        Returns:
            C: np.ndarray of shape (7, 7)
        """
        n = 7
        eps = 1e-8

        # Compute dM/dq_i via central differences
        dMdq = np.zeros((n, n, n))  # dMdq[i] = dM/dq_i
        for i in range(n):
            q_plus = q.copy()
            q_minus = q.copy()
            q_plus[i] += eps
            q_minus[i] -= eps
            dMdq[i] = (self.mass_matrix(q_plus) - self.mass_matrix(q_minus)) / (2 * eps)

        # Christoffel symbols: c_{kji} = 0.5 * (dM_{kj}/dq_i + dM_{ki}/dq_j - dM_{ji}/dq_k)
        C = np.zeros((n, n))
        for k in range(n):
            for j in range(n):
                for i in range(n):
                    c_kji = 0.5 * (dMdq[i][k, j] + dMdq[j][k, i] - dMdq[k][j, i])
                    C[k, j] += c_kji * qd[i]

        return C


def main():
    # Example usage
    robot = KDL_7DOF()
    # robot.load_config_from_urdf('robot_config.json')
    # joint_angles = [0, 0, 0, 0., 0, 0, 0]  # Replace with actual joint angles
    # joint_angles = [-1.571, 0, 0, -0.807, 0, 0, 0]  # Replace with actual joint angles
    # joint_angles = [0, -1.2741, 0, -1.2741, 0, 0, 0]  # Replace with actual joint angles
    # joint_angles = [-np.pi/2, -1.2741, 0, 0, 0, 0, 0]  # Replace with actual joint angles
    joint_angles = [-1.50209, -0.73429, 1.25966, -2.2, 2.95369, -0.36629, 1.19724]
    # joint_angles = [0, -1.2741, 0, 0, 0, 0, 0]  # Replace with actual joint angles
    # joint_angles = [0.5, -0.5, 0.5, -0.5, 0.5, 0.5, 0.5]  # Replace with actual joint angles

    # end_effector_pose = robot.fk(joint_angles)
    # print("End-Effector Pose:\n", end_effector_pose)
    end_effector_pose2 = robot.fk2(joint_angles)
    print("End-Effector Pose from fk2:\n", end_effector_pose2)

    import time

    current_time = time.time()
    ik_solutions = robot.ik(end_effector_pose2, reference_angles=joint_angles)
    # ik_solutions = robot.ik(end_effector_pose2)
    # ik_solutions = robot.ik(end_effector_pose2, reference_angles=[0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    print("time consuming:", time.time() - current_time)
    print("IK Solutions:")
    for i, sol in enumerate(ik_solutions):
        print(f"Solution {i + 1}: {sol}")
        print(f"FK of Solution {i + 1}:\n{robot.fk2(sol)}")


if __name__ == "__main__":
    main()
