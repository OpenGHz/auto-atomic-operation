import os

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = ""

import numpy as np
import mujoco
try:
    import cv2
except:
    cv2 = None

class TactileSensorManager:
    TANGENTIAL_FORCE_LIMIT_N = 10.0
    NORMAL_FORCE_LIMIT_N = 25.0
    MIN_DETECTABLE_FORCE_N = 0.01
    SAFE_OVERLOAD_RATIO = 2.0
    TACTILE_DIM = 6

    def __init__(
        self, mj_model, mj_data, enable=True, window_name="tactile_force_arrows"
    ):
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.enable = bool(enable)
        self.window_name = window_name

        self.ready = False
        self.scalar_auto_max = 1e-12
        self.arrow_scale_px = 85.0
        self.external_tactile_tensor = None
        self.external_wrench_tensor = None

        self.force_sensor_ids = {}
        self.torque_sensor_ids = {}
        self.site_ids = {}
        self.site_coords_2d = {}

        self.panel_meta = {}
        self.panel_order = []
        self.row_order = []
        self.row_to_panels = {}

        self._collect_sensor_ids()
        self._collect_site_and_geom_ids()
        self._build_layout()
        self._compute_site_2d_coords()

        counts = [len(v) for v in self.force_sensor_ids.values()]
        self.n_points = max(counts) if counts else 0
        self.n_panels = len(self.panel_order)

        total_force = sum(len(v) for v in self.force_sensor_ids.values())
        total_torque = sum(len(v) for v in self.torque_sensor_ids.values())

        if not self.enable:
            print(
                f"[INFO] Tactile sensors initialized (visualization disabled), "
                f"force sensors={total_force}, torque sensors={total_torque}, "
                f"rows={len(self.row_order)}, panels={self.n_panels}, n_points_per_panel={self.n_points}"
            )
            return

        self.ready = True
        print(
            f"[INFO] Tactile visualizer ready, force sensors={total_force}, torque sensors={total_torque}, "
            f"rows={len(self.row_order)}, panels={self.n_panels}, n_points_per_panel={self.n_points}"
        )
        for prefix in self.panel_order:
            print(
                f"[INFO] panel={prefix.rstrip('_')}: "
                f"force={len(self.force_sensor_ids.get(prefix, []))}, "
                f"torque={len(self.torque_sensor_ids.get(prefix, []))}"
            )

    def _compose_tactile_tensor(self, force_vectors, torque_vectors):
        # Determine N_POINTS dynamically from force_vectors
        n_points = max(len(v) for v in force_vectors.values()) if force_vectors else 0
        if n_points == 0:
            n_points = 1

        tactile = np.zeros(
            (self.n_panels, n_points, self.TACTILE_DIM), dtype=np.float32
        )
        for finger_idx, prefix in enumerate(self.panel_order):
            fv = np.asarray(
                force_vectors.get(prefix, np.zeros((n_points, 3))), dtype=np.float32
            )
            tv = np.asarray(
                torque_vectors.get(prefix, np.zeros((n_points, 3))), dtype=np.float32
            )

            if fv.ndim == 2 and fv.shape[1] >= 3:
                count = min(n_points, fv.shape[0])
                tactile[finger_idx, :count, :3] = fv[:count, :3]
            if tv.ndim == 2 and tv.shape[1] >= 3:
                count = min(n_points, tv.shape[0])
                tactile[finger_idx, :count, 3:6] = tv[:count, :3]

        return tactile

    def get_tactile_tensor(self):
        """Return tactile tensor with shape (n_panels, n_points, 6).

        n_points is auto-detected from the model (self.n_points).
        """
        force_vectors = self.get_sensor_force_vectors()
        torque_vectors = self.get_sensor_torque_vectors()
        return self._compose_tactile_tensor(force_vectors, torque_vectors)

    def get_wrench_tensor(self):
        """Return wrench tensor with shape (n_panels, 6)."""
        wrenches = self.get_finger_wrenches()
        wrench_tensor = np.zeros((self.n_panels, 6), dtype=np.float32)
        for finger_idx, prefix in enumerate(self.panel_order):
            wrench = np.asarray(
                wrenches.get(prefix, np.zeros(6)), dtype=np.float32
            ).reshape(-1)
            if wrench.shape[0] >= 6:
                wrench_tensor[finger_idx] = wrench[:6]
        return wrench_tensor

    def _tactile_tensor_to_vectors(self, tactile):
        tactile = np.asarray(tactile, dtype=np.float64)
        if tactile.ndim == 4:
            tactile = tactile[0]
        if (
            tactile.ndim != 3
            or tactile.shape[0] != self.n_panels
            or tactile.shape[2] != self.TACTILE_DIM
        ):
            raise ValueError(
                f"Expected tactile shape ({self.n_panels}, N_POINTS, {self.TACTILE_DIM}), got {tactile.shape}"
            )

        n_points = tactile.shape[1]
        tactile_vectors = {}
        for finger_idx, prefix in enumerate(self.panel_order):
            fv = np.zeros((n_points, 3), dtype=np.float64)
            tv = np.zeros((n_points, 3), dtype=np.float64)
            finger = tactile[finger_idx]
            fv[:, :] = finger[:, :3]
            tv[:, :] = finger[:, 3:6]
            tactile_vectors[prefix] = {
                "force_vectors": fv,
                "torque_vectors": tv,
            }
        return tactile_vectors

    def set_external_tactile_data(self, tactile, wrench):
        """Set external tactile+wrench data for playback visualization.

        tactile: (n_panels, n_points, 6) or (1, n_panels, n_points, 6)
        wrench:  (n_panels, 6) or (1, n_panels, 6)
        """
        if tactile is None or wrench is None:
            self.clear_external_tactile_data()
            return

        tactile_np = np.asarray(tactile, dtype=np.float32)
        wrench_np = np.asarray(wrench, dtype=np.float32)

        if tactile_np.ndim == 4:
            tactile_np = tactile_np[0]
        if wrench_np.ndim == 3:
            wrench_np = wrench_np[0]

        if (
            tactile_np.ndim != 3
            or tactile_np.shape[0] != self.n_panels
            or tactile_np.shape[2] != self.TACTILE_DIM
        ):
            raise ValueError(
                f"Invalid external tactile shape: {tactile_np.shape}, expected ({self.n_panels}, n_points, {self.TACTILE_DIM})"
            )
        if wrench_np.shape != (self.n_panels, 6):
            raise ValueError(f"Invalid external wrench shape: {wrench_np.shape}")

        self.external_tactile_tensor = tactile_np
        self.external_wrench_tensor = wrench_np

    def clear_external_tactile_data(self):
        self.external_tactile_tensor = None
        self.external_wrench_tensor = None

    def _collect_sensor_ids(self):
        for sensor_id in range(self.mj_model.nsensor):
            sensor_name = mujoco.mj_id2name(
                self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id
            )
            sensor_type = int(self.mj_model.sensor_type[sensor_id])

            site_name = self._sensor_site_name(sensor_id)
            inferred_prefix = self._extract_prefix_from_site_name(site_name)

            if inferred_prefix is None:
                continue

            if inferred_prefix not in self.force_sensor_ids:
                self.force_sensor_ids[inferred_prefix] = []
                self.torque_sensor_ids[inferred_prefix] = []
                self.site_ids[inferred_prefix] = []
                self.site_coords_2d[inferred_prefix] = np.zeros(
                    (0, 2), dtype=np.float64
                )
                self.panel_meta[inferred_prefix] = self._parse_panel_meta(
                    inferred_prefix
                )

            if sensor_type == int(mujoco.mjtSensor.mjSENS_FORCE):
                self.force_sensor_ids[inferred_prefix].append(sensor_id)
            elif sensor_type == int(mujoco.mjtSensor.mjSENS_TORQUE):
                self.torque_sensor_ids[inferred_prefix].append(sensor_id)

        for prefix in self.force_sensor_ids.keys():
            self.force_sensor_ids[prefix].sort(key=lambda sid: self._sensor_index(sid))
            self.torque_sensor_ids[prefix].sort(key=lambda sid: self._sensor_index(sid))

    def _collect_site_and_geom_ids(self):
        for site_id in range(self.mj_model.nsite):
            site_name = mujoco.mj_id2name(
                self.mj_model, mujoco.mjtObj.mjOBJ_SITE, site_id
            )
            if not site_name:
                continue
            prefix = self._extract_prefix_from_site_name(site_name)
            if prefix is None:
                continue
            if prefix not in self.site_ids:
                self.site_ids[prefix] = []
                self.force_sensor_ids.setdefault(prefix, [])
                self.torque_sensor_ids.setdefault(prefix, [])
                self.site_coords_2d[prefix] = np.zeros((0, 2), dtype=np.float64)
                self.panel_meta[prefix] = self._parse_panel_meta(prefix)
            self.site_ids[prefix].append(site_id)

        for prefix in self.site_ids.keys():
            self.site_ids[prefix].sort(
                key=lambda sid: self._sensor_index_from_name(
                    mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, sid)
                )
            )

    def _extract_prefix_from_site_name(self, site_name):
        if not site_name or "touch_point" not in site_name:
            return None
        idx = site_name.find("touch_point")
        if idx <= 0:
            return None
        return site_name[:idx]

    def _parse_panel_meta(self, prefix):
        tokens = prefix.strip("_").split("_")
        if len(tokens) >= 2 and tokens[-1] in ("left", "right"):
            arm = "_".join(tokens[:-1])
            finger = tokens[-1]
        else:
            arm = prefix.strip("_")
            finger = "left"
        return {"arm": arm, "finger": finger}

    def _panel_sort_key(self, prefix):
        meta = self.panel_meta[prefix]
        finger_order = 0 if meta["finger"] == "left" else 1
        return (meta["arm"], finger_order, prefix)

    def _build_layout(self):
        self.panel_order = sorted(
            self.force_sensor_ids.keys(), key=self._panel_sort_key
        )
        rows = {}
        for prefix in self.panel_order:
            meta = self.panel_meta[prefix]
            arm = meta["arm"]
            finger = meta["finger"]
            if arm not in rows:
                rows[arm] = {"left": None, "right": None}
            if finger in ("left", "right"):
                rows[arm][finger] = prefix

        self.row_order = sorted(rows.keys())
        self.row_to_panels = rows

    def _compute_site_2d_coords(self):
        """从 mj_data.site_xpos 读取 3D 坐标，用 PCA 投影到方差最大的 2D 平面。"""
        mujoco.mj_forward(self.mj_model, self.mj_data)
        for prefix in self.panel_order:
            ids = self.site_ids.get(prefix, [])
            if len(ids) == 0:
                self.site_coords_2d[prefix] = np.zeros((0, 2), dtype=np.float64)
                continue

            pts = np.array(
                [self.mj_data.site_xpos[sid] for sid in ids], dtype=np.float64
            )  # (N, 3)
            centered = pts - pts.mean(axis=0)

            # PCA: SVD 取方差最大的两个主方向
            if len(pts) >= 2:
                _, _, Vt = np.linalg.svd(centered, full_matrices=False)
                coords_2d = centered @ Vt[:2].T  # (N, 2)
            else:
                coords_2d = centered[:, :2]

            self.site_coords_2d[prefix] = coords_2d  # 单位：米

    def _sensor_site_name(self, sensor_id):
        objtype = int(self.mj_model.sensor_objtype[sensor_id])
        if objtype != int(mujoco.mjtObj.mjOBJ_SITE):
            return None
        objid = int(self.mj_model.sensor_objid[sensor_id])
        return mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, objid)

    def _sensor_index(self, sensor_id):
        name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)
        return self._sensor_index_from_name(name)

    def _sensor_index_from_name(self, name):
        if not name:
            return 0
        try:
            return int(name.split("_")[-1])
        except (ValueError, IndexError):
            return 0

    def _sensor_vector(self, sensor_id):
        adr = int(self.mj_model.sensor_adr[sensor_id])
        dim = int(self.mj_model.sensor_dim[sensor_id])
        if dim <= 0:
            return np.zeros(3, dtype=np.float64)
        data = -self.mj_data.sensordata[adr : adr + dim]
        vec = np.zeros(3, dtype=np.float64)
        copy_dim = min(3, dim)
        vec[:copy_dim] = data[:copy_dim]
        return vec

    def _apply_force_spec(self, vec):
        out = np.array(vec, dtype=np.float64, copy=True)
        out[0] = np.clip(
            out[0], -self.TANGENTIAL_FORCE_LIMIT_N, self.TANGENTIAL_FORCE_LIMIT_N
        )
        out[1] = np.clip(
            out[1], -self.TANGENTIAL_FORCE_LIMIT_N, self.TANGENTIAL_FORCE_LIMIT_N
        )
        out[2] = np.clip(out[2], -self.NORMAL_FORCE_LIMIT_N, self.NORMAL_FORCE_LIMIT_N)
        if np.linalg.norm(out) < self.MIN_DETECTABLE_FORCE_N:
            out[:] = 0.0
        return out

    def _apply_torque_spec(self, vec):
        out = np.array(vec, dtype=np.float64, copy=True)
        force_equivalent_limit = self.NORMAL_FORCE_LIMIT_N * self.SAFE_OVERLOAD_RATIO
        out = np.clip(out, -force_equivalent_limit, force_equivalent_limit)
        return out

    def get_sensor_force_vectors(self):
        """Return per-finger per-sensor 3D force vectors from force sensors.

        Returns:
            Dict[str, np.ndarray] each shape (n_sensors, 3)
        """
        out = {}
        for prefix in self.panel_order:
            ids = self.force_sensor_ids.get(prefix, [])
            n_sensors = len(ids)
            vectors = np.zeros((n_sensors, 3), dtype=np.float64)
            for i in range(n_sensors):
                vectors[i] = self._apply_force_spec(self._sensor_vector(ids[i]))
            out[prefix] = vectors
        return out

    def get_sensor_torque_vectors(self):
        """Return per-finger per-sensor 3D torque vectors from torque sensors.

        Returns:
            Dict[str, np.ndarray] each shape (n_sensors, 3)
        """
        out = {}
        for prefix in self.panel_order:
            ids = self.torque_sensor_ids.get(prefix, [])
            n_sensors = len(ids)
            vectors = np.zeros((n_sensors, 3), dtype=np.float64)
            for i in range(n_sensors):
                vectors[i] = self._apply_torque_spec(self._sensor_vector(ids[i]))
            out[prefix] = vectors
        return out

    def get_sensor_scalars(self):
        """Return per-finger tactile scalar values.

        Returns:
            Dict[str, np.ndarray] where each value shape is (52,)
        """
        out = {}
        force_vectors = self.get_sensor_force_vectors()
        for prefix in self.panel_order:
            out[prefix] = np.linalg.norm(force_vectors[prefix], axis=1)
        return out

    def get_finger_wrenches(self):
        """Return 6D wrench [Fx,Fy,Fz,Mx,My,Mz] per finger from force/torque sensors."""
        out = {}
        force_vectors = self.get_sensor_force_vectors()
        torque_vectors = self.get_sensor_torque_vectors()
        for prefix in self.panel_order:
            wrench_world = np.zeros(6, dtype=np.float64)
            wrench_world[:3] = np.sum(force_vectors[prefix], axis=0)
            wrench_world[3:6] = np.sum(torque_vectors[prefix], axis=0)
            out[prefix] = wrench_world
        return out

    def get_data(self):
        """Unified tactile data API for external use."""
        tactile = self.get_tactile_tensor()
        wrench = self.get_wrench_tensor()
        return {
            "tactile": tactile,
            "wrench": wrench,
        }

    def _project_coords_to_panel(self, coords_2d, x0, y0, w, h, margin=28):
        """将 2D PCA 坐标数组 (N,2) 归一化映射到面板像素坐标，返回 (N,2) int 数组。"""
        if len(coords_2d) == 0:
            return np.zeros((0, 2), dtype=np.int32)
        span = coords_2d.max(axis=0) - coords_2d.min(axis=0)
        span = np.where(span < 1e-9, 1.0, span)
        norm = (coords_2d - coords_2d.min(axis=0)) / span  # (N,2) in [0,1]
        px = (x0 + margin + norm[:, 0] * (w - 2 * margin)).astype(np.int32)
        py = (y0 + h - margin - norm[:, 1] * (h - 2 * margin)).astype(np.int32)
        return np.stack([px, py], axis=1)

    def _compute_force_center_and_resultant(self, scalars, vectors, coords_2d=None):
        """Compute center of pressure (weighted by force) and resultant XY force."""
        weights = np.asarray(scalars, dtype=np.float64).reshape(-1)
        if weights.shape[0] == 0:
            return None, np.zeros(2, dtype=np.float64)

        total_weight = float(np.sum(weights))
        if total_weight <= 1e-12:
            return None, np.zeros(2, dtype=np.float64)

        if coords_2d is not None and len(coords_2d) == len(weights):
            center_2d = np.sum(coords_2d * weights[:, None], axis=0) / total_weight
        else:
            n = len(weights)
            indices = np.arange(n, dtype=np.float64)
            center_2d = np.array([np.sum(indices * weights) / total_weight, 0.0])

        vectors_np = np.asarray(vectors, dtype=np.float64)
        if vectors_np.ndim == 2 and vectors_np.shape[1] >= 2:
            resultant_xy = np.sum(vectors_np[:, :2], axis=0)
        else:
            resultant_xy = np.zeros(2, dtype=np.float64)

        return center_2d, resultant_xy

    def get_slip_status(self):
        return {prefix: False for prefix in self.panel_order}

    def _draw_panel(
        self, canvas, x0, y0, w, h, title, scalars, vectors, coords_2d=None
    ):
        cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (60, 60, 60), 1)
        cv2.putText(
            canvas,
            title,
            (x0 + 10, y0 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 220, 220),
            1,
            lineType=cv2.LINE_AA,
        )

        scale = max(self.scalar_auto_max, 1e-12)
        n_sensors = len(scalars) if scalars is not None else 0

        if n_sensors == 0:
            cv2.putText(
                canvas,
                "No sensors",
                (x0 + 20, y0 + 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (100, 100, 100),
                1,
            )
            return

        # 使用真实 2D 坐标映射，若无则回退到网格布局
        if coords_2d is not None and len(coords_2d) == n_sensors:
            pixel_coords = self._project_coords_to_panel(coords_2d, x0, y0, w, h)
        else:
            cols = max(1, int(np.ceil(np.sqrt(n_sensors))))
            cell = max(1, (w - 56) // cols)
            xs = x0 + 28 + (np.arange(n_sensors) % cols) * cell
            ys = y0 + 40 + (np.arange(n_sensors) // cols) * cell
            pixel_coords = np.stack([xs, ys], axis=1).astype(np.int32)

        for idx in range(n_sensors):
            px, py = int(pixel_coords[idx, 0]), int(pixel_coords[idx, 1])
            val = float(scalars[idx])
            intensity = np.clip(val / scale, 0.0, 1.0)
            color = (
                int(30 + 225 * intensity),
                int(60 + 120 * (1.0 - intensity)),
                int(255 - 150 * intensity),
            )
            cv2.circle(canvas, (px, py), 4, color, -1)

            if vectors is not None and idx < len(vectors):
                vxy = vectors[idx, :2]
                arrow_len = (
                    float(np.linalg.norm(vxy))
                    / max(scale, 1e-12)
                    * self.arrow_scale_px
                    * 0.3
                )
                if arrow_len > 0.5:
                    nxy = vxy / (np.linalg.norm(vxy) + 1e-12)
                    end = (int(px + nxy[0] * arrow_len), int(py - nxy[1] * arrow_len))
                    cv2.arrowedLine(
                        canvas, (px, py), end, (0, 0, 255), 1, tipLength=0.4
                    )

        # 绘制合力中心
        center_2d, resultant_xy = self._compute_force_center_and_resultant(
            scalars, vectors, coords_2d
        )
        if center_2d is not None:
            if coords_2d is not None and len(coords_2d) == n_sensors:
                # center_2d 是真实 2D 坐标，需要用相同的归一化再投影
                dummy = np.vstack([coords_2d, center_2d.reshape(1, 2)])
                pix_all = self._project_coords_to_panel(dummy, x0, y0, w, h)
                cx, cy = int(pix_all[-1, 0]), int(pix_all[-1, 1])
            else:
                cols = max(1, int(np.ceil(np.sqrt(n_sensors))))
                cell = max(1, (w - 56) // cols)
                cx = int(x0 + 28 + (center_2d[0] % cols) * cell)
                cy = int(y0 + 40 + (center_2d[0] // cols) * cell)
            cv2.circle(canvas, (cx, cy), 7, (0, 220, 255), 2)
            cv2.circle(canvas, (cx, cy), 2, (0, 220, 255), -1)
            resultant_norm = float(np.linalg.norm(resultant_xy))
            if resultant_norm > self.MIN_DETECTABLE_FORCE_N:
                arrow_len = float(
                    np.clip(
                        resultant_norm / max(scale, 1e-12) * self.arrow_scale_px * 0.3,
                        8.0,
                        120.0,
                    )
                )
                nxy = resultant_xy / (resultant_norm + 1e-12)
                end = (int(cx + nxy[0] * arrow_len), int(cy - nxy[1] * arrow_len))
                cv2.arrowedLine(canvas, (cx, cy), end, (0, 255, 255), 2, tipLength=0.28)

    def render(self):
        if not self.ready:
            return

        if (
            self.external_tactile_tensor is not None
            and self.external_wrench_tensor is not None
        ):
            vectors_dict = self._tactile_tensor_to_vectors(self.external_tactile_tensor)
            scalars = {}
            vectors = {}
            wrenches = {}
            for finger_idx, prefix in enumerate(self.panel_order):
                fv = vectors_dict[prefix]["force_vectors"]
                scalars[prefix] = np.linalg.norm(fv, axis=1)
                vectors[prefix] = fv
                wrenches[prefix] = self.external_wrench_tensor[finger_idx].astype(
                    np.float64
                )
        else:
            force_vectors = self.get_sensor_force_vectors()
            wrenches_raw = self.get_finger_wrenches()
            scalars = {}
            vectors = {}
            wrenches = {}
            for prefix in self.panel_order:
                fv = force_vectors[prefix]
                vectors[prefix] = fv
                scalars[prefix] = np.linalg.norm(fv, axis=1)
                wrenches[prefix] = wrenches_raw[prefix]

        frame_max = 0.0
        nonzero_points = 0
        for prefix in self.panel_order:
            scalar_array = scalars[prefix]
            if len(scalar_array) > 0:
                frame_max = max(frame_max, float(np.max(scalar_array)))
                nonzero_points += int(np.sum(scalar_array > 1e-12))

        self.scalar_auto_max = max(self.scalar_auto_max * 0.95, frame_max, 1e-12)

        panel_w, panel_h = 480, 280
        n_panels = len(self.panel_order)

        # 单臂（2个面板）：横向一行；双臂（4个面板）：2x2网格
        if n_panels <= 2:
            canvas_w = panel_w * n_panels + 20
            canvas_h = panel_h + 20
            layout = [(i, 0) for i in range(n_panels)]  # 单行
        else:
            canvas_w = panel_w * 2 + 20
            canvas_h = panel_h * 2 + 20
            layout = [(i % 2, i // 2) for i in range(n_panels)]  # 2x2网格

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        for panel_idx, prefix in enumerate(self.panel_order):
            col, row = layout[panel_idx]
            x0 = 10 + col * panel_w
            y0 = 10 + row * panel_h

            title = f"{self.panel_meta[prefix]['arm']} / {self.panel_meta[prefix]['finger']}"
            coords = self.site_coords_2d.get(prefix)
            self._draw_panel(
                canvas,
                x0,
                y0,
                panel_w - 10,
                panel_h - 10,
                title,
                scalars[prefix],
                vectors[prefix],
                coords,
            )

            wrench = wrenches[prefix]
            txt_f = f"F=({wrench[0]:+.2e},{wrench[1]:+.2e},{wrench[2]:+.2e})"
            txt_t = f"T=({wrench[3]:+.2e},{wrench[4]:+.2e},{wrench[5]:+.2e})"
            cv2.putText(
                canvas,
                txt_f,
                (x0 + 8, y0 + panel_h - 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (180, 220, 180),
                1,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                txt_t,
                (x0 + 8, y0 + panel_h - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (180, 200, 255),
                1,
                lineType=cv2.LINE_AA,
            )

        return canvas

    def close(self):
        if self.ready:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass
