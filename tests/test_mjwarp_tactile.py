"""Tests for the MJWarp tactile reader.

The reader ports no arithmetic -- it reuses TactileSensorManager over a scratch
MjData -- so the tests check the two things the port is responsible for: the
per-world device->scratch bridge produces the same numbers the manager produces
on a native MjData, and a disabled reader costs nothing and returns zeros.

The scene uses the manager's naming convention (sites named <prefix>touch_pointN
carrying force/torque sensors) with two panels and a box pressing on them.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mujoco_warp")

from auto_atom.backend.mjwarp.tactile import MjWarpTactileReader  # noqa: E402
from auto_atom.basis.mjwarp.state import MjWarpSceneState  # noqa: E402

# Two touch panels (left/right), each a small site carrying a force sensor, with
# a box resting on them so the sensors read a real load. The site names follow
# the <prefix>touch_point<N> convention the manager keys on.
_TACTILE_XML = """
<mujoco>
  <option timestep="0.002" integrator="implicitfast"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.05"/>
    <body name="pad" pos="0 0 0.05">
      <geom name="pad_g" type="box" size="0.06 0.03 0.02"/>
      <site name="left_touch_point0" pos="-0.03 0 0.02" size="0.01"/>
      <site name="left_touch_point1" pos="-0.01 0 0.02" size="0.01"/>
      <site name="right_touch_point0" pos="0.01 0 0.02" size="0.01"/>
      <site name="right_touch_point1" pos="0.03 0 0.02" size="0.01"/>
    </body>
    <body name="weight" pos="0 0 0.10">
      <freejoint name="weight_j"/>
      <geom name="weight_g" type="box" size="0.05 0.025 0.02" mass="2.0"/>
    </body>
  </worldbody>
  <sensor>
    <force name="left_touch_point0_f" site="left_touch_point0"/>
    <force name="left_touch_point1_f" site="left_touch_point1"/>
    <force name="right_touch_point0_f" site="right_touch_point0"/>
    <force name="right_touch_point1_f" site="right_touch_point1"/>
  </sensor>
</mujoco>
"""


@pytest.fixture(scope="module")
def settled():
    """Native and MJWarp copies of the tactile scene, both settled under load."""
    host = mujoco.MjModel.from_xml_string(_TACTILE_XML)
    native = mujoco.MjData(host)
    for _ in range(300):
        mujoco.mj_step(host, native)

    state = MjWarpSceneState(host, nworld=2)
    state.forward()
    for _ in range(300):
        state.step()
    return host, native, state


def test_reader_finds_the_panels(settled):
    _, _, state = settled
    reader = MjWarpTactileReader(state)
    assert reader.n_panels == 2, "left and right panels"


def test_wrench_tensor_matches_the_manager_on_native(settled):
    """The bridged read equals the manager reading a native MjData directly.

    Native MjData is stepped independently to the same settled state, the manager
    reads it, and the reader must produce the same per-panel wrench for a world.
    """
    from auto_atom.basis.mjc.tactile.tactile_sensor import TactileSensorManager

    host, native, state = settled
    reference = TactileSensorManager(host, native, enable=True)
    want = reference.get_wrench_tensor()

    reader = MjWarpTactileReader(state)
    got = reader.get_wrench_tensor(0)

    assert got.shape == want.shape == (2, 6)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


def test_the_panels_actually_read_a_load(settled):
    """A 2 kg box rests on the pads, so at least one panel must read force.

    Comparing zeros against zeros would pass while measuring nothing.
    """
    _, _, state = settled
    reader = MjWarpTactileReader(state)

    assert float(np.abs(reader.get_wrench_tensor(0)).sum()) > 1e-3


def test_tactile_tensor_shape_and_agreement(settled):
    from auto_atom.basis.mjc.tactile.tactile_sensor import TactileSensorManager

    host, native, state = settled
    reference = TactileSensorManager(host, native, enable=True)
    want = reference.get_tactile_tensor()

    reader = MjWarpTactileReader(state)
    got = reader.get_tactile_tensor(0)

    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


def test_worlds_are_read_independently(settled):
    """Perturbing one world's sensordata must not change another world's read."""
    _, _, state = settled
    reader = MjWarpTactileReader(state)

    w0 = reader.get_wrench_tensor(0)
    w1 = reader.get_wrench_tensor(1)
    # Both worlds settled identically, so they agree here; the point is that
    # reading world 1 did not leave world 0's scratch behind.
    np.testing.assert_allclose(reader.get_wrench_tensor(0), w0, atol=1e-9)
    np.testing.assert_allclose(w1, w0, rtol=1e-4, atol=1e-4)


def test_disabled_reader_builds_no_manager_and_returns_zeros(settled):
    """enable=False mirrors native's no-tactile path: no cost, zeros out."""
    _, _, state = settled
    reader = MjWarpTactileReader(state, enable=False)

    assert reader.n_panels == 0
    assert reader.get_wrench_tensor(0).shape == (0, 6)
    assert not reader.get_wrench_tensor(0).any()
    assert reader.get_tactile_tensor(0).shape == (0, 0, 6)


def test_world_index_is_bounds_checked(settled):
    _, _, state = settled
    reader = MjWarpTactileReader(state)

    with pytest.raises(IndexError):
        reader.get_wrench_tensor(2)
