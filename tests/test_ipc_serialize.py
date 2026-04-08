from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_atom.ipc.serialize import deserialize_value, serialize_value


def test_ndarray_binary_roundtrip_preserves_shape_and_dtype() -> None:
    value = np.arange(24, dtype=np.float32).reshape(2, 3, 4).transpose(1, 0, 2)

    wire = serialize_value(value)

    assert wire["__ndarray__"] is True
    assert isinstance(wire["data"], bytes)
    assert wire["dtype"] == "float32"
    assert wire["shape"] == [3, 2, 4]

    restored = deserialize_value(wire)

    assert isinstance(restored, np.ndarray)
    assert restored.shape == (3, 2, 4)
    assert restored.dtype == np.float32
    np.testing.assert_array_equal(restored, value)


def test_ndarray_deserialize_remains_compatible_with_legacy_list_format() -> None:
    wire = {
        "__ndarray__": True,
        "data": [[1, 2, 3], [4, 5, 6]],
        "dtype": "int64",
    }

    restored = deserialize_value(wire)

    assert isinstance(restored, np.ndarray)
    assert restored.shape == (2, 3)
    assert restored.dtype == np.int64
    np.testing.assert_array_equal(restored, np.array(wire["data"], dtype=np.int64))
