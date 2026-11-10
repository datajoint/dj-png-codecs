from __future__ import annotations

import struct
import zlib
from typing import Any

import datajoint as dj
import numpy as np
import numpy.typing as npt
import pytest

from dj_png_codecs import PngCodec, PngRef, decode_png, encode_png

RNG = np.random.default_rng(0)
IMAGES = {
    "bool": RNG.random((24, 32)) > 0.5,
    "uint8": RNG.integers(0, 256, (24, 32), dtype=np.uint8),
    "uint16": RNG.integers(0, 65536, (24, 32), dtype=np.uint16),
    "uint8 single channel": RNG.integers(0, 256, (24, 32, 1), dtype=np.uint8),
    "uint8 gray + alpha": RNG.integers(0, 256, (24, 32, 2), dtype=np.uint8),
    "uint8 RGB": RNG.integers(0, 256, (24, 32, 3), dtype=np.uint8),
    "uint16 RGBA": RNG.integers(0, 65536, (24, 32, 4), dtype=np.uint16),
}


def _header(png: bytes) -> tuple[int, int, int, int]:
    """Width, height, bit depth, and color type."""
    width, height, bit_depth, color_type = struct.unpack(">IIBB", png[16:26])
    return width, height, bit_depth, color_type


def _with_scanlines(png: bytes, raw: np.ndarray) -> bytes:
    """The PNG with its image data replaced by these filtered scanlines."""
    out = bytearray(png)
    start = out.index(b"IDAT") - 4
    (length,) = struct.unpack(">I", out[start : start + 4])
    body = zlib.compress(raw.tobytes())
    chunk = (
        struct.pack(">I", len(body))
        + b"IDAT"
        + body
        + struct.pack(">I", zlib.crc32(b"IDAT" + body) & 0xFFFFFFFF)
    )
    out[start : start + 12 + length] = chunk
    return bytes(out)


class _Files:
    """A storage backend holding files in memory, for loading a PngRef without a store."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files

    def get_buffer(self, path: str) -> bytes:
        return self.files[path]


def _pixels(png: bytes) -> np.ndarray:
    """Decompressed scanlines of a PNG written with the Up filter, filters undone."""
    pos, idat, header = 8, [], None
    while pos < len(png):
        (length,) = struct.unpack(">I", png[pos : pos + 4])
        kind, body = png[pos + 4 : pos + 8], png[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat.append(body)
    assert header is not None
    height = header[1]
    raw = np.frombuffer(zlib.decompress(b"".join(idat)), np.uint8).reshape(height, -1)
    return np.cumsum(raw[:, 1:], axis=0, dtype=np.uint8)


# Format, without a database


@pytest.mark.parametrize("name", IMAGES)
def test_format_roundtrip(name: str) -> None:
    image = IMAGES[name]
    decoded = decode_png(encode_png(image))
    assert decoded.dtype == image.dtype
    np.testing.assert_array_equal(decoded.reshape(image.shape), image)


def test_signature_and_header() -> None:
    png = encode_png(IMAGES["uint16 RGBA"])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert _header(png) == (32, 24, 16, 6)


@pytest.mark.parametrize("width", [1, 7, 8, 9, 33])
def test_bool_written_as_1_bit(width: int) -> None:
    mask = RNG.random((5, width)) > 0.5
    png = encode_png(mask)
    assert _header(png) == (width, 5, 1, 0)
    # eight pixels per byte, the first in the high bit, each row padded to a whole byte
    np.testing.assert_array_equal(_pixels(png), np.packbits(mask, axis=1))
    decoded = decode_png(png)
    assert decoded.dtype == bool
    np.testing.assert_array_equal(decoded, mask)


def test_bool_single_channel_shape() -> None:
    mask = RNG.random((6, 10, 1)) > 0.5
    png = encode_png(mask)
    assert _header(png) == (10, 6, 1, 0)
    np.testing.assert_array_equal(decode_png(png), mask[:, :, 0])


def test_8_bit_masks_still_load_as_bool() -> None:
    """Masks written by earlier versions, as 0 and 255 at 8 bits, load as bool."""
    mask = IMAGES["bool"]
    png = encode_png(np.where(mask, np.uint8(255), np.uint8(0)))
    assert _header(png)[2] == 8
    ref = PngRef(
        {
            "path": "x/mask.png",
            "store": None,
            "dtype": "bool",
            "shape": list(mask.shape),
        },
        backend=_Files({"x/mask.png": png}),
    )
    loaded = ref.load()
    assert loaded.dtype == bool
    np.testing.assert_array_equal(loaded, mask)


def test_binary_mask_compresses() -> None:
    mask = np.zeros((512, 512), bool)
    mask[100:200, 150:300] = True
    assert len(encode_png(mask)) < 4096


def test_decode_sub_filter() -> None:
    """Files from other writers may use the Sub filter."""
    image = IMAGES["uint8 RGB"].astype(np.uint8)
    rows = image.reshape(image.shape[0], -1)
    sub = rows.copy()
    sub[:, 3:] -= rows[:, :-3]
    raw = np.hstack([np.ones((rows.shape[0], 1), np.uint8), sub])
    png = _with_scanlines(encode_png(image), raw)
    np.testing.assert_array_equal(decode_png(png), image)


def test_decode_1_bit_sub_filter() -> None:
    """In a 1-bit image, the Sub filter steps by one byte."""
    mask = RNG.random((6, 21)) > 0.5
    rows = np.packbits(mask, axis=1)
    sub = rows.copy()
    sub[:, 1:] -= rows[:, :-1]
    raw = np.hstack([np.ones((rows.shape[0], 1), np.uint8), sub])
    png = _with_scanlines(encode_png(mask), raw)
    np.testing.assert_array_equal(decode_png(png), mask)


def test_decode_refuses_2_bit() -> None:
    png = bytearray(encode_png(IMAGES["bool"]))
    png[24] = 2  # bit depth, in IHDR
    with pytest.raises(dj.DataJointError, match="bit depth 2"):
        decode_png(bytes(png))


@pytest.mark.parametrize(
    "value",
    [
        np.zeros((4, 4), np.float32),
        np.zeros((4, 4), np.int16),
        np.zeros((4, 4), np.uint32),
        np.zeros((4, 4, 5), np.uint8),
        np.zeros((2, 4, 4, 3), np.uint8),
        np.zeros(16, np.uint8),
        np.zeros((4, 4, 3), bool),
        np.zeros((0, 4), np.uint8),
        [[0, 1], [1, 0]],
    ],
    ids=[
        "float",
        "int16",
        "uint32",
        "5 channels",
        "4D",
        "1D",
        "bool RGB",
        "empty",
        "list",
    ],
)
def test_refuses(value: object) -> None:
    with pytest.raises(dj.DataJointError):
        PngCodec().validate(value)
    with pytest.raises(dj.DataJointError):
        encode_png(value)  # type: ignore[arg-type]


def test_png_ref_metadata_and_repr() -> None:
    ref = PngRef(
        {"path": "x/mask_ab.png", "store": None, "dtype": "bool", "shape": [512, 512]},
        backend=None,
    )
    assert ref.shape == (512, 512)
    assert ref.dtype == np.dtype(bool)
    assert not ref.is_loaded
    assert repr(ref) == "PngRef(shape=(512, 512), dtype=bool, not loaded)"


def test_png_ref_refuses_mmap() -> None:
    ref = PngRef(
        {"path": "x.png", "store": None, "dtype": "uint8", "shape": [2, 2]},
        backend=None,
    )
    with pytest.raises(dj.DataJointError):
        ref.load(mmap_mode="r")


# The codec context, which DataJoint 2.3.4 and later pass

# How DataJoint 2.3 still fills `key`; the codec does not read the underscore keys.
LEGACY_KEY = {
    "id": 1,
    "_schema": "s",
    "_table": "t",
    "_field": "f",
    "_config": dj.config,
}


@pytest.mark.parametrize(
    "context",
    [None, {"schema": "s", "table": "t", "field": "f"}],
    ids=["no context", "no config"],
)
def test_encode_needs_context(context: dict[str, Any] | None) -> None:
    with pytest.raises(dj.DataJointError, match=r"DataJoint 2\.3\.4"):
        PngCodec().encode(
            IMAGES["uint8"], key=LEGACY_KEY, context=context, store_name="store"
        )


def test_decode_needs_context() -> None:
    stored = {"path": "x.png", "store": None, "dtype": "uint8", "shape": [2, 2]}
    with pytest.raises(dj.DataJointError, match=r"missing: config"):
        PngCodec().decode(stored, key={"_config": dj.config}, context=None)


# Through DataJoint, with a database and a file store


def test_bool_mask_roundtrip(schema: dj.Schema) -> None:
    @schema
    class Segmentation(dj.Manual):  # type: ignore[misc]
        definition = """
        id : int32
        ---
        mask : <png@store>
        """

    mask = np.zeros((64, 48), bool)
    mask[10:30, 5:20] = True
    Segmentation.insert1({"id": 1, "mask": mask})
    ref = (Segmentation & {"id": 1}).fetch1("mask")  # type: ignore[operator]
    assert isinstance(ref, PngRef)
    assert ref.shape == (64, 48)
    assert ref.dtype == np.dtype(bool)
    assert ref.path.endswith(".png")
    loaded: npt.NDArray[Any] = np.asarray(ref)
    assert loaded.dtype == bool
    np.testing.assert_array_equal(loaded, mask)


@pytest.mark.parametrize("name", [n for n in IMAGES if n != "bool"])
def test_image_roundtrip(schema: dj.Schema, name: str) -> None:
    @schema
    class Picture(dj.Manual):  # type: ignore[misc]
        definition = """
        id : int32
        ---
        image : <png@store>
        """

    image = IMAGES[name]
    Picture.insert1({"id": 1, "image": image})
    loaded = (Picture & {"id": 1}).fetch1("image").load()  # type: ignore[operator]
    assert loaded.dtype == image.dtype
    assert loaded.shape == image.shape
    np.testing.assert_array_equal(loaded, image)


def test_insert_refuses_float(schema: dj.Schema) -> None:
    @schema
    class Picture(dj.Manual):  # type: ignore[misc]
        definition = """
        id : int32
        ---
        image : <png@store>
        """

    with pytest.raises(dj.DataJointError):
        Picture.insert1({"id": 1, "image": np.zeros((4, 4), np.float64)})
