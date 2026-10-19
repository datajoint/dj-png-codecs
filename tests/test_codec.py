from __future__ import annotations

import struct
import zlib

import datajoint as dj
import numpy as np
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
    if image.dtype == bool:
        decoded = decoded > 0
    np.testing.assert_array_equal(decoded.reshape(image.shape), image)


def test_signature_and_header() -> None:
    png = encode_png(IMAGES["uint16 RGBA"])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    width, height, bit_depth, color_type = struct.unpack(">IIBB", png[16:26])
    assert (width, height, bit_depth, color_type) == (32, 24, 16, 6)


def test_bool_written_as_0_and_255() -> None:
    mask = np.zeros((8, 8), bool)
    mask[2:5, 3:6] = True
    pixels = _pixels(encode_png(mask))
    assert set(np.unique(pixels)) == {0, 255}
    np.testing.assert_array_equal(pixels == 255, mask)


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
    png = bytearray(encode_png(image))
    start = png.index(b"IDAT") - 4
    (length,) = struct.unpack(">I", png[start : start + 4])
    body = zlib.compress(raw.tobytes())
    chunk = (
        struct.pack(">I", len(body))
        + b"IDAT"
        + body
        + struct.pack(">I", zlib.crc32(b"IDAT" + body) & 0xFFFFFFFF)
    )
    png[start : start + 12 + length] = chunk
    np.testing.assert_array_equal(decode_png(bytes(png)), image)


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
    loaded = np.asarray(ref)
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
