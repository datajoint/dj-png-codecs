"""PNG codec implementation for DataJoint."""

from __future__ import annotations

import struct
import zlib
from typing import TYPE_CHECKING, Any

import numpy as np
from datajoint import DataJointError
from datajoint.builtin_codecs import NpyRef, SchemaCodec

if TYPE_CHECKING:
    import numpy.typing as npt

_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# channels -> PNG color type: gray, gray + alpha, RGB, RGBA
_COLOR_TYPES = {1: 0, 2: 4, 3: 2, 4: 6}
_CHANNELS = {v: k for k, v in _COLOR_TYPES.items()}
_DTYPES = ("bool", "uint8", "uint16")
_UP = 2  # PNG filter type "Up", written on every row


def _check_image(value: Any) -> None:
    """Raise unless ``value`` is an array that PNG can hold exactly."""
    if not isinstance(value, np.ndarray):
        msg = f"<png> requires numpy.ndarray, got {type(value).__name__}"
        raise DataJointError(msg)
    if value.dtype.name not in _DTYPES:
        msg = f"<png> supports bool, uint8, and uint16 arrays, got {value.dtype}"
        raise DataJointError(msg)
    if value.ndim == 2:
        channels = 1
    elif value.ndim == 3 and value.shape[2] in _COLOR_TYPES:
        channels = value.shape[2]
    else:
        msg = (
            "<png> requires an image of shape (height, width) or "
            f"(height, width, channels) with 1 to 4 channels, got shape {value.shape}"
        )
        raise DataJointError(msg)
    if value.dtype == bool and channels != 1:
        msg = f"<png> stores bool arrays as single-channel images, got shape {value.shape}"
        raise DataJointError(msg)
    if value.shape[0] == 0 or value.shape[1] == 0:
        msg = f"<png> cannot store an empty image, got shape {value.shape}"
        raise DataJointError(msg)


def _chunk(kind: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)


def encode_png(image: npt.NDArray[Any]) -> bytes:
    """
    Encode an image array as PNG bytes.

    ``bool`` arrays are written as 1-bit grayscale, PNG's own format for bilevel
    images, which any image viewer shows as black and white. ``uint8`` and
    ``uint16`` arrays are written at 8 and 16 bits per sample. Every row uses the
    PNG "Up" filter.

    Parameters
    ----------
    image : numpy.ndarray
        Array of shape (height, width) or (height, width, channels), channels 1-4.

    Returns
    -------
    bytes
        A complete PNG file.
    """
    _check_image(image)
    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    if image.dtype == bool:
        bit_depth = 1
        # eight pixels per byte, the first in the high bit, each row padded to a whole byte
        rows = np.packbits(image.reshape(height, width), axis=1)
    else:
        bit_depth = 16 if image.dtype == np.uint16 else 8
        samples = np.ascontiguousarray(
            image, dtype=">u2" if bit_depth == 16 else np.uint8
        )
        rows = samples.view(np.uint8).reshape(height, -1)
    # Up filter: each byte minus the byte above it, modulo 256
    filtered = rows.copy()
    filtered[1:] -= rows[:-1]
    raw = np.hstack([np.full((height, 1), _UP, np.uint8), filtered])
    color_type = _COLOR_TYPES[channels]
    header = struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0)
    return (
        _SIGNATURE
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(raw.tobytes(), 6))
        + _chunk(b"IEND", b"")
    )


def decode_png(data: bytes) -> npt.NDArray[Any]:
    """
    Decode PNG bytes written by :func:`encode_png`.

    Supports non-interlaced 1-bit grayscale images, and grayscale, gray + alpha,
    RGB, and RGBA images at 8 or 16 bits per sample, with the PNG filter types
    None, Sub, and Up.

    Parameters
    ----------
    data : bytes
        A complete PNG file.

    Returns
    -------
    numpy.ndarray
        ``bool`` array of shape (height, width) for a 1-bit image; otherwise a
        ``uint8`` or ``uint16`` array of shape (height, width) or
        (height, width, channels).
    """
    if data[:8] != _SIGNATURE:
        msg = "Not a PNG file"
        raise DataJointError(msg)
    pos, header, idat = 8, None, []
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        kind = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat.append(body)
        elif kind == b"IEND":
            break
    if header is None:
        msg = "PNG file has no header"
        raise DataJointError(msg)
    width, height, bit_depth, color_type, _, _, interlace = header
    bilevel = bit_depth == 1 and color_type == 0
    if not (bilevel or (bit_depth in (8, 16) and color_type in _CHANNELS)) or interlace:
        msg = (
            "<png> reads non-interlaced images: 1-bit gray, and 8- and 16-bit gray, "
            f"gray + alpha, RGB, and RGBA (got bit depth {bit_depth}, color type "
            f"{color_type}, interlace {interlace})"
        )
        raise DataJointError(msg)
    channels = _CHANNELS[color_type]
    # bytes per pixel, which the Sub filter steps by: one for a 1-bit image
    pixel_bytes = max(1, channels * bit_depth // 8)
    row_bytes = (width * channels * bit_depth + 7) // 8
    stream = np.frombuffer(zlib.decompress(b"".join(idat)), np.uint8)
    raw = stream.reshape(height, 1 + row_bytes)
    filters, rows = raw[:, 0], raw[:, 1:].copy()
    if (filters == _UP).all():  # as encode_png writes: undo Up for all rows at once
        rows = np.cumsum(rows, axis=0, dtype=np.uint8)
        filters = np.zeros_like(filters)
    for r in range(height):
        if filters[r] == 0:
            continue
        if filters[r] == 1:  # Sub: running sum along the row, per byte of the pixel
            pixels = rows[r].reshape(-1, pixel_bytes)
            rows[r] = np.cumsum(pixels, axis=0, dtype=np.uint8).ravel()
        elif filters[r] == _UP:
            if r:
                rows[r] += rows[r - 1]
        else:
            msg = f"<png> does not decode PNG filter type {filters[r]}"
            raise DataJointError(msg)
    if bilevel:
        return np.unpackbits(rows, axis=1, count=width).astype(bool)
    image: npt.NDArray[Any] = (
        rows.view(">u2").astype(np.uint16) if bit_depth == 16 else rows
    )
    shape = (height, width) if channels == 1 else (height, width, channels)
    return image.reshape(shape)


class PngRef(NpyRef):  # type: ignore[misc]
    """
    Lazy reference to an image stored as a .png file.

    Like :class:`NpyRef`, it gives the shape and dtype without I/O and loads the
    array on first use. A ``bool`` image is returned as ``bool``.
    """

    __slots__ = ()

    def load(self, mmap_mode: str | None = None) -> npt.NDArray[Any]:
        """
        Download, decode, and return the image array.

        Parameters
        ----------
        mmap_mode : None
            Memory mapping is not supported for PNG files.

        Returns
        -------
        numpy.ndarray
            The image, with the dtype and shape it was stored with.
        """
        if mmap_mode is not None:
            msg = "<png> images cannot be memory-mapped"
            raise DataJointError(msg)
        if self._cached is None:  # type: ignore[has-type]
            image = decode_png(self._backend.get_buffer(self.path))
            # a bool image is 1-bit, or 0 and 255 at 8 bits as earlier versions wrote it
            if self.dtype == bool:
                image = image > 0
            self._cached = image.reshape(self.shape)  # type: ignore[misc]  # slot of NpyRef
        return self._cached

    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded else "not loaded"
        return f"PngRef(shape={self.shape}, dtype={self.dtype}, {status})"


def _require_context(context: dict[str, Any] | None, *names: str) -> dict[str, Any]:
    """
    Return ``context``, checked to carry ``names``.

    DataJoint 2.3.4 and later pass ``context`` to every codec that declares it:
    ``schema``, ``table``, ``field``, and ``config`` on insert, ``config`` on
    fetch. The codec reads them there only, never from underscore keys in ``key``.
    """
    missing = [n for n in names if context is None or context.get(n) is None]
    if context is None or missing:
        msg = (
            "<png> needs the context that DataJoint 2.3.4 and later pass to codecs "
            f"(missing: {', '.join(missing)}); install datajoint>=2.3.4"
        )
        raise DataJointError(msg)
    return context


def _primary_key(key: dict[str, Any] | None) -> dict[str, Any]:
    """The primary key values in ``key``, without the underscore keys DataJoint 2.3 adds."""
    return {k: v for k, v in (key or {}).items() if not k.startswith("_")}


class PngCodec(SchemaCodec):  # type: ignore[misc]
    """
    Schema-addressed storage for images as .png files.

    The ``<png@>`` codec stores 2D images as standard PNG files, which any image
    viewer or browser displays, using the same schema-addressed paths as ``<npy@>``:
    ``{schema_prefix}/{schema}/{table}/{pk}/{attribute}_{token}.png``.
    PNG compression is lossless, so the stored array is returned exactly.

    Store only - requires ``@`` modifier. Installing ``dj-png-codecs`` registers
    ``<png@>`` with DataJoint through the ``datajoint.codecs`` entry point.

    Supported arrays:
        - ``bool`` of shape (height, width): written as a 1-bit black-and-white
          image, returned as ``bool``
        - ``uint8`` or ``uint16`` of shape (height, width): grayscale
        - ``uint8`` or ``uint16`` of shape (height, width, c), c = 1 to 4:
          gray, gray + alpha, RGB, RGBA

    Other arrays (floats, more than 4 channels, more than two spatial dimensions)
    are refused; store them with ``<npy@>``.

    Example::

        @schema
        class Segmentation(dj.Computed):
            definition = '''
            -> Image
            ---
            mask : <png@>      # bool array, viewable as a black-and-white image
            '''

        Segmentation.insert1({**key, 'mask': image > threshold})

        ref = (Segmentation & key).fetch1('mask')  # PngRef (lazy)
        ref.shape, ref.dtype                       # no download
        mask = np.asarray(ref)                     # bool array

    Storage Details:
        - File format: PNG, non-interlaced; 1 bit per pixel for ``bool``,
          otherwise 8 or 16 bits per sample
        - Path: ``{schema_prefix}/{schema}/{table}/{pk}/{attribute}_{token}.png``
        - Database column: JSON with ``{path, store, dtype, shape}``

    Deletion: Requires garbage collection via ``dj.gc.GarbageCollector``.

    See Also
    --------
    NpyCodec : Schema-addressed storage for any numpy array.
    PngRef : The lazy image reference returned on fetch.
    """

    name = "png"

    def validate(self, value: Any) -> None:
        """
        Validate that value is an image array that PNG can hold exactly.

        Raises
        ------
        DataJointError
            If value is not a bool, uint8, or uint16 array of shape (height, width)
            or (height, width, 1 to 4).
        """
        _check_image(value)

    def encode(
        self,
        value: Any,
        *,
        key: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
        store_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Encode the image as PNG and upload it to storage.

        Returns
        -------
        dict
            JSON metadata: ``{path, store, dtype, shape}``.

        Raises
        ------
        DataJointError
            If ``context`` lacks the ``schema``, ``table``, ``field``, or
            ``config`` that DataJoint 2.3.4 and later pass.
        """
        context = _require_context(context, "schema", "table", "field", "config")
        config = context["config"]
        path, _ = self._build_path(
            context["schema"],
            context["table"],
            context["field"],
            _primary_key(key),
            ext=".png",
            store_name=store_name,
            config=config,
        )
        backend = self._get_backend(store_name, config=config)
        backend.put_buffer(encode_png(value), path)
        # Record the store by name; DataJoint 2.3 has no resolver and keeps None for the default store.
        resolve = getattr(config, "resolve_store_name", None)
        return {
            "path": path,
            "store": resolve(store_name) if resolve else store_name,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }

    def decode(
        self,
        stored: dict[str, Any],
        *,
        key: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> PngRef:
        """
        Create a lazy PngRef from stored metadata.

        Raises
        ------
        DataJointError
            If ``context`` lacks the ``config`` that DataJoint 2.3.4 and later pass.
        """
        del key  # the stored metadata locates the file
        config = _require_context(context, "config")["config"]
        backend = self._get_backend(stored.get("store"), config=config)
        return PngRef(stored, backend)
