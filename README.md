# dj-png-codecs

DataJoint codecs for storing images as [PNG](https://www.w3.org/TR/png/) files
with schema-addressed paths.

## Overview

This package provides a DataJoint codec that stores 2D images, including binary
masks, as standard PNG files in object storage, using DataJoint's
schema-addressed storage system. PNG compression is lossless, so every array is
returned exactly, and any image viewer, browser, or file browser opens the
stored file as a picture.

## Features

- **Viewable files**: every stored image is a standard PNG
- **Lossless**: the array is returned exactly, dtype and shape included
- **Binary masks**: `bool` arrays are written as 1-bit black-and-white images,
  PNG's own format for masks, so they display correctly anywhere, and are
  returned as `bool`
- **Compact**: a 512 × 512 binary mask takes a few kilobytes instead of 256 KB
- **Schema-addressed storage**: paths mirror the database structure
  (`{schema}/{table}/{pk}/{field}_{token}.png`)
- **Lazy loading**: shape and dtype are known without downloading the file
- **Automatic registration**: the codec is available after installation

## Installation

```bash
pip install dj-png-codecs
```

Requires Python 3.10 or later and `datajoint>=2.3.4`.

## Quick Start

```python
import datajoint as dj
import numpy as np

schema = dj.Schema("my_schema")


@schema
class Segmentation(dj.Manual):
    definition = """
    segmentation_id : int32
    ---
    mask : <png@>           # a bool image, stored as a black-and-white PNG
    """


Segmentation.insert1({"segmentation_id": 1, "mask": np.eye(512, dtype=bool)})

# Fetch returns a lazy reference
ref = (Segmentation & {"segmentation_id": 1}).fetch1("mask")
print(ref.shape, ref.dtype)  # (512, 512) bool, without downloading
mask = np.asarray(ref)  # downloads and decodes; a bool array
```

## Configuration

Configure your object storage in DataJoint:

```python
dj.config["stores"] = {
    "mystore": {
        "protocol": "s3",
        "endpoint": "s3.amazonaws.com",
        "bucket": "my-bucket",
        "location": "datajoint",
    }
}
```

## Codecs

### `<png@>` / `<png@store>`

Store images as PNG files with schema-addressed paths.

**Supported arrays:**

| dtype             | shape                       | stored as                     |
| ----------------- | --------------------------- | ----------------------------- |
| `bool`            | (height, width)             | 1-bit grayscale               |
| `uint8`, `uint16` | (height, width)             | 8- or 16-bit grayscale        |
| `uint8`, `uint16` | (height, width, c), c = 1–4 | gray, gray + alpha, RGB, RGBA |

Other arrays (floats, wider integers, more than 4 channels, more than two
spatial dimensions) are refused rather than converted; store them with `<npy@>`.

**Usage:**

```python
class MyTable(dj.Manual):
    definition = """
    id : int32
    ---
    mask : <png@>           # Default store
    image : <png@s3>        # Specific store
    """
```

The encoder and decoder use numpy and zlib only.

## Development

### Setup

```bash
git clone https://github.com/datajoint/dj-png-codecs.git
cd dj-png-codecs
pip install -e ".[dev]"
```

### Testing

```bash
pytest
```

The database tests start MySQL in Docker through testcontainers and are skipped
when Docker is not available.

### Code Style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and
formatting:

```bash
ruff check src tests
ruff format src tests
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

Apache License 2.0. Copyright 2026 DataJoint Inc. See [LICENSE](LICENSE) for
details.

## Related Projects

- [DataJoint](https://datajoint.com) - Framework for scientific data pipelines
- [dj-zarr-codecs](https://github.com/datajoint/dj-zarr-codecs) - DataJoint
  codecs for Zarr arrays
- [datajoint-python](https://github.com/datajoint/datajoint-python) - DataJoint
  for Python

## Documentation & Support

- [DataJoint Documentation](https://docs.datajoint.com) - Complete DataJoint
  documentation
- [GitHub Issues](https://github.com/datajoint/dj-png-codecs/issues) - Report
  bugs and request features
