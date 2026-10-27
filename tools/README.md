# Inspecting trajectory files

Run from the repository root. All tools preserve inputs; use `--help` for options.

```bash
export MPI_DISABLE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

## ECSV viewer

[ecsv_reader.py](ecsv_reader.py) displays raw columns, units, metadata, and full
cell values. For a noninteractive view of an ECSV you generated:

```bash
python tools/ecsv_reader.py "$RUN_DIR/fyst_trajectory.ecsv" --view text --pager never --rows 5
```

## HDF5 viewer

[h5reader.py](h5reader.py) displays tables, the file tree, and attributes.
Its default trajectory view converts Az/El to degrees; raw datasets retain radians.

```bash
python tools/h5reader.py data/fyst_ces_000_001.h5 --view text --tree
```

Omit `--view text` for interactive browsing: arrows select, Enter opens full values,
`m` shows metadata, `?` shows help, and `q` exits. In HDF5, Tab switches to the tree.
The viewers use NumPy, Astropy, h5py, and PyYAML without importing TOAST.
[_hdf5_table.py](_hdf5_table.py) is an internal adapter, not a CLI.

## Writer round-trip check

[check_writer_roundtrip.py](check_writer_roundtrip.py) uses TOAST LoadHDF5 to check
writer output against its input ECSV, including hashes, flags, and geometry.
Each file must also pass the SimFYST loader, and each boresight quaternion must match
the one rebuilt from Az/El/roll (q and -q are equivalent).
It prints JSON on success and exits with an error on mismatch.

```bash
python tools/check_writer_roundtrip.py "$RUN_DIR/fyst_trajectory.ecsv" "$RUN_DIR/hdf5"
```
