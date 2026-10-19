"""Read-only, page-cached HDF5 adapters for the existing ECSV grid viewer."""

import itertools
import json

import h5py
import numpy as np
import yaml

TOAST_UNITS = {
    "times": "s",
    "fyst_time_s": "s",
    "azimuth": "rad",
    "elevation": "rad",
    "fyst_az_deg": "deg",
    "fyst_el_deg": "deg",
    "fyst_az_velocity_deg_s": "deg / s",
    "fyst_el_velocity_deg_s": "deg / s",
    "boresight_azel": "1",
    "boresight_radec": "1",
    "flags": "bitmask",
    "fyst_scan_flag": "enum",
    "scan_index": "index",
    "subscan_index": "index",
}


def decoded(value):
    """Convert HDF5 attributes into readable values without executing embedded data."""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return decoded(value.tolist())
    if isinstance(value, (list, tuple)):
        return [decoded(item) for item in value]
    if isinstance(value, np.generic):
        return decoded(value.item())
    if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            return json.loads(value)
        except ValueError:
            pass
    return value


def attributes(obj):
    """Keep names and scalar/JSON attributes available in the metadata inspector."""
    return {key: decoded(value) for key, value in obj.attrs.items()}


class Column:
    """Expose a column backed by at most 256 rows of a real HDF5 dataset."""

    def __init__(
        self, dataset, *, field=None, component=(), unit=None, degrees=False, detector=None
    ):
        self.dataset, self.field, self.component = dataset, field, component
        self.unit, self.degrees, self.detector = unit, degrees, detector
        dtype = dataset.dtype.fields[field][0] if field is not None else dataset.dtype
        self.dtype = dtype.subdtype[0] if dtype.subdtype else dtype
        self.begin, self.cache = -1, None

    def __getitem__(self, index):
        if self.dataset.shape == ():
            value = self.dataset[()]
        else:
            count = self.dataset.shape[0 if self.detector is None else 1]
            if not 0 <= index < count:
                raise IndexError(index)
            begin = index // 256 * 256
            if begin != self.begin:
                end = min(begin + 256, count)
                self.cache = (
                    self.dataset[begin:end]
                    if self.detector is None
                    else self.dataset[self.detector, begin:end]
                )
                self.begin = begin
            value = self.cache[index - begin]
        if self.field is not None:
            value = value[self.field]
        if self.component:
            value = value[self.component]
        return np.degrees(value) if self.degrees else decoded(value)


class TableView:
    """Minimal read-only table interface consumed by GridViewer."""

    def __init__(self, columns, size, meta, title):
        self.columns, self.colnames = columns, list(columns)
        self.size, self.meta, self.title = size, meta, title

    def __len__(self):
        return self.size

    def __getitem__(self, name):
        return self.columns[name]


def dataset_columns(dataset, prefix="", detector=None):
    """Flatten vector/record fields into columns; cap very wide generic tensors."""
    columns = {}
    unit = decoded(dataset.attrs.get("unit", dataset.attrs.get("units")))
    if "toast_format_version" in dataset.file.attrs and dataset.name.startswith("/shared/"):
        unit = TOAST_UNITS.get(dataset.name.rsplit("/", 1)[-1], unit)
    field_units = {}
    schema_name = dataset.name + ".__table_column_meta__"
    if dataset.dtype.names and schema_name in dataset.file:
        lines = [decoded(line) for line in dataset.file[schema_name][:1000]]
        # 只读普通列描述，不解析后面的 Astropy tagged metadata。
        schema = yaml.safe_load("\n".join(lines).split("\nmeta:", 1)[0]) or {}
        field_units = {field["name"]: field.get("unit") for field in schema.get("datatype", [])}
    fields = dataset.dtype.names or (None,)
    for field in fields:
        dtype = dataset.dtype.fields[field][0] if field is not None else dataset.dtype
        shape = dtype.subdtype[1] if dtype.subdtype else ()
        shape = dataset.shape[(2 if detector is not None else 1) :] + shape
        components = list(itertools.islice(itertools.product(*(range(n) for n in shape)), 64))
        for component in components:
            suffix = "" if not component else "[" + ",".join(map(str, component)) + "]"
            name = f"{prefix}{field or ('value' if not prefix else '')}{suffix}"
            columns[name] = Column(
                dataset,
                field=field,
                component=component,
                unit=field_units.get(field, unit),
                detector=detector,
            )
    return columns


def table_view(file, node="trajectory", detector=0):
    """Create curated trajectory, combined shared, or arbitrary dataset views lazily."""
    native = "toast_format_version" in file.attrs
    if node == "trajectory":
        if not native or "shared/times" not in file:
            raise ValueError("这个文件没有 TOAST shared/times；请按 Tab 浏览 HDF5 结构")
        # degree 列是界面转换，文件内 canonical Az/El 仍为 rad。
        fields = [
            ("time_s", "fyst_time_s", "s", False),
            ("timestamp_unix_s", "times", "s", False),
            ("az_deg", "azimuth", "deg", True),
            ("el_deg", "elevation", "deg", True),
            ("scan", "scan_index", "index", False),
            ("subscan", "subscan_index", "index", False),
            ("toast_flags", "flags", "bitmask", False),
            ("fyst_flag", "fyst_scan_flag", "enum", False),
        ]
        columns = {
            label: Column(file[f"shared/{name}"], unit=unit, degrees=degrees)
            for label, name, unit, degrees in fields
            if f"shared/{name}" in file
        }
        meta = {
            "file_attributes": attributes(file),
            "shared_attributes": attributes(file["shared"]),
        }
        if "metadata/other" in file:
            meta.update(attributes(file["metadata/other"]))
        info = meta.get("trajectory_info", {})
        meta["start_utc"] = info.get("start_utc", "")
        meta["display_units"] = "Az/El shown in degrees; original shared datasets are radians"
        return TableView(columns, len(file["shared/times"]), meta, "Trajectory · degree view")
    obj = file[node]
    if isinstance(obj, h5py.Group):
        if obj.name != "/shared" or "times" not in obj:
            raise ValueError("Group 没有共同时间轴；可展开后打开 dataset，m 查看 attributes")
        count = len(obj["times"])
        columns = {}
        for name, dataset in obj.items():
            if isinstance(dataset, h5py.Dataset) and dataset.shape and dataset.shape[0] == count:
                columns.update(dataset_columns(dataset, prefix=name))
        meta = attributes(obj)
        meta["units_note"] = "Raw datasets; use trajectory view for degree conversion"
        return TableView(columns, count, meta, "shared · raw arrays")
    if obj.shape is None:
        return TableView({}, 0, attributes(obj) | {"shape": "null dataspace"}, obj.name)
    selected = None
    if obj.name.startswith("/detdata/") and obj.ndim >= 2:
        if not 0 <= detector < obj.shape[0]:
            raise ValueError(f"detector index 应在 0–{obj.shape[0] - 1} 内")
        selected = detector
    count = 1 if obj.shape == () else obj.shape[0 if selected is None else 1]
    meta = attributes(obj) | {
        "dataset": obj.name,
        "shape": list(obj.shape),
        "dtype": str(obj.dtype),
        "detector": selected,
        "display_note": "256-row cache; wide tensors show first 64 components",
    }
    if obj.dtype.names:
        meta["fields"] = list(obj.dtype.names)
    columns = dataset_columns(obj, detector=selected)
    if native and obj.name.startswith("/shared/"):
        key = obj.name.rsplit("/", 1)[-1]
        units = {
            "times": "s",
            "fyst_time_s": "s",
            "azimuth": "rad",
            "elevation": "rad",
            "fyst_az_deg": "deg",
            "fyst_el_deg": "deg",
            "fyst_az_velocity_deg_s": "deg / s",
            "fyst_el_velocity_deg_s": "deg / s",
        }
        for column in columns.values():
            column.unit = units.get(key, column.unit)
    title = obj.name + (f" · detector {selected}" if selected is not None else "")
    return TableView(columns, count, meta, title)
