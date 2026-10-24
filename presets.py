"""Three blur strengths. Detection settings are not part of a preset."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Preset:
    name: str
    pad: float
    kernel_frac: float
    passes: int
    feather: int

    def text(self):
        return (
            f"name={self.name}\n"
            f"pad={self.pad:.4f}\n"
            f"kernel_frac={self.kernel_frac:.4f}\n"
            f"passes={self.passes}\n"
            f"feather={self.feather}\n"
        )

    def same_numbers(self, other):
        return (
            abs(self.pad - other.pad) < 1e-4
            and abs(self.kernel_frac - other.kernel_frac) < 1e-4
            and self.passes == other.passes
            and self.feather == other.feather
        )


# medium is the measured night blur.
# soft and hard change only the spot, not the detector.
LEVELS = (
    Preset("soft", pad=0.55, kernel_frac=1 / 3, passes=1, feather=11),
    Preset("medium", pad=1.05, kernel_frac=0.5, passes=2, feather=15),
    Preset("hard", pad=1.40, kernel_frac=2 / 3, passes=2, feather=21),
)
BY_NAME = {item.name: item for item in LEVELS}


def parse_preset(path):
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    name = values.get("name", "medium")
    return Preset(
        name=name,
        pad=float(values["pad"]),
        kernel_frac=float(values["kernel_frac"]),
        passes=int(values["passes"]),
        feather=int(values["feather"]),
    )


def conf_dir(folder):
    return Path(folder) / "conf"


def load_current(folder):
    pointer = conf_dir(folder) / "current"
    if not pointer.is_file():
        return None
    name = pointer.read_text(encoding="utf-8").strip()
    path = conf_dir(folder) / name
    if not path.is_file():
        return None
    return parse_preset(path)


def _next_path(folder, level):
    plain = folder / f"{level}.txt"
    if not plain.exists():
        return plain
    number = 2
    while True:
        path = folder / f"{level}-{number}.txt"
        if not path.exists():
            return path
        number += 1


def save_choice(folder, preset):
    """Point conf/current at these numbers. Write a new file only if they are new."""
    store = conf_dir(folder)
    store.mkdir(exist_ok=True)
    for path in sorted(store.glob("*.txt")):
        try:
            existing = parse_preset(path)
        except (OSError, KeyError, ValueError):
            continue
        if existing.same_numbers(preset):
            (store / "current").write_text(path.name + "\n", encoding="utf-8")
            return path
    path = _next_path(store, preset.name)
    path.write_text(preset.text(), encoding="utf-8")
    (store / "current").write_text(path.name + "\n", encoding="utf-8")
    return path
