from __future__ import annotations
from dataclasses import dataclass, fields
from typing import Type
import json
from pathlib import Path

@dataclass
class Saveable:
    excluded: tuple[str, ...] = ()
    def save(self, path: Path) -> None:
        path = Path(path)
        data = {}
        for field in fields(self):
            if field.name in self.excluded:
                continue
            value = getattr(self, field.name)
            match value:
                case str() | float() | None:
                    data[field.name] = value
                case list() | tuple() | dict():
                    raise TypeError(f"Unsupported field type '{type(value)}' for field '{field.name}'.")
                case Saveable():
                    value.save(path / field.name)
                case _:
                    raise TypeError(f"Unsupported field type '{type(value)}' for field '{field.name}'.")
        with path.open("w") as outfile:
            json.dump(data, outfile)

    @classmethod
    def load(cls: Type[Saveable], filename: str, excluded: list[str] = []) -> Saveable:
        with open(filename, "r") as infile:
            data = json.load(infile)
        for field in fields(cls):
            if field.name not in excluded and field.name not in data:
                raise ValueError(f"Missing field '{field.name}' in the loaded data.")
        return cls(**data)
