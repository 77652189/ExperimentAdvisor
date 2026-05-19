from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiment_advisor.paths import CONFIG_DIR
from experiment_advisor.storage import now_iso, read_json, write_json


def _validate_config_name(config_name: str) -> str:
    name = config_name.strip()
    if not name:
        raise ValueError("config_name cannot be empty")
    if any(part in name for part in ("/", "\\", "..")):
        raise ValueError("config_name cannot contain path separators")
    return name


@dataclass
class ConfigManager:
    config_dir: Path = CONFIG_DIR

    def _path_for(self, config_name: str) -> Path:
        return self.config_dir / f"{_validate_config_name(config_name)}.json"

    def list_configs(self) -> list[dict[str, Any]]:
        if not self.config_dir.exists():
            return []
        configs: list[dict[str, Any]] = []
        for path in sorted(self.config_dir.glob("*.json")):
            payload = read_json(path, {})
            configs.append(
                {
                    "name": payload.get("config_name", path.stem),
                    "created_at": payload.get("created_at"),
                    "is_default": bool(payload.get("is_default", False)),
                }
            )
        return configs

    def load_config(self, config_name: str) -> dict[str, Any]:
        path = self._path_for(config_name)
        if not path.exists():
            raise ValueError(f"config not found: {config_name}")
        return read_json(path, {})

    def save_config(
        self,
        config_name: str,
        variables: list[dict[str, Any]],
        optimization_mode: str = "maximize_yield",
        objective_weights: dict[str, float] | None = None,
        is_default: bool = False,
    ) -> None:
        name = _validate_config_name(config_name)
        existing = read_json(self._path_for(name), {})
        payload = {
            "config_name": name,
            "created_at": existing.get("created_at") or now_iso(),
            "updated_at": now_iso(),
            "is_default": bool(is_default or existing.get("is_default", False)),
            "optimization_mode": optimization_mode,
            "objective_weights": objective_weights or {"yield": 1.0, "cost": 0.0, "duration": 0.0},
            "variables": variables,
        }
        write_json(self._path_for(name), payload)

    def set_default(self, config_name: str) -> None:
        target = self._path_for(config_name)
        if not target.exists():
            raise ValueError(f"config not found: {config_name}")
        for path in self.config_dir.glob("*.json"):
            payload = read_json(path, {})
            payload["is_default"] = path == target
            payload["updated_at"] = now_iso()
            write_json(path, payload)

    def delete_config(self, config_name: str) -> None:
        path = self._path_for(config_name)
        if not path.exists():
            raise ValueError(f"config not found: {config_name}")
        path.unlink()

    def get_active_config(self) -> dict[str, Any] | None:
        for path in self.config_dir.glob("*.json") if self.config_dir.exists() else []:
            payload = read_json(path, {})
            if payload.get("is_default") is True:
                return payload
        return None
