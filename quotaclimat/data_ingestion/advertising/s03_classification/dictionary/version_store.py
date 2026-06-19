"""Versioned local cache for the brand dictionary."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd

from quotaclimat.data_ingestion.advertising.s03_classification.dictionary.sheets_loader import (
    TIER_SHEET_NAMES,
    fetch_dictionary_sheets,
    read_dictionary_xlsx,
)
from quotaclimat.data_ingestion.advertising.s03_classification.settings import (
    BrandDictionarySettings,
)

logger = logging.getLogger(__name__)
CACHED_XLSX_NAME = "dictionnaire_marques_secteurs.xlsx"


@dataclass(frozen=True, slots=True)
class DictionaryVersion:
    version_id: str
    content_hash: str
    source: Literal["local", "google_sheets"]
    local_path: Path
    fetched_at: str
    spreadsheet_id: str | None = None

    def as_metadata(self) -> dict:
        data = asdict(self)
        data["local_path"] = str(self.local_path)
        return data


def compute_dictionary_hash(sheets: dict[str, pd.DataFrame]) -> str:
    """stable SHA-256 over all tier sheet contents"""
    parts: list[str] = []
    for sheet_name in TIER_SHEET_NAMES:
        df = sheets[sheet_name].fillna("").astype(str)
        if not df.empty:
            df = df.sort_values(by=list(df.columns)).reset_index(drop=True)
        parts.append(df.to_csv(index=False))
    return hashlib.sha256("".join(parts).encode()).hexdigest()


def save_dictionary_xlsx(sheets: dict[str, pd.DataFrame], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name in TIER_SHEET_NAMES:
            sheets[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _make_version_id(content_hash: str, fetched_at: str) -> str:
    dt = datetime.fromisoformat(fetched_at)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    ts = dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{content_hash[:8]}"


def _load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        return {"latest": None, "versions": {}}
    return json.loads(manifest_path.read_text())


def _save_manifest(manifest_path: Path, manifest: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _version_from_record(record: dict) -> DictionaryVersion:
    return DictionaryVersion(
        version_id=record["version_id"],
        content_hash=record["content_hash"],
        source=record["source"],
        local_path=Path(record["local_path"]),
        fetched_at=record["fetched_at"],
        spreadsheet_id=record.get("spreadsheet_id"),
    )


def _find_version_by_hash(manifest: dict, content_hash: str) -> DictionaryVersion | None:
    for record in manifest.get("versions", {}).values():
        if record.get("content_hash") == content_hash:
            path = Path(record["local_path"])
            if path.exists():
                return _version_from_record(record)
    return None


def _register_version(
    manifest_path: Path,
    manifest: dict,
    *,
    version: DictionaryVersion,
) -> DictionaryVersion:
    manifest.setdefault("versions", {})[version.version_id] = version.as_metadata()
    manifest["latest"] = version.version_id
    _save_manifest(manifest_path, manifest)
    return version


def resolve_local_dictionary(path: Path, *, source: Literal["local", "google_sheets"] = "local") -> DictionaryVersion:
    sheets = read_dictionary_xlsx(path)
    content_hash = compute_dictionary_hash(sheets)
    fetched_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).replace(
        microsecond=0
    ).isoformat()
    return DictionaryVersion(
        version_id=f"local_{content_hash[:8]}",
        content_hash=content_hash,
        source=source,
        local_path=path,
        fetched_at=fetched_at,
    )


def sync_google_sheets_dictionary(settings: BrandDictionarySettings) -> DictionaryVersion:
    """Fetch google sheetss and reuse latest cached copy when content is unchanged."""
    versions_dir = settings.versions_dir
    manifest_path = versions_dir / "manifest.json"
    manifest = _load_manifest(manifest_path)

    remote_sheets = fetch_dictionary_sheets(settings)
    content_hash = compute_dictionary_hash(remote_sheets)

    cached = _find_version_by_hash(manifest, content_hash)
    if cached is not None:
        if manifest.get("latest") != cached.version_id:
            manifest["latest"] = cached.version_id
            _save_manifest(manifest_path, manifest)
        logger.info(
            "Dictionary unchanged (hash=%s), using cached version %s",
            content_hash[:8],
            cached.version_id,
        )
        return cached

    fetched_at = _utc_now_iso()
    version_id = _make_version_id(content_hash, fetched_at)
    cached_path = versions_dir / version_id / CACHED_XLSX_NAME
    save_dictionary_xlsx(remote_sheets, cached_path)

    version = DictionaryVersion(
        version_id=version_id,
        content_hash=content_hash,
        source="google_sheets",
        local_path=cached_path,
        fetched_at=fetched_at,
        spreadsheet_id=settings.spreadsheet_id,
    )
    _register_version(manifest_path, manifest, version=version)
    logger.info(
        "Dictionary updated, saved new version %s (hash=%s) to %s",
        version.version_id,
        content_hash[:8],
        cached_path,
    )
    return version


def resolve_dictionary(settings: BrandDictionarySettings) -> DictionaryVersion:
    if settings.source == "google_sheets":
        return sync_google_sheets_dictionary(settings)
    return resolve_local_dictionary(settings.local_path)
