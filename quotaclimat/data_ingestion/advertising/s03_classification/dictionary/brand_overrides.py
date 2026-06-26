from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd
from gspread.exceptions import WorksheetNotFound

from quotaclimat.data_ingestion.advertising.s03_classification.dictionary.normalize import (
    normalize, nospace)
from quotaclimat.data_ingestion.advertising.s03_classification.dictionary.sheets_loader import (
    get_client,
)
from quotaclimat.data_ingestion.advertising.s03_classification.settings import (
    BrandDictionarySettings,
)

logger = logging.getLogger(__name__)

MERGES_TAB = "brand_merges_audit"
NEAR_MISS_TAB = "brand_near_misses_audit"
OVERRIDES_TAB = "brand_overrides"

VERDICT_VALUES = ("merge", "split")  # merge = allowlist, split = blocklist

OVERRIDE_COLUMNS = [
    "key_a",
    "key_b",
    "verdict",
    "display_a",
    "display_b",
    "source",
    "reviewer",
    "decided_at",
    "dict_content_hash",
]


@dataclass
class OverridesResult:
    blocklist: set[frozenset[str]]  # pairs the reviewer marked 'split' (never merge)
    allowlist: set[frozenset[str]]  # pairs the reviewer marked 'merge' (force merge)
    verdict_by_pair: dict[frozenset[str], str]  # pair -> verdict, for re-joining onto audits


def _norm_key(display: str | None) -> str:
    return nospace(normalize(display))


def _pair(a_display: str | None, b_display: str | None) -> tuple[str, str] | None:
    ka, kb = _norm_key(a_display), _norm_key(b_display)
    if not ka or not kb or ka == kb:
        return None
    return tuple(sorted((ka, kb)))  # type: ignore[return-value]


def _verdict_for(
    verdict_by_pair: dict[frozenset[str], str], a_display: str, b_display: str
) -> str:
    pair = _pair(a_display, b_display)
    if pair is None:
        return ""
    return verdict_by_pair.get(frozenset(pair), "")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _get_or_create_ws(spreadsheet, title: str, n_cols: int):
    try:
        return spreadsheet.worksheet(title)
    except WorksheetNotFound:
        return spreadsheet.add_worksheet(title=title, rows=100, cols=max(n_cols, 1))


def _try_set_dropdown(spreadsheet, worksheet, col_index: int, n_rows: int, values) -> None:
    try:
        spreadsheet.batch_update(
            {
                "requests": [
                    {
                        "setDataValidation": {
                            "range": {
                                "sheetId": worksheet.id,
                                "startRowIndex": 1,
                                "endRowIndex": n_rows + 1,
                                "startColumnIndex": col_index,
                                "endColumnIndex": col_index + 1,
                            },
                            "rule": {
                                "condition": {
                                    "type": "ONE_OF_LIST",
                                    "values": [
                                        {"userEnteredValue": v} for v in values
                                    ],
                                },
                                "showCustomUi": True,
                                "strict": False,
                            },
                        }
                    }
                ]
            }
        )
    except Exception as e:
        logger.warning("could not set dropdown on tab %r: %s", worksheet.title, e)


def _overwrite_worksheet(
    spreadsheet,
    title: str,
    df: pd.DataFrame,
    *,
    dropdown_col: str | None = None,
    dropdown_values=VERDICT_VALUES,
):
    ws = _get_or_create_ws(spreadsheet, title, len(df.columns))
    header = list(df.columns)
    body = df.fillna("").values.tolist() if not df.empty else []
    values = [header] + body
    ws.clear()
    ws.resize(rows=max(len(values), 1), cols=max(len(header), 1))
    ws.update(range_name="A1", values=values)
    if dropdown_col is not None and dropdown_col in header and body:
        _try_set_dropdown(spreadsheet, ws, header.index(dropdown_col), len(body), dropdown_values)
    return ws


def _read_store(spreadsheet) -> dict[tuple[str, str], dict]:
    try:
        ws = spreadsheet.worksheet(OVERRIDES_TAB)
    except WorksheetNotFound:
        ws = spreadsheet.add_worksheet(
            title=OVERRIDES_TAB, rows=100, cols=len(OVERRIDE_COLUMNS)
        )
        ws.update(range_name="A1", values=[OVERRIDE_COLUMNS])
        return {}

    store: dict[tuple[str, str], dict] = {}
    for row in ws.get_all_records():
        ka = str(row.get("key_a", "")).strip()
        kb = str(row.get("key_b", "")).strip()
        if not ka or not kb:
            continue
        pair = tuple(sorted((ka, kb)))
        record = {c: row.get(c, "") for c in OVERRIDE_COLUMNS}
        record["key_a"], record["key_b"] = pair
        record["verdict"] = str(row.get("verdict", "")).strip().lower()
        store[pair] = record  # type: ignore[index]
    return store


def _write_store(spreadsheet, store: dict[tuple[str, str], dict]) -> None:
    df = pd.DataFrame(
        [{c: rec.get(c, "") for c in OVERRIDE_COLUMNS} for rec in store.values()],
        columns=OVERRIDE_COLUMNS,
    )
    if not df.empty:
        df = df.sort_values(["verdict", "key_a"]).reset_index(drop=True)
    _overwrite_worksheet(spreadsheet, OVERRIDES_TAB, df, dropdown_col="verdict")


def _harvest(spreadsheet, tab: str, variant_col: str, canonical_col: str) -> list[tuple]:
    try:
        ws = spreadsheet.worksheet(tab)
    except WorksheetNotFound:
        return []
    out: list[tuple] = []
    for row in ws.get_all_records():
        verdict = str(row.get("verdict", "")).strip().lower()
        if verdict not in VERDICT_VALUES:
            continue
        da = str(row.get(variant_col, "")).strip()
        db = str(row.get(canonical_col, "")).strip()
        pair = _pair(da, db)
        if pair is None:
            continue
        out.append((pair[0], pair[1], verdict, da, db, tab))
    return out


def sync_overrides(
    settings: BrandDictionarySettings, *, dict_content_hash: str
) -> OverridesResult:
    if settings.source != "google_sheets":
        return OverridesResult(set(), set(), {})

    client = get_client(settings.google_credentials)
    spreadsheet = client.open_by_key(settings.spreadsheet_id)

    store = _read_store(spreadsheet)
    harvested = _harvest(spreadsheet, MERGES_TAB, "brand_variant", "brand_canonical")
    harvested += _harvest(spreadsheet, NEAR_MISS_TAB, "variant", "near_canonical")

    now = _utc_now_iso()
    changed = False
    for ka, kb, verdict, da, db, source in harvested:
        pair = (ka, kb)
        prev = store.get(pair)
        if prev is None or str(prev.get("verdict", "")).strip().lower() != verdict:
            store[pair] = {
                "key_a": ka,
                "key_b": kb,
                "verdict": verdict,
                "display_a": da,
                "display_b": db,
                "source": source,
                "reviewer": (prev or {}).get("reviewer", ""),
                "decided_at": now,
                "dict_content_hash": dict_content_hash,
            }
            changed = True

    if changed:
        _write_store(spreadsheet, store)

    blocklist: set[frozenset[str]] = set()
    allowlist: set[frozenset[str]] = set()
    verdict_by_pair: dict[frozenset[str], str] = {}
    for (ka, kb), rec in store.items():
        fs = frozenset((ka, kb))
        verdict = str(rec.get("verdict", "")).strip().lower()
        if verdict == "split":
            blocklist.add(fs)
            verdict_by_pair[fs] = verdict
        elif verdict == "merge":
            allowlist.add(fs)
            verdict_by_pair[fs] = verdict
    return OverridesResult(blocklist, allowlist, verdict_by_pair)


def publish_audit(
    settings: BrandDictionarySettings,
    merges_df: pd.DataFrame,
    near_misses_df: pd.DataFrame,
    overrides: OverridesResult,
) -> None:
    if settings.source != "google_sheets":
        return

    client = get_client(settings.google_credentials)
    spreadsheet = client.open_by_key(settings.spreadsheet_id)

    merges = merges_df.copy()
    merges.insert(
        0,
        "verdict",
        [
            _verdict_for(overrides.verdict_by_pair, row["brand_variant"], row["brand_canonical"])
            for _, row in merges.iterrows()
        ],
    )
    _overwrite_worksheet(spreadsheet, MERGES_TAB, merges, dropdown_col="verdict")

    near = near_misses_df.copy()
    near.insert(
        0,
        "verdict",
        [
            _verdict_for(overrides.verdict_by_pair, row["variant"], row["near_canonical"])
            for _, row in near.iterrows()
        ],
    )
    _overwrite_worksheet(spreadsheet, NEAR_MISS_TAB, near, dropdown_col="verdict")