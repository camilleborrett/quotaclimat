"""Stage 5: brand canonicalisation"""

import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

import pandas as pd
from sqlalchemy import bindparam, select
from sqlalchemy.orm import sessionmaker
from thefuzz import fuzz

from postgres.database_connection import connect_to_db
from postgres.schemas.advertising.models import Ad
from quotaclimat.data_ingestion.advertising.s03_classification.dictionary.brand_overrides import (
    OverridesResult, publish_audit, sync_overrides)
from quotaclimat.data_ingestion.advertising.s03_classification.dictionary.matcher import \
    BrandDictionary
from quotaclimat.data_ingestion.advertising.s03_classification.dictionary.normalize import (
    normalize, nospace)
from quotaclimat.data_ingestion.advertising.s03_classification.settings import (
    BrandDictionarySettings,
)
from quotaclimat.utils.logger import getLogger

LEADING_NOISE = {"le", "la", "les", "l", "de", "du", "des", "st", "saint"}
DICT_FUZZY_THRESHOLD = 92
DICT_MAX_LENGTH_DIFF = 2

MERGE_AUDIT_COLUMNS = [
    "brand_canonical",
    "brand_variant",
    "pass",
    "score",
    "sectors",
    "variant_n_ads",
    "canonical_n_ads",
]
NEAR_MISS_COLUMNS = [
    "variant",
    "near_canonical",
    "score",
    "threshold",
    "reason",
    "variant_sectors",
    "near_canonical_sectors",
    "variant_n_ads",
    "near_canonical_n_ads",
]


@dataclass
class CanonicalisationResult:
    brand_map: dict[str, str]  # brand_clean -> canonical display form
    merges: pd.DataFrame  # enriched merge audit (Job 1)
    near_misses: pd.DataFrame  # near-miss audit (Job 2)


def _ad_type_entry(prediction: list[dict] | None) -> dict:
    if not prediction:
        return {}
    for entry in prediction:
        if entry.get("stage") == "ad_type":
            return entry.get("raw_response") or {}
    return {}


def _select_rows(engine, recanonicalize_existing: bool = False) -> pd.DataFrame:
    """Rows with an LLM brand to canonicalise.

    By default, only process rows that do not have a canonical brand yet.
    recanonicalize_existing=True to rerun if changes to merge rules.
    """
    Session = sessionmaker(bind=engine)
    with Session() as session:
        stmt = select(
            Ad.id, Ad.prediction, Ad.predicted_sector, Ad.predicted_brand
        ).where(
            Ad.prediction_status.isnot(None)
        )
        if not recanonicalize_existing:
            stmt = stmt.where(Ad.predicted_brand.is_(None))
        rows = session.execute(stmt).all()
    out = []
    for ad_id, prediction, sector, predicted_brand in rows:
        ad_type = _ad_type_entry(prediction)
        brand = predicted_brand if recanonicalize_existing else ad_type.get("brand_name")
        if not brand:
            continue
        out.append(
            {
                "ad_id": ad_id,
                "brand_raw": brand,
                "brand_clean": normalize(brand),
                "sector": sector,
            }
        )
    return pd.DataFrame(out)


def _build_dict_lookup(brand_dict: BrandDictionary) -> dict[str, str]:
    """brand_key (nospace+normalised): display name from the dictionary"""
    out: dict[str, str] = {}
    for key, rows in brand_dict.t3.items():
        out[key] = rows[0][3]
    for key, entry in brand_dict.t2.items():
        out[key] = entry["brand_raw"]
    for key, (_sector, _subcat, brand_raw) in brand_dict.t1.items():
        out[key] = brand_raw
    return out


def _resolve_canonical_keys(
    canonical_of: dict[str, str],
    allowlist: set[frozenset[str]],
    dict_display_for_key,
    group_freq: dict[str, int],
) -> dict[str, str]:
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for key, canonical_key in canonical_of.items():
        union(key, canonical_key)
    for pair in allowlist:
        members = tuple(pair)
        if len(members) != 2:
            continue
        a, b = members
        if a in canonical_of and b in canonical_of:  # both present in this batch
            union(a, b)

    groups: dict[str, list[str]] = defaultdict(list)
    for key in canonical_of:
        groups[find(key)].append(key)

    rep_of_root = {
        root: min(
            keys,
            key=lambda k: (dict_display_for_key(k) is None, -group_freq.get(k, 0), k),
        )
        for root, keys in groups.items()
    }
    return {key: rep_of_root[find(key)] for key in canonical_of}


def build_canonical_audit(
    df: pd.DataFrame,
    *,
    fuzzy_threshold: int = 88,
    short_threshold: int = 92,
    short_len: int = 5,
    dict_lookup: dict[str, str] | None = None,
    near_miss_margin: int = 8,
    blocklist: set[frozenset[str]] | None = None,
    allowlist: set[frozenset[str]] | None = None,
) -> CanonicalisationResult:
    dict_lookup = dict_lookup or {}
    blocklist = blocklist or set()
    allowlist = allowlist or set()

    uniques = df["brand_clean"].unique()
    freq = df["brand_clean"].value_counts().to_dict()
    brand_keys = df["brand_clean"].map(nospace)
    group_freq = (
        df.assign(_brand_key=brand_keys)
        .groupby("_brand_key")["brand_clean"]
        .agg(lambda values: sum(freq.get(value, 0) for value in values.unique()))
        .to_dict()
    )
    sector_map = (
        df.assign(_brand_key=brand_keys)
        .groupby("_brand_key")["sector"]
        .agg(lambda s: set(s.dropna()))
        .to_dict()
    )

    def shares_sector(a_key: str, b_key: str) -> bool:
        return bool(sector_map.get(a_key, set()) & sector_map.get(b_key, set()))

    def pick_readable(variants: list[str]) -> str:
        """among spacing variants of the same brand, pick the best display form"""
        return max(
            variants,
            key=lambda v: (
                freq.get(v, 0),  # most frequent wins
                " " in v,  # prefer spaced form
                not any(v.startswith(a + " ") for a in LEADING_NOISE),  # avoid le la"
            ),
        )

    def dict_match_for_key(key: str) -> tuple[str | None, int | None]:
        if key in dict_lookup:
            return dict_lookup[key], 100

        matches = [
            (fuzz.ratio(key, dict_key), abs(len(key) - len(dict_key)), dict_key)
            for dict_key in dict_lookup
            if abs(len(key) - len(dict_key)) <= DICT_MAX_LENGTH_DIFF
        ]
        if not matches:
            return None, None

        score, _length_diff, dict_key = max(matches)
        if score >= DICT_FUZZY_THRESHOLD:
            return dict_lookup[dict_key], score
        return None, None

    def dict_display_for_key(key: str) -> str | None:
        return dict_match_for_key(key)[0]

    nospace_groups: dict[str, list[str]] = defaultdict(list)
    for clean in uniques:
        nospace_groups[nospace(clean)].append(clean)

    nospace_to_readable = {
        ns: dict_display_for_key(ns) or pick_readable(vs)
        for ns, vs in nospace_groups.items()
    }

    keys_by_freq = sorted(
        nospace_to_readable,
        key=lambda k: (
            dict_display_for_key(k) is None,  # prefer dictionary spellings
            -group_freq.get(k, 0),
        ),
    )
    canonical_of: dict[str, str] = {}
    near_miss_records: list[tuple[str, str, int, int, str]] = []
    for key in keys_by_freq:
        threshold = short_threshold if len(key) <= short_len else fuzzy_threshold
        canonicals = set(canonical_of.values())
        scored = [(fuzz.ratio(key, ck), ck, shares_sector(key, ck)) for ck in canonicals]
        match = next(
            (
                ck
                for score, ck, shares in scored
                if score >= threshold and shares and frozenset((key, ck)) not in blocklist
            ),
            None,
        )
        canonical_of[key] = match if match else key

        if match is None and scored:
            same_sector = [(score, ck) for score, ck, shares in scored if shares]
            diff_sector = [(score, ck) for score, ck, shares in scored if not shares]
            if same_sector:
                score, ck = max(same_sector)
                pair = frozenset((key, ck))
                if (
                    pair not in blocklist
                    and pair not in allowlist
                    and threshold - near_miss_margin <= score < threshold
                ):
                    near_miss_records.append((key, ck, score, threshold, "below_threshold"))
            if diff_sector:
                score, ck = max(diff_sector)
                pair = frozenset((key, ck))
                if pair not in blocklist and pair not in allowlist and score >= threshold:
                    near_miss_records.append((key, ck, score, threshold, "sector_blocked"))

    canonical_key_of = (
        _resolve_canonical_keys(canonical_of, allowlist, dict_display_for_key, group_freq)
        if allowlist
        else canonical_of
    )

    brand_map = {
        clean: nospace_to_readable[canonical_key_of[nospace(clean)]] for clean in uniques
    }

    display_n_ads: dict[str, int] = defaultdict(int)
    for clean in uniques:
        display_n_ads[brand_map[clean]] += freq.get(clean, 0)

    merge_rows: list[dict] = []
    for clean in uniques:
        key = nospace(clean)
        final_key = canonical_key_of[key]
        display = nospace_to_readable[final_key]
        if clean == display:
            continue

        fuzzy_key = canonical_of[key]
        if fuzzy_key != key:
            pass_name = "fuzzy"
            score: int | None = fuzz.ratio(key, fuzzy_key)
        elif final_key != key:
            pass_name = "allowlist"
            score = None
        else:
            dict_display, dict_score = dict_match_for_key(key)
            if dict_display is not None:
                pass_name = "dict_override"
                score = dict_score
            else:
                pass_name = "spacing"
                score = 100

        merge_rows.append(
            {
                "brand_canonical": display,
                "brand_variant": clean,
                "pass": pass_name,
                "score": score,
                "sectors": "|".join(sorted(sector_map.get(key, set()))),
                "variant_n_ads": freq.get(clean, 0),
                "canonical_n_ads": display_n_ads[display],
            }
        )

    merges = pd.DataFrame(merge_rows, columns=MERGE_AUDIT_COLUMNS)
    if not merges.empty:
        merges = merges.sort_values(
            ["score", "canonical_n_ads"], ascending=[True, False], na_position="last"
        ).reset_index(drop=True)

    near_miss_rows = [
        {
            "variant": nospace_to_readable[key],
            "near_canonical": nospace_to_readable[ck],
            "score": score,
            "threshold": threshold,
            "reason": reason,
            "variant_sectors": "|".join(sorted(sector_map.get(key, set()))),
            "near_canonical_sectors": "|".join(sorted(sector_map.get(ck, set()))),
            "variant_n_ads": group_freq.get(key, 0),
            "near_canonical_n_ads": group_freq.get(ck, 0),
        }
        for key, ck, score, threshold, reason in near_miss_records
    ]
    near_misses = pd.DataFrame(near_miss_rows, columns=NEAR_MISS_COLUMNS)
    if not near_misses.empty:
        near_misses = near_misses.sort_values(
            ["score", "variant_n_ads"], ascending=[False, False]
        ).reset_index(drop=True)

    return CanonicalisationResult(
        brand_map=brand_map, merges=merges, near_misses=near_misses
    )


_AD = Ad.__table__
_UPDATE = (
    _AD.update()
    .where(_AD.c.id == bindparam("b_id"))
    .values(predicted_brand=bindparam("b_brand"))
)


def _flush(session_factory, rows: list[dict], batch_size: int) -> None:
    if not rows:
        return
    with session_factory() as session:
        for i in range(0, len(rows), batch_size):
            session.execute(_UPDATE, rows[i : i + batch_size])
        session.commit()


def _write_audit(df: pd.DataFrame, path: str, label: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logging.info("[canonicalise_brand] wrote %d %s to %s", len(df), label, path)


def run(
    audit_path: str = "data/brand_merges.csv",
    near_miss_path: str = "data/brand_merge_near_misses.csv",
    batch_size: int = 1000,
    recanonicalize_existing: bool = False,
    near_miss_margin: int = 8,
    sync_sheet_audit: bool = True,
) -> dict[str, int]:
    load_dotenv()
    engine = connect_to_db(use_custom_json_serializer=True)
    session_factory = sessionmaker(bind=engine)
    df = _select_rows(engine, recanonicalize_existing=recanonicalize_existing)
    if df.empty:
        logging.info("[canonicalise_brand] nothing to process")
        engine.dispose()
        return {"processed": 0}

    settings = BrandDictionarySettings()  # type: ignore[call-arg]
    brand_dict = BrandDictionary(settings=settings)
    dict_lookup = _build_dict_lookup(brand_dict)
    logging.info(
        "[canonicalise_brand] dict spellings available for %d brand keys",
        len(dict_lookup),
    )

    use_sheets = sync_sheet_audit and settings.source == "google_sheets"
    overrides = OverridesResult(set(), set(), {})
    if use_sheets:
        try:
            overrides = sync_overrides(
                settings, dict_content_hash=brand_dict.version.content_hash
            )
            logging.info(
                "[canonicalise_brand] overrides loaded: %d block, %d allow",
                len(overrides.blocklist),
                len(overrides.allowlist),
            )
        except Exception as e:  # noqa: BLE001
            logging.warning(
                "[canonicalise_brand] override sync failed, proceeding without: %s", e
            )

    result = build_canonical_audit(
        df,
        dict_lookup=dict_lookup,
        near_miss_margin=near_miss_margin,
        blocklist=overrides.blocklist,
        allowlist=overrides.allowlist,
    )
    df["brand_canonical"] = df["brand_clean"].map(result.brand_map)

    _write_audit(result.merges, audit_path, "merges")
    _write_audit(result.near_misses, near_miss_path, "near-misses")

    if use_sheets:
        try:
            publish_audit(settings, result.merges, result.near_misses, overrides)
            logging.info(
                "[canonicalise_brand] published audit tabs to spreadsheet %s",
                settings.spreadsheet_id,
            )
        except Exception as e:  # noqa: BLE001
            logging.warning(
                "[canonicalise_brand] publishing audit to sheets failed: %s", e
            )

    rows = [
        {"b_id": r.ad_id, "b_brand": r.brand_canonical}
        for r in df.itertuples(index=False)
    ]
    try:
        _flush(session_factory, rows, batch_size)
    finally:
        engine.dispose()

    logging.info("[canonicalise_brand] done. processed=%d", len(rows))
    return {
        "processed": len(rows),
        "merges": len(result.merges),
        "near_misses": len(result.near_misses),
        "overrides_block": len(overrides.blocklist),
        "overrides_allow": len(overrides.allowlist),
    }


if __name__ == "__main__":
    getLogger()
    run(
        audit_path=os.environ.get("AUDIT_PATH", "data/brand_merges.csv"),
        near_miss_path=os.environ.get(
            "NEAR_MISS_PATH", "data/brand_merge_near_misses.csv"
        ),
        batch_size=int(os.environ.get("BATCH_SIZE", 1000)),
        recanonicalize_existing=os.environ.get("RECANONICALIZE_EXISTING", "0") == "1",
        near_miss_margin=int(os.environ.get("NEAR_MISS_MARGIN", 8)),
        sync_sheet_audit=os.environ.get("SYNC_SHEET_AUDIT", "1") == "1",
    )