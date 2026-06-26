"""Stage 3: rules based dictionary lookup, for ads where content_type == AD.

Two modes:

  - normal (default): process rows at ad_type_done in pipeline order, writing
    every outcome (dict_tier1/2/3, dict_tier2_no_kw, dict_tier3_no_kw, dict_miss).

  - rematch (REMATCH=1): after the dictionary has been updated (e.g. new brands
    added), re-run the matcher over rows that previously fell through to the LLM
    and only upgrade the ones a newly-added brand can now fully resolve by rule
    (tier 1/2/3). This costs no inference: the expensive stage-2 output (brand,
    product, sector) is read back from the persisted ad_type prediction entry.

    Rows that the dictionary still can't fully resolve are left completely
    untouched, so no stage-4 (LLM) sub-category work is ever discarded or
    re-triggered, and rows already resolved by the dictionary are never downgraded.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any

import tqdm
from sqlalchemy import bindparam, select
from sqlalchemy.orm import sessionmaker

from postgres.database_connection import connect_to_db
from postgres.schemas.advertising.models import Ad
from quotaclimat.data_ingestion.advertising.s03_classification.dictionary.matcher import (
    BrandDictionary, DictMatch)
from quotaclimat.utils.logger import getLogger

# Rows eligible for a rematch: everything that carries a stage-2 ad_type result but was not fully resolved by the dictionary.
# Pure dict resolutions (dict_tier1/2/3) are deliberately excluded so a rematch can only ever upgrade, never downgrade them.
# (ad_type_done is excluded too, as those are fresh rows handled by the normal mode.)
REMATCH_STATUSES = (
    "dict_miss",
    "dict_tier2_no_kw",
    "dict_tier3_no_kw",
    "subcat_done",
    "subcat_parse_error",
)

# Matcher outcomes that fully resolve a row by dictionary alone (both sector and sub-category come from the dictionary,
# consistently). Only these are committed in rematch mode.
DICT_RESOLVED_METHODS = ("dict_tier1", "dict_tier2", "dict_tier3")


def select_pending(
    engine, limit: int | None, rematch: bool = False
) -> list[tuple[str, list[dict] | None, str | None]]:
    Session = sessionmaker(bind=engine)
    with Session() as session:
        status_filter = (
            Ad.prediction_status.in_(REMATCH_STATUSES)
            if rematch
            else Ad.prediction_status == "ad_type_done"
        )
        stmt = (
            select(Ad.id, Ad.prediction, Ad.transcript)
            .where(status_filter)
            .order_by(Ad.first_detection_date.desc())
        )
        if limit:
            stmt = stmt.limit(limit)
        return [(r[0], r[1], r[2]) for r in session.execute(stmt).all()]


def _ad_type_entry(prediction: list[dict] | None) -> dict:
    if not prediction:
        return {}
    for entry in prediction:
        if entry.get("stage") == "ad_type":
            return entry.get("raw_response") or {}
    return {}


def _build_dict_match_entry(
    match: DictMatch | None,
    brand_in: str | None,
    dict_version,
) -> dict | None:
    timestamp = datetime.now(timezone.utc).isoformat()
    version_fields = {
        "dict_version": dict_version.version_id,
        "dict_content_hash": dict_version.content_hash,
    }

    if match is None:
        if not brand_in:
            return None
        return {
            "stage": "dict_match",
            "timestamp": timestamp,
            "method": "dict_miss",
            "brand_in": brand_in,
            **version_fields,
        }

    entry: dict[str, Any] = {
        "stage": "dict_match",
        "timestamp": timestamp,
        "method": match.method,
        "matched_brand": match.matched_brand,
        "sector_code": match.sector_code,
        "subcat_code": match.subcat_code,
        **version_fields,
    }
    if match.matched_keyword is not None:
        entry["matched_keyword"] = match.matched_keyword
    if match.rule_keywords is not None:
        entry["rule_keywords"] = match.rule_keywords
    if match.tried_rules is not None:
        entry["tried_rules"] = match.tried_rules
    if match.haystack is not None:
        entry["haystack"] = match.haystack
    return entry


def _plan_normal(
    ad_id: str,
    prediction: list[dict] | None,
    brand: str | None,
    llm_sector: str | None,
    match: DictMatch | None,
    dict_version,
) -> tuple[str, dict]:
    if match is None:
        status = method = "dict_miss"
        sector, subcat, canonical_brand = llm_sector, None, None
    elif match.method == "dict_tier3_no_kw":
        status = method = "dict_tier3_no_kw"
        sector, subcat, canonical_brand = llm_sector, None, None
    else:
        status = method = match.method
        sector = match.sector_code
        subcat = match.subcat_code
        canonical_brand = match.matched_brand

    existing = list(prediction) if prediction else []
    audit_entry = _build_dict_match_entry(match, brand, dict_version)
    if audit_entry is not None:
        existing = [e for e in existing if e.get("stage") != "dict_match"]
        existing.append(audit_entry)

    return status, {
        "b_id": ad_id,
        "b_prediction": existing,
        "b_status": status,
        "b_method": method,
        "b_sector": sector,
        "b_subcat": subcat,
        "b_brand": canonical_brand,
    }


def _plan_rematch(
    ad_id: str,
    prediction: list[dict] | None,
    brand: str | None,
    match: DictMatch | None,
    dict_version,
) -> tuple[str, dict | None]:
    if match is None or match.method not in DICT_RESOLVED_METHODS:
        return "unchanged", None

    status = method = match.method

    existing = list(prediction) if prediction else []

    existing = [e for e in existing if e.get("stage") not in ("dict_match", "subcat")]
    existing.append(_build_dict_match_entry(match, brand, dict_version))

    return status, {
        "b_id": ad_id,
        "b_prediction": existing,
        "b_status": status,
        "b_method": method,
        "b_sector": match.sector_code,
        "b_subcat": match.subcat_code,
        "b_brand": match.matched_brand,
    }


_AD = Ad.__table__
_UPDATE = (
    _AD.update()
    .where(_AD.c.id == bindparam("b_id"))
    .values(
        prediction=bindparam("b_prediction"),
        prediction_status=bindparam("b_status"),
        prediction_method=bindparam("b_method"),
        predicted_sector=bindparam("b_sector"),
        predicted_product_category=bindparam("b_subcat"),
        predicted_brand=bindparam("b_brand"),
    )
)


def _flush(session_factory, rows: list[dict]) -> None:
    if not rows:
        return
    with session_factory() as session:
        session.execute(_UPDATE, rows)
        session.commit()


def run(
    batch_size: int = 500,
    limit: int | None = None,
    rematch: bool = False,
    dry_run: bool = False,
) -> dict[str, int]:
    engine = connect_to_db(use_custom_json_serializer=True)
    session_factory = sessionmaker(bind=engine)
    mode = "rematch" if rematch else "normal"
    pending = select_pending(engine, limit, rematch=rematch)
    if not pending:
        logging.info("[dict_match:%s] nothing to process", mode)
        engine.dispose()
        return {}

    bd = BrandDictionary()
    logging.info(
        "[dict_match:%s] dictionary version=%s hash=%s path=%s%s",
        mode,
        bd.version.version_id,
        bd.version.content_hash[:8],
        bd.version.local_path,
        " (dry-run: nothing will be written)" if dry_run else "",
    )
    counts: dict[str, int] = {}
    buf: list[dict] = []
    progress = tqdm.tqdm(pending, desc=f"dict_match:{mode}", smoothing=0.05)

    try:
        for ad_id, prediction, transcript in progress:
            ad_type = _ad_type_entry(prediction)

            if ad_type.get("content_type") != "AD":
                continue

            brand = ad_type.get("brand_name")
            product = ad_type.get("product_name")
            llm_sector = ad_type.get("sector_code")

            match = bd.match(brand, product, transcript)

            if rematch:
                key, row = _plan_rematch(ad_id, prediction, brand, match, bd.version)
            else:
                key, row = _plan_normal(
                    ad_id, prediction, brand, llm_sector, match, bd.version
                )

            counts[key] = counts.get(key, 0) + 1
            if row is not None:
                buf.append(row)

            if len(buf) >= batch_size:
                if not dry_run:
                    _flush(session_factory, buf)
                buf = []
            progress.set_postfix(counts)
    finally:
        if not dry_run:
            _flush(session_factory, buf)
        progress.close()
        engine.dispose()

    logging.info(
        "[dict_match:%s] done.%s %s",
        mode,
        " (dry-run, nothing written)" if dry_run else "",
        counts,
    )
    return counts


if __name__ == "__main__":
    getLogger()
    run(
        batch_size=int(os.environ.get("BATCH_SIZE", 500)),
        limit=int(os.environ.get("LIMIT", 0)) or None,
        rematch=os.environ.get("REMATCH", "0") == "1",
        dry_run=os.environ.get("DRY_RUN", "0") == "1",
    )