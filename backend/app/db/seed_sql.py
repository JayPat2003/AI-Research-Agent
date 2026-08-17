from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.db.models import ClinicalTrial

# Public, well-known outcomes trials — demo catalog for Text-to-SQL, not a live registry.
ROWS: list[dict] = [
    {"trial_name": "EMPA-REG OUTCOME", "condition": "type 2 diabetes", "intervention": "empagliflozin", "drug_class": "SGLT2", "phase": "3", "n_participants": 7020, "region": "global", "status": "completed", "primary_endpoint": "3-point MACE", "start_year": 2010},
    {"trial_name": "CANVAS Program", "condition": "type 2 diabetes", "intervention": "canagliflozin", "drug_class": "SGLT2", "phase": "3", "n_participants": 10142, "region": "global", "status": "completed", "primary_endpoint": "3-point MACE", "start_year": 2009},
    {"trial_name": "DECLARE-TIMI 58", "condition": "type 2 diabetes", "intervention": "dapagliflozin", "drug_class": "SGLT2", "phase": "3", "n_participants": 17160, "region": "global", "status": "completed", "primary_endpoint": "3-point MACE", "start_year": 2013},
    {"trial_name": "VERTIS CV", "condition": "type 2 diabetes", "intervention": "ertugliflozin", "drug_class": "SGLT2", "phase": "3", "n_participants": 8246, "region": "global", "status": "completed", "primary_endpoint": "3-point MACE", "start_year": 2013},
    {"trial_name": "CREDENCE", "condition": "type 2 diabetes", "intervention": "canagliflozin", "drug_class": "SGLT2", "phase": "3", "n_participants": 4401, "region": "global", "status": "completed", "primary_endpoint": "kidney composite", "start_year": 2014},
    {"trial_name": "DAPA-CKD", "condition": "chronic kidney disease", "intervention": "dapagliflozin", "drug_class": "SGLT2", "phase": "3", "n_participants": 4304, "region": "global", "status": "completed", "primary_endpoint": "kidney composite", "start_year": 2017},
    {"trial_name": "EMPA-KIDNEY", "condition": "chronic kidney disease", "intervention": "empagliflozin", "drug_class": "SGLT2", "phase": "3", "n_participants": 6609, "region": "global", "status": "completed", "primary_endpoint": "kidney composite", "start_year": 2019},
    {"trial_name": "DAPA-HF", "condition": "heart failure", "intervention": "dapagliflozin", "drug_class": "SGLT2", "phase": "3", "n_participants": 4744, "region": "global", "status": "completed", "primary_endpoint": "worsening HF or CV death", "start_year": 2017},
    {"trial_name": "EMPEROR-Reduced", "condition": "heart failure", "intervention": "empagliflozin", "drug_class": "SGLT2", "phase": "3", "n_participants": 3730, "region": "global", "status": "completed", "primary_endpoint": "CV death or HF hospitalization", "start_year": 2017},
    {"trial_name": "EMPEROR-Preserved", "condition": "heart failure", "intervention": "empagliflozin", "drug_class": "SGLT2", "phase": "3", "n_participants": 5988, "region": "global", "status": "completed", "primary_endpoint": "CV death or HF hospitalization", "start_year": 2017},
    {"trial_name": "DELIVER", "condition": "heart failure", "intervention": "dapagliflozin", "drug_class": "SGLT2", "phase": "3", "n_participants": 6263, "region": "global", "status": "completed", "primary_endpoint": "worsening HF or CV death", "start_year": 2018},
    {"trial_name": "LEADER", "condition": "type 2 diabetes", "intervention": "liraglutide", "drug_class": "GLP-1", "phase": "3", "n_participants": 9340, "region": "global", "status": "completed", "primary_endpoint": "3-point MACE", "start_year": 2010},
    {"trial_name": "SUSTAIN-6", "condition": "type 2 diabetes", "intervention": "semaglutide", "drug_class": "GLP-1", "phase": "3", "n_participants": 3297, "region": "global", "status": "completed", "primary_endpoint": "3-point MACE", "start_year": 2013},
    {"trial_name": "REWIND", "condition": "type 2 diabetes", "intervention": "dulaglutide", "drug_class": "GLP-1", "phase": "3", "n_participants": 9901, "region": "global", "status": "completed", "primary_endpoint": "3-point MACE", "start_year": 2011},
    {"trial_name": "PIONEER-6", "condition": "type 2 diabetes", "intervention": "oral semaglutide", "drug_class": "GLP-1", "phase": "3", "n_participants": 3183, "region": "global", "status": "completed", "primary_endpoint": "3-point MACE", "start_year": 2017},
    {"trial_name": "SELECT", "condition": "overweight or obesity", "intervention": "semaglutide", "drug_class": "GLP-1", "phase": "3", "n_participants": 17604, "region": "global", "status": "completed", "primary_endpoint": "3-point MACE", "start_year": 2018},
    {"trial_name": "FLOW", "condition": "type 2 diabetes", "intervention": "semaglutide", "drug_class": "GLP-1", "phase": "3", "n_participants": 3534, "region": "global", "status": "completed", "primary_endpoint": "kidney composite", "start_year": 2019},
    {"trial_name": "SOUL", "condition": "type 2 diabetes", "intervention": "oral semaglutide", "drug_class": "GLP-1", "phase": "3", "n_participants": 9650, "region": "global", "status": "completed", "primary_endpoint": "3-point MACE", "start_year": 2019},
    {"trial_name": "SURPASS-CVOT", "condition": "type 2 diabetes", "intervention": "tirzepatide", "drug_class": "GIP/GLP-1", "phase": "3", "n_participants": 13299, "region": "global", "status": "ongoing", "primary_endpoint": "3-point MACE", "start_year": 2020},
    {"trial_name": "UKPDS", "condition": "type 2 diabetes", "intervention": "metformin", "drug_class": "biguanide", "phase": "3", "n_participants": 5102, "region": "UK", "status": "completed", "primary_endpoint": "diabetes complications", "start_year": 1977},
    {"trial_name": "HARMONY Outcomes", "condition": "type 2 diabetes", "intervention": "albiglutide", "drug_class": "GLP-1", "phase": "3", "n_participants": 9463, "region": "global", "status": "completed", "primary_endpoint": "3-point MACE", "start_year": 2015},
    {"trial_name": "EXSCEL", "condition": "type 2 diabetes", "intervention": "exenatide", "drug_class": "GLP-1", "phase": "3", "n_participants": 14752, "region": "global", "status": "completed", "primary_endpoint": "3-point MACE", "start_year": 2010},
]


def seed_trials(session: Session) -> int:
    count = session.scalar(select(func.count()).select_from(ClinicalTrial)) or 0
    if count:
        return 0
    for row in ROWS:
        session.add(ClinicalTrial(**row))
    session.commit()
    return len(ROWS)


async def seed_trials_async(session: AsyncSession) -> int:
    count = await session.scalar(select(func.count()).select_from(ClinicalTrial)) or 0
    if count:
        return 0
    for row in ROWS:
        session.add(ClinicalTrial(**row))
    await session.commit()
    return len(ROWS)
