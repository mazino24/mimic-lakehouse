"""Environment-driven configuration for every Spark job.

Nothing in this repo hard-codes a bucket, endpoint or credential: jobs read
their wiring from the environment so the exact same code runs in the local
Docker Compose stack, in CI (local filesystem, no MinIO) and against a real
S3 bucket.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(key: str, default: str) -> str:
    value = os.getenv(key)
    return default if value is None or value == "" else value


def _env_bool(key: str, default: bool) -> bool:
    return _env(key, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LakeConfig:
    """Where the lakehouse layers live."""

    # ``s3a://`` against MinIO/S3, or a plain path when running tests locally.
    root: str = field(default_factory=lambda: _env("LAKE_ROOT", "s3a://mimic-lake"))
    bronze_prefix: str = "bronze"
    silver_prefix: str = "silver"
    gold_prefix: str = "gold"
    quality_prefix: str = "_quality"

    def layer(self, layer: str, table: str) -> str:
        return f"{self.root.rstrip('/')}/{layer}/{table}"

    def bronze(self, table: str) -> str:
        return self.layer(self.bronze_prefix, table)

    def silver(self, table: str) -> str:
        return self.layer(self.silver_prefix, table)

    def gold(self, table: str) -> str:
        return self.layer(self.gold_prefix, table)

    def quality(self, table: str) -> str:
        return self.layer(self.quality_prefix, table)


@dataclass(frozen=True)
class S3Config:
    """MinIO / S3 connection details injected into the Hadoop config."""

    endpoint: str = field(default_factory=lambda: _env("S3_ENDPOINT", "http://minio:9000"))
    access_key: str = field(default_factory=lambda: _env("S3_ACCESS_KEY", "minioadmin"))
    secret_key: str = field(default_factory=lambda: _env("S3_SECRET_KEY", "minioadmin"))
    path_style_access: bool = field(default_factory=lambda: _env_bool("S3_PATH_STYLE_ACCESS", True))
    enabled: bool = field(default_factory=lambda: _env_bool("S3_ENABLED", True))


@dataclass(frozen=True)
class WarehouseConfig:
    """Postgres warehouse that gold tables are published to."""

    host: str = field(default_factory=lambda: _env("WAREHOUSE_HOST", "warehouse"))
    port: int = field(default_factory=lambda: int(_env("WAREHOUSE_PORT", "5432")))
    database: str = field(default_factory=lambda: _env("WAREHOUSE_DB", "mimic"))
    user: str = field(default_factory=lambda: _env("WAREHOUSE_USER", "mimic"))
    password: str = field(default_factory=lambda: _env("WAREHOUSE_PASSWORD", "mimic"))
    schema: str = field(default_factory=lambda: _env("WAREHOUSE_LOAD_SCHEMA", "lake"))

    @property
    def jdbc_url(self) -> str:
        return f"jdbc:postgresql://{self.host}:{self.port}/{self.database}"

    @property
    def jdbc_properties(self) -> dict[str, str]:
        return {
            "user": self.user,
            "password": self.password,
            "driver": "org.postgresql.Driver",
            "stringtype": "unspecified",
        }

    @property
    def sqlalchemy_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


@dataclass(frozen=True)
class ClinicalConfig:
    """Clinical rules that define the cohort.

    Kept in one place because these constants are the actual business logic of
    the pipeline and are asserted against in the tests.
    """

    # MIMIC-IV stores ICD codes without dots: I25.110 -> "I25110".
    # I20*  = angina pectoris (stable, unstable, unspecified)
    # I251* = atherosclerotic heart disease of native coronary artery w/ angina
    angina_icd10_prefixes: tuple[str, ...] = ("I20", "I251")
    # Any circulatory-system code disqualifies a stay from the control group:
    # heart failure / MI / arrhythmia patients share the angina lab signature,
    # so leaving them in teaches the model "cardiac vs non-cardiac" instead.
    cardiac_icd10_prefix: str = "I"
    min_age: int = 18
    # Acute markers are diagnostic at presentation -> take the FIRST in-stay
    # measurement. Everything else is averaged over the stay.
    acute_marker_patterns: tuple[str, ...] = ("troponin", "ck-mb")
    # Cardiac markers + the routine chemistry/haematology panel that is drawn
    # on virtually every admission. Superset of the seven panels used in the
    # original thesis, kept as substring patterns because MIMIC lab labels are
    # free text ("Cholesterol, LDL, Calculated").
    target_lab_patterns: tuple[str, ...] = (
        "troponin", "ck-mb", "ldl", "hdl", "cholesterol", "glucose",
        "creatinine", "hemoglobin", "hematocrit", "platelet", "potassium",
        "sodium", "bicarbonate", "urea nitrogen", "albumin",
        "white blood cells", "magnesium",
    )
    # Labs must be drawn inside the admission window to count as features.
    require_labs_within_admission: bool = True
    # Cap on how many controls are kept per case when balancing the cohort.
    controls_per_case: int = 1


LAKE = LakeConfig()
S3 = S3Config()
WAREHOUSE = WarehouseConfig()
CLINICAL = ClinicalConfig()
