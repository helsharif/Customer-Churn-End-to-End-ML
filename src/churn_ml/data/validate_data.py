"""Validation checks for the raw Telco Customer Churn dataset."""

from dataclasses import dataclass, field

import pandas as pd

from churn_ml.config import (
    CUSTOMER_ID_COLUMN,
    EXPECTED_RAW_COLUMNS,
    RAW_CATEGORICAL_LEVELS,
    RAW_NUMERIC_COLUMNS,
    TARGET_COLUMN,
)


@dataclass
class ValidationReport:
    """Structured result from data validation."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, int | float | str] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Return whether validation found blocking errors."""
        return not self.errors

    def raise_for_errors(self) -> None:
        """Raise a readable exception when blocking validation errors exist."""
        if self.errors:
            message = "\n".join(f"- {error}" for error in self.errors)
            raise ValueError(f"Raw data validation failed:\n{message}")


def validate_raw_data(df: pd.DataFrame, strict: bool = True) -> ValidationReport:
    """Validate raw churn data for schema, target, IDs, and common quality issues."""
    report = ValidationReport()
    report.diagnostics["row_count"] = len(df)
    report.diagnostics["column_count"] = len(df.columns)

    missing_columns = sorted(EXPECTED_RAW_COLUMNS.difference(df.columns))
    extra_columns = sorted(set(df.columns).difference(EXPECTED_RAW_COLUMNS))
    if missing_columns:
        report.errors.append(f"Missing required columns: {', '.join(missing_columns)}")
    if extra_columns:
        report.warnings.append(f"Unexpected extra columns: {', '.join(extra_columns)}")

    if df.empty:
        report.errors.append("Dataset is empty.")

    duplicate_rows = int(df.duplicated().sum())
    report.diagnostics["duplicate_rows"] = duplicate_rows
    if duplicate_rows:
        report.errors.append(f"Found {duplicate_rows} duplicate rows.")

    if CUSTOMER_ID_COLUMN in df.columns:
        missing_ids = int(df[CUSTOMER_ID_COLUMN].isna().sum())
        duplicate_ids = int(df[CUSTOMER_ID_COLUMN].duplicated().sum())
        report.diagnostics["missing_customer_ids"] = missing_ids
        report.diagnostics["duplicate_customer_ids"] = duplicate_ids
        if missing_ids:
            report.errors.append(f"Found {missing_ids} missing customer IDs.")
        if duplicate_ids:
            report.errors.append(f"Found {duplicate_ids} duplicate customer IDs.")

    if TARGET_COLUMN in df.columns:
        target_values = set(pd.Series(df[TARGET_COLUMN]).dropna().unique().tolist())
        invalid_targets = sorted(target_values.difference({0, 1}))
        missing_targets = int(df[TARGET_COLUMN].isna().sum())
        report.diagnostics["missing_targets"] = missing_targets
        if invalid_targets:
            report.errors.append(f"Invalid target values in {TARGET_COLUMN}: {invalid_targets}")
        if missing_targets:
            report.errors.append(f"Found {missing_targets} missing target values.")

    missing_values = int(df.isna().sum().sum())
    blank_strings = int(df.astype("string").apply(lambda col: col.str.strip().eq("").sum()).sum())
    report.diagnostics["missing_values"] = missing_values
    report.diagnostics["blank_strings"] = blank_strings
    if missing_values:
        report.warnings.append(f"Found {missing_values} missing values.")
    if blank_strings:
        report.warnings.append(f"Found {blank_strings} blank string values.")

    for column in RAW_NUMERIC_COLUMNS:
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column].replace(r"^\s*$", pd.NA, regex=True), errors="coerce")
        invalid_numeric = int(numeric.isna().sum() - df[column].replace(r"^\s*$", pd.NA, regex=True).isna().sum())
        report.diagnostics[f"invalid_numeric_{column}"] = max(invalid_numeric, 0)
        if invalid_numeric > 0:
            report.errors.append(f"Column {column} contains {invalid_numeric} invalid numeric values.")
        if column in {"Tenure Months", "Monthly Charges", "Total Charges", "CLTV"}:
            negative_count = int((numeric.dropna() < 0).sum())
            if negative_count:
                report.errors.append(f"Column {column} contains {negative_count} negative values.")

    for column, allowed_levels in RAW_CATEGORICAL_LEVELS.items():
        if column not in df.columns:
            continue
        values = set(df[column].dropna().astype(str).unique().tolist())
        unexpected = sorted(values.difference(allowed_levels))
        if unexpected:
            message = f"Column {column} has unexpected levels: {unexpected}"
            if strict:
                report.errors.append(message)
            else:
                report.warnings.append(message)

    return report
