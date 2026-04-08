"""
AuditAgentEnv — Pydantic Models
Explainable AI Audit Simulation with Decision Tracking
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class ActionType(str, Enum):
    EXTRACT_DATA = "extract_data"
    CHECK_MISSING_FIELDS = "check_missing_fields"
    COMPARE_FIELDS = "compare_fields"
    FLAG_RISK = "flag_risk"
    GENERATE_REPORT = "generate_report"
    NOOP = "noop"


class Phase(str, Enum):
    EXTRACTION = "extraction"
    ANALYSIS = "analysis"
    COMPLETED = "completed"


class Severity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IssueStatus(str, Enum):
    CONFIRMED = "confirmed"
    INCORRECT = "incorrect"
    PARTIAL = "partial"


class TaskName(str, Enum):
    MISSING_FIELD_DETECTION = "missing_field_detection"
    MISMATCH_DETECTION = "mismatch_detection"
    RISK_ANALYSIS = "risk_analysis"


class ExtractDataParams(BaseModel):
    source: str = Field(..., description="Document to extract: invoice or ledger")
    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("invoice", "ledger"):
            raise ValueError("source must be 'invoice' or 'ledger'")
        return v


class CheckMissingFieldsParams(BaseModel):
    source: str = Field(..., description="Which document to check")
    fields: List[str] = Field(..., min_length=1, description="Fields believed missing")
    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("invoice", "ledger"):
            raise ValueError("source must be 'invoice' or 'ledger'")
        return v


class CompareFieldsParams(BaseModel):
    invoice_field: str = Field(..., description="Field path in invoice")
    ledger_field: str = Field(..., description="Field path in ledger")
    discrepancy: Union[float, str] = Field(..., description="The discrepancy found")


class FlagRiskParams(BaseModel):
    risk_type: str = Field(..., description="Type of risk")
    details: str = Field(..., min_length=5, description="Explanation")
    severity: Severity = Field(..., description="Risk severity")


class GenerateReportParams(BaseModel):
    findings: List[str] = Field(..., description="All findings")
    risk_level: Severity = Field(..., description="Overall risk level")


class NoopParams(BaseModel):
    pass


class AuditAction(BaseModel):
    action: ActionType = Field(..., description="Action to perform")
    params: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")
    def get_typed_params(self) -> BaseModel:
        param_map = {
            ActionType.EXTRACT_DATA: ExtractDataParams,
            ActionType.CHECK_MISSING_FIELDS: CheckMissingFieldsParams,
            ActionType.COMPARE_FIELDS: CompareFieldsParams,
            ActionType.FLAG_RISK: FlagRiskParams,
            ActionType.GENERATE_REPORT: GenerateReportParams,
            ActionType.NOOP: NoopParams,
        }
        return param_map[self.action](**self.params)


class LineItem(BaseModel):
    item_id: str
    description: str
    quantity: Union[int, float]
    unit_price: Union[float, str]
    total: float
    tax_rate: float
    tax_amount: float


class Invoice(BaseModel):
    invoice_id: str
    vendor_name: str
    vendor_gstin: Optional[str] = None
    buyer_name: str
    buyer_gstin: Optional[str] = None
    invoice_date: str
    due_date: Optional[str] = None
    line_items: List[LineItem]
    subtotal: Union[float, str]
    total_tax: float
    grand_total: float
    payment_terms: Optional[str] = None
    approval_threshold: Optional[float] = None


class LedgerEntry(BaseModel):
    entry_id: str
    date: str
    ref: str
    vendor: str
    desc: str
    debit: float
    credit: float
    tax: float
    total: float


class Ledger(BaseModel):
    entries: List[LedgerEntry]


# ── UPGRADED: Explainable issue with reason + confidence ──
class IdentifiedIssue(BaseModel):
    type: str
    field: Optional[str] = None
    details: Optional[str] = None
    reason: str = Field(..., description="Why this issue was flagged — human-readable explanation")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0.0 to 1.0")
    status: IssueStatus
    correct: bool


class AuditObservation(BaseModel):
    phase: Phase
    available_documents: List[str] = Field(default_factory=lambda: ["invoice", "ledger"])
    extracted: List[str] = Field(default_factory=list)
    documents: Dict[str, Any] = Field(default_factory=dict)
    identified_issues: List[IdentifiedIssue] = Field(default_factory=list)
    action_history: List[str] = Field(default_factory=list)
    steps_taken: int = Field(default=0, ge=0)
    max_steps: int = Field(..., gt=0)
    current_task: TaskName
    done: bool = Field(default=False)
    error: Optional[str] = None


class AuditReward(BaseModel):
    value: float
    reason: str


class StepResult(BaseModel):
    observation: AuditObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class GroundTruthIssue(BaseModel):
    type: str
    field: Optional[str] = None
    details: str
    severity: Optional[Severity] = None


class TaskConfig(BaseModel):
    task_name: TaskName
    description: str
    difficulty: str
    max_steps: int = Field(..., gt=0)
    ground_truth: List[GroundTruthIssue]
    expected_risk_level: Severity
    scoring_weights: Dict[str, float]


class ResetRequest(BaseModel):
    task: TaskName = Field(default=TaskName.MISSING_FIELD_DETECTION)


# ── UPGRADED: Report with summary ──
class AuditReport(BaseModel):
    findings: List[str]
    risk_level: Severity
    summary: str = Field(default="", description="Human-readable audit summary")
