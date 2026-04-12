"""
AuditAgentEnv — Core Environment
=================================
Explainable AI Audit Simulation with Decision Tracking.

Implements OpenEnv: reset(), step(), state()
State Machine: EXTRACTION → ANALYSIS → COMPLETED
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

from models import (
    ActionType, AuditAction, AuditObservation, AuditReport, AuditReward,
    CheckMissingFieldsParams, CompareFieldsParams, ExtractDataParams,
    FlagRiskParams, GenerateReportParams, GroundTruthIssue, IdentifiedIssue,
    Invoice, IssueStatus, Ledger, Phase, Severity, StepResult, TaskConfig, TaskName,
)


# ============================================================
# TASK CONFIGS
# ============================================================

TASK_CONFIGS: Dict[TaskName, TaskConfig] = {
    TaskName.MISSING_FIELD_DETECTION: TaskConfig(
        task_name=TaskName.MISSING_FIELD_DETECTION,
        description="Detect missing or empty fields in the invoice document",
        difficulty="easy", max_steps=8,
        ground_truth=[
            GroundTruthIssue(type="missing_field", field="buyer_gstin",
                             details="buyer_gstin is null/missing in invoice"),
        ],
        expected_risk_level=Severity.LOW,
        scoring_weights={"extraction_completed": 0.10, "correct_detection": 0.50,
                         "no_false_positives": 0.20, "report_submitted": 0.20},
    ),
    TaskName.MISMATCH_DETECTION: TaskConfig(
        task_name=TaskName.MISMATCH_DETECTION,
        description="Compare invoice and ledger to find mismatches",
        difficulty="medium", max_steps=10,
        ground_truth=[
            GroundTruthIssue(type="amount_mismatch",
                             field="line_items[1].total vs entries[1].debit",
                             details="Invoice LI-002 total 250000 vs Ledger LED-002 debit 230000"),
            GroundTruthIssue(type="vendor_name_mismatch",
                             field="vendor_name vs entries[0].vendor",
                             details="Pvt Ltd vs Pvt. Ltd"),
        ],
        expected_risk_level=Severity.MEDIUM,
        scoring_weights={"extraction_completed": 0.10, "amount_mismatch_found": 0.30,
                         "vendor_mismatch_found": 0.20, "no_false_positives": 0.15,
                         "report_submitted": 0.15, "correct_risk_level": 0.10},
    ),
    TaskName.RISK_ANALYSIS: TaskConfig(
        task_name=TaskName.RISK_ANALYSIS,
        description="Detect calculation errors, tax mismatches, threshold breaches, subtotal inconsistencies",
        difficulty="hard", max_steps=15,
        ground_truth=[
            GroundTruthIssue(type="calculation_error", field="line_items[1]",
                             details="qty(20) x price(12000) = 240000 but total shows 250000",
                             severity=Severity.HIGH),
            GroundTruthIssue(type="tax_mismatch", field="line_items[1].tax_amount",
                             details="tax 43200 = 18% of 240000, not 250000",
                             severity=Severity.MEDIUM),
            GroundTruthIssue(type="threshold_breach", field="grand_total vs approval_threshold",
                             details="grand_total 548700 exceeds threshold 500000",
                             severity=Severity.HIGH),
            GroundTruthIssue(type="subtotal_inconsistency", field="subtotal",
                             details="Stated subtotal 475000 vs correct sum 465000",
                             severity=Severity.MEDIUM),
        ],
        expected_risk_level=Severity.HIGH,
        scoring_weights={"extraction_completed": 0.05, "calculation_error_found": 0.25,
                         "tax_mismatch_found": 0.20, "threshold_breach_found": 0.15,
                         "subtotal_inconsistency_found": 0.15, "no_false_positives": 0.05,
                         "report_submitted": 0.10, "correct_risk_level": 0.05},
    ),
}


# ============================================================
# REWARD VALUES
# ============================================================

class RewardValues:
    EXTRACT_VALID = 0.05
    EXTRACT_DUPLICATE = -0.1
    CORRECT_DETECTION = 0.3
    PARTIAL_DETECTION = 0.1
    FALSE_POSITIVE = -0.2
    CORRECT_RISK_CORRECT_SEVERITY = 0.3
    CORRECT_RISK_WRONG_SEVERITY = 0.15
    WRONG_RISK = -0.2
    REPORT_WITH_FINDINGS = 0.5
    REPORT_EMPTY = 0.1
    NOOP = -0.1
    INVALID_ACTION = -0.3
    WRONG_PHASE = -0.3


# ============================================================
# CONFIDENCE SCORES (rule-based, simple)
# ============================================================

CONFIDENCE = {
    "missing_field_confirmed": 0.95,
    "missing_field_rejected": 0.90,
    "amount_mismatch": 0.90,
    "vendor_mismatch": 0.80,
    "comparison_rejected": 0.85,
    "risk_confirmed_exact": 0.95,
    "risk_confirmed_partial": 0.75,
    "risk_rejected": 0.80,
}


# ============================================================
# THE ENVIRONMENT
# ============================================================

class AuditEnv:

    def __init__(self, invoice: Invoice, ledger: Ledger):
        self._invoice = invoice
        self._ledger = ledger
        self._task_config: Optional[TaskConfig] = None
        self._phase: Phase = Phase.EXTRACTION
        self._extracted: List[str] = []
        self._documents: Dict[str, Any] = {}
        self._identified_issues: List[IdentifiedIssue] = []
        self._action_history: List[str] = []
        self._steps_taken: int = 0
        self._done: bool = False
        self._error: Optional[str] = None
        self._report: Optional[AuditReport] = None
        self._rewards: List[float] = []
        self._initialized: bool = False

    # ── reset() ──
    def reset(self, task_name: TaskName = TaskName.MISSING_FIELD_DETECTION) -> AuditObservation:
        self._task_config = TASK_CONFIGS[task_name]
        self._phase = Phase.EXTRACTION
        self._extracted = []
        self._documents = {}
        self._identified_issues = []
        self._action_history = []
        self._steps_taken = 0
        self._done = False
        self._error = None
        self._report = None
        self._rewards = []
        self._initialized = True
        return self._build_observation()

    # ── step() ──
    def step(self, action: AuditAction) -> StepResult:
        if not self._initialized:
            raise RuntimeError("Call reset() first.")
        if self._done:
            raise RuntimeError("Episode done. Call reset().")

        self._steps_taken += 1
        self._error = None
        reward, reason = self._process_action(action)
        self._rewards.append(reward)

        if not self._done and self._steps_taken >= self._task_config.max_steps:
            self._done = True
            self._phase = Phase.COMPLETED

        observation = self._build_observation()
        info = {"reward_reason": reason, "step": self._steps_taken,
                "cumulative_reward": sum(self._rewards)}
        if self._done:
            info["grader_score"] = self.grade()
        return StepResult(observation=observation, reward=reward, done=self._done, info=info)

    # ── state() ──
    def state(self) -> AuditObservation:
        if not self._initialized:
            raise RuntimeError("Call reset() first.")
        return self._build_observation()

    # ── grade() ──
    def grade(self) -> float:
        if self._task_config is None:
            return 0.0
        t = self._task_config.task_name
        if t == TaskName.MISSING_FIELD_DETECTION: return self._grade_task1()
        if t == TaskName.MISMATCH_DETECTION: return self._grade_task2()
        if t == TaskName.RISK_ANALYSIS: return self._grade_task3()
        return 0.0

    # ============================================================
    # ACTION PROCESSING
    # ============================================================

    def _process_action(self, action: AuditAction) -> Tuple[float, str]:
        action_type = action.action

        if action_type == ActionType.NOOP:
            self._action_history.append(f"Step {self._steps_taken} → noop (skipped)")
            return RewardValues.NOOP, "noop"

        try:
            typed_params = action.get_typed_params()
        except Exception as e:
            self._error = f"Invalid parameters: {str(e)}"
            self._action_history.append(f"Step {self._steps_taken} → {action_type.value} → ERROR: {self._error}")
            return RewardValues.INVALID_ACTION, self._error

        handler_map = {
            ActionType.EXTRACT_DATA: self._handle_extract_data,
            ActionType.CHECK_MISSING_FIELDS: self._handle_check_missing_fields,
            ActionType.COMPARE_FIELDS: self._handle_compare_fields,
            ActionType.FLAG_RISK: self._handle_flag_risk,
            ActionType.GENERATE_REPORT: self._handle_generate_report,
        }
        handler = handler_map.get(action_type)
        if handler is None:
            self._error = f"Unknown action: {action_type}"
            self._action_history.append(f"Step {self._steps_taken} → unknown action")
            return RewardValues.INVALID_ACTION, self._error
        return handler(typed_params)

    # ── extract_data ──
    def _handle_extract_data(self, params: ExtractDataParams) -> Tuple[float, str]:
        source = params.source
        if source in self._extracted:
            self._error = f"Already extracted: {source}"
            self._action_history.append(f"Step {self._steps_taken} → extract_data({source}) → DUPLICATE")
            return RewardValues.EXTRACT_DUPLICATE, self._error

        if source == "invoice":
            self._documents["invoice"] = self._invoice.model_dump()
        elif source == "ledger":
            self._documents["ledger"] = self._ledger.model_dump()

        self._extracted.append(source)
        if set(self._extracted) >= {"invoice", "ledger"}:
            self._phase = Phase.ANALYSIS

        self._action_history.append(f"Step {self._steps_taken} → extract_data({source}) → loaded")
        return RewardValues.EXTRACT_VALID, f"Extracted {source}"

    # ── check_missing_fields ──
    def _handle_check_missing_fields(self, params: CheckMissingFieldsParams) -> Tuple[float, str]:
        if self._phase == Phase.EXTRACTION:
            self._error = "Must extract both documents before analysis"
            self._action_history.append(f"Step {self._steps_taken} → check_missing_fields → BLOCKED (extraction incomplete)")
            return RewardValues.WRONG_PHASE, self._error

        actual_missing = self._get_missing_fields(params.source)
        total_reward = 0.0
        reasons = []

        for field in params.fields:
            if field in actual_missing:
                self._identified_issues.append(IdentifiedIssue(
                    type="missing_field", field=field,
                    details=f"{field} is missing in {params.source}",
                    reason=f"Field '{field}' is null or empty in the {params.source} document",
                    confidence=CONFIDENCE["missing_field_confirmed"],
                    status=IssueStatus.CONFIRMED, correct=True))
                total_reward += RewardValues.CORRECT_DETECTION
                reasons.append(f"confirmed: {field}")
                self._action_history.append(f"Step {self._steps_taken} → check_missing_fields → found {field} missing ✓")
            else:
                self._identified_issues.append(IdentifiedIssue(
                    type="missing_field", field=field,
                    details=f"{field} claimed missing but exists",
                    reason=f"Field '{field}' exists and has a value in {params.source}",
                    confidence=CONFIDENCE["missing_field_rejected"],
                    status=IssueStatus.INCORRECT, correct=False))
                total_reward += RewardValues.FALSE_POSITIVE
                reasons.append(f"false positive: {field}")
                self._action_history.append(f"Step {self._steps_taken} → check_missing_fields → {field} NOT missing ✗")

        return total_reward, "; ".join(reasons)

    # ── compare_fields ──
    def _handle_compare_fields(self, params: CompareFieldsParams) -> Tuple[float, str]:
        if self._phase == Phase.EXTRACTION:
            self._error = "Must extract both documents before analysis"
            self._action_history.append(f"Step {self._steps_taken} → compare_fields → BLOCKED")
            return RewardValues.WRONG_PHASE, self._error

        is_valid, issue_type, details = self._validate_comparison(params)

        if is_valid:
            conf_key = (
                "vendor_mismatch"
                if issue_type == "vendor_name_mismatch"
                else issue_type
            )
            conf = CONFIDENCE.get(conf_key, 0.85)
            self._identified_issues.append(IdentifiedIssue(
                type=issue_type,
                field=f"{params.invoice_field} vs {params.ledger_field}",
                details=details,
                reason=f"Mismatch detected: invoice field and ledger field have different values",
                confidence=conf,
                status=IssueStatus.CONFIRMED, correct=True))
            self._action_history.append(f"Step {self._steps_taken} → compare_fields → {issue_type} confirmed ✓")
            return RewardValues.CORRECT_DETECTION, f"correct: {details}"
        else:
            self._identified_issues.append(IdentifiedIssue(
                type="claimed_mismatch",
                field=f"{params.invoice_field} vs {params.ledger_field}",
                details="No real mismatch",
                reason="Comparison did not reveal an actual discrepancy between the documents",
                confidence=CONFIDENCE["comparison_rejected"],
                status=IssueStatus.INCORRECT, correct=False))
            self._action_history.append(f"Step {self._steps_taken} → compare_fields → no mismatch found ✗")
            return RewardValues.FALSE_POSITIVE, "false positive comparison"

    # ── flag_risk ──
    def _handle_flag_risk(self, params: FlagRiskParams) -> Tuple[float, str]:
        if self._phase == Phase.EXTRACTION:
            self._error = "Must extract both documents before analysis"
            self._action_history.append(f"Step {self._steps_taken} → flag_risk → BLOCKED")
            return RewardValues.WRONG_PHASE, self._error

        is_valid, correct_severity = self._validate_risk(params)

        if is_valid and params.severity == correct_severity:
            self._identified_issues.append(IdentifiedIssue(
                type=params.risk_type, field=None, details=params.details,
                reason=f"Risk '{params.risk_type}' verified against document data with matching severity",
                confidence=CONFIDENCE["risk_confirmed_exact"],
                status=IssueStatus.CONFIRMED, correct=True))
            self._action_history.append(f"Step {self._steps_taken} → flag_risk({params.risk_type}, {params.severity.value}) → confirmed ✓")
            return RewardValues.CORRECT_RISK_CORRECT_SEVERITY, f"correct: {params.risk_type}"
        elif is_valid:
            self._identified_issues.append(IdentifiedIssue(
                type=params.risk_type, field=None, details=params.details,
                reason=f"Risk '{params.risk_type}' is valid but severity should be {correct_severity.value}",
                confidence=CONFIDENCE["risk_confirmed_partial"],
                status=IssueStatus.PARTIAL, correct=True))
            self._action_history.append(f"Step {self._steps_taken} → flag_risk({params.risk_type}) → partial (wrong severity) ~")
            return RewardValues.CORRECT_RISK_WRONG_SEVERITY, f"partial: {params.risk_type}"
        else:
            self._identified_issues.append(IdentifiedIssue(
                type=params.risk_type, field=None, details=params.details,
                reason=f"Claimed risk '{params.risk_type}' not supported by document evidence",
                confidence=CONFIDENCE["risk_rejected"],
                status=IssueStatus.INCORRECT, correct=False))
            self._action_history.append(f"Step {self._steps_taken} → flag_risk({params.risk_type}) → not valid ✗")
            return RewardValues.WRONG_RISK, f"incorrect: {params.risk_type}"

    # ── generate_report ──
    def _handle_generate_report(self, params: GenerateReportParams) -> Tuple[float, str]:
        # Auto-generate summary from findings
        confirmed = [i for i in self._identified_issues if i.correct]
        if confirmed:
            issue_names = [i.type.replace("_", " ") for i in confirmed]
            summary = f"Audit found {len(confirmed)} issue(s): {', '.join(issue_names)}. Risk level: {params.risk_level.value}."
        else:
            summary = "Audit completed. No significant issues detected."

        self._report = AuditReport(
            findings=params.findings,
            risk_level=params.risk_level,
            summary=summary,
        )
        self._done = True
        self._phase = Phase.COMPLETED

        self._action_history.append(f"Step {self._steps_taken} → generate_report → {len(params.findings)} findings, risk={params.risk_level.value}")

        if len(params.findings) > 0:
            return RewardValues.REPORT_WITH_FINDINGS, f"Report: {len(params.findings)} findings"
        return RewardValues.REPORT_EMPTY, "Report with no findings"

    # ============================================================
    # VALIDATION HELPERS (unchanged logic)
    # ============================================================

    def _get_missing_fields(self, source: str) -> List[str]:
        missing = []
        if source == "invoice":
            inv = self._invoice
            check = {
                "invoice_id": inv.invoice_id, "vendor_name": inv.vendor_name,
                "vendor_gstin": inv.vendor_gstin, "buyer_name": inv.buyer_name,
                "buyer_gstin": inv.buyer_gstin, "invoice_date": inv.invoice_date,
                "due_date": inv.due_date, "payment_terms": inv.payment_terms,
                "approval_threshold": inv.approval_threshold,
            }
            for name, value in check.items():
                if value is None or (isinstance(value, str) and value.strip() == ""):
                    missing.append(name)
        elif source == "ledger":
            for i, entry in enumerate(self._ledger.entries):
                for name in ["entry_id", "date", "ref", "vendor", "desc"]:
                    val = getattr(entry, name, None)
                    if val is None or (isinstance(val, str) and val.strip() == ""):
                        missing.append(f"entries[{i}].{name}")
        return missing

    def _validate_comparison(self, params: CompareFieldsParams) -> Tuple[bool, str, str]:
        inv_field = params.invoice_field.lower().strip()
        led_field = params.ledger_field.lower().strip()

        amount_inv = ["line_items[1].total", "li-002", "total", "line_items[1]", "250000"]
        amount_led = ["entries[1].debit", "led-002", "debit", "entries[1]", "230000"]
        if any(kw in inv_field for kw in amount_inv) and any(kw in led_field for kw in amount_led):
            return True, "amount_mismatch", "Invoice LI-002 total (250000) vs Ledger LED-002 debit (230000)"

        if isinstance(params.discrepancy, (int, float)):
            if abs(params.discrepancy) in [20000, 20000.0]:
                if ("total" in inv_field or "line" in inv_field) and ("debit" in led_field or "entr" in led_field):
                    return True, "amount_mismatch", "Amount mismatch of 20000"

        vendor_kw = ["vendor", "name", "pvt", "supplier"]
        if any(kw in inv_field for kw in vendor_kw) and any(kw in led_field for kw in vendor_kw):
            return True, "vendor_name_mismatch", "Vendor name: Pvt Ltd vs Pvt. Ltd"

        if isinstance(params.discrepancy, str):
            d = params.discrepancy.lower()
            if any(kw in d for kw in ["vendor", "name", "mismatch", "pvt", "dot"]):
                return True, "vendor_name_mismatch", "Vendor name mismatch detected"

        return False, "", ""

    def _validate_risk(self, params: FlagRiskParams) -> Tuple[bool, Optional[Severity]]:
        risk_type = params.risk_type.lower().strip()
        details_lower = params.details.lower()

        if any(kw in risk_type for kw in ["calculation", "calc", "arithmetic", "math"]):
            return True, Severity.HIGH
        if any(kw in risk_type for kw in ["tax"]):
            return True, Severity.MEDIUM
        if any(kw in risk_type for kw in ["threshold", "breach", "approval", "exceed"]):
            return True, Severity.HIGH
        if any(kw in risk_type for kw in ["subtotal", "sub_total", "sub-total", "inconsist"]):
            return True, Severity.MEDIUM

        calc_kw = ["qty", "quantity", "unit_price", "multiply", "arithmetic", "incorrect total", "wrong total", "calculation"]
        if any(kw in details_lower for kw in calc_kw) and "total" in details_lower:
            return True, Severity.HIGH
        if "tax" in details_lower and any(kw in details_lower for kw in ["mismatch", "incorrect", "wrong", "error", "base", "18%"]):
            return True, Severity.MEDIUM
        if any(kw in details_lower for kw in ["threshold", "approval", "exceed", "breach", "limit"]):
            if any(kw in details_lower for kw in ["548700", "500000"]):
                return True, Severity.HIGH
        if any(kw in details_lower for kw in ["subtotal", "sub total", "sub-total"]):
            if any(kw in details_lower for kw in ["475000", "465000", "inconsist", "mismatch", "wrong"]):
                return True, Severity.MEDIUM

        return False, None

    # ============================================================
    # GRADING (unchanged logic)
    # ============================================================

    def _grade_task1(self) -> float:
        score = 0.0
        if set(self._extracted) >= {"invoice", "ledger"}: score += 0.10
        gt_fields = {gt.field for gt in self._task_config.ground_truth}
        found = {i.field for i in self._identified_issues if i.type == "missing_field" and i.correct}
        if len(gt_fields) > 0: score += 0.50 * (len(gt_fields & found) / len(gt_fields))
        fps = [i for i in self._identified_issues if not i.correct]
        score += 0.20 - min(len(fps) * 0.05, 0.20)
        if self._report is not None: score += 0.20
        return max(0.0, min(1.0, score))

    def _grade_task2(self) -> float:
        score = 0.0
        if set(self._extracted) >= {"invoice", "ledger"}: score += 0.10
        confirmed = {i.type for i in self._identified_issues if i.correct}
        if "amount_mismatch" in confirmed: score += 0.30
        if "vendor_name_mismatch" in confirmed: score += 0.20
        fps = [i for i in self._identified_issues if not i.correct]
        score += 0.15 - min(len(fps) * 0.05, 0.15)
        if self._report is not None:
            score += 0.15
            if self._report.risk_level == self._task_config.expected_risk_level: score += 0.10
        return max(0.0, min(1.0, score))

    def _grade_task3(self) -> float:
        score = 0.0
        if set(self._extracted) >= {"invoice", "ledger"}: score += 0.05
        confirmed = {i.type for i in self._identified_issues if i.correct}
        if "calculation_error" in confirmed: score += 0.25
        if "tax_mismatch" in confirmed: score += 0.20
        if "threshold_breach" in confirmed: score += 0.15
        if "subtotal_inconsistency" in confirmed: score += 0.15
        fps = [i for i in self._identified_issues if not i.correct]
        score += 0.05 - min(len(fps) * 0.05, 0.05)
        if self._report is not None:
            score += 0.10
            if self._report.risk_level == self._task_config.expected_risk_level: score += 0.05
        return max(0.0, min(1.0, score))

    # ── Observation builder ──
    def _build_observation(self) -> AuditObservation:
        return AuditObservation(
            phase=self._phase, available_documents=["invoice", "ledger"],
            extracted=list(self._extracted),
            documents=copy.deepcopy(self._documents),
            identified_issues=[copy.deepcopy(i) for i in self._identified_issues],
            action_history=list(self._action_history),
            steps_taken=self._steps_taken,
            max_steps=self._task_config.max_steps if self._task_config else 8,
            current_task=self._task_config.task_name if self._task_config else TaskName.MISSING_FIELD_DETECTION,
            done=self._done, error=self._error,
        )

    @property
    def rewards(self) -> List[float]: return list(self._rewards)
    @property
    def is_done(self) -> bool: return self._done
    @property
    def task_config(self) -> Optional[TaskConfig]: return self._task_config
