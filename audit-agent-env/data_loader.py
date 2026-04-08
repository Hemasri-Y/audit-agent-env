"""
AuditAgentEnv — Data Loader
Reads invoice.xlsx and ledger.xlsx → Pydantic models.
"""

import os
from typing import Any, Dict, Tuple

import pandas as pd

from models import Invoice, Ledger, LedgerEntry, LineItem


DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DEFAULT_INVOICE_PATH = os.path.join(DEFAULT_DATA_DIR, "invoice.xlsx")
DEFAULT_LEDGER_PATH = os.path.join(DEFAULT_DATA_DIR, "ledger.xlsx")


def load_invoice(filepath: str = DEFAULT_INVOICE_PATH) -> Invoice:
    df_meta = pd.read_excel(filepath, sheet_name="Invoice_Details")
    meta = {}
    for _, row in df_meta.iterrows():
        field = str(row["Field"]).strip()
        value = row["Value"]
        if pd.isna(value):
            value = None
        elif isinstance(value, str) and value.strip() == "":
            value = None
        meta[field] = value

    df_items = pd.read_excel(filepath, sheet_name="Line_Items")
    line_items = []
    for _, row in df_items.iterrows():
        unit_price = row["unit_price"]
        if pd.isna(unit_price):
            unit_price = 0.0
        item = LineItem(
            item_id=str(row["item_id"]).strip(),
            description=str(row["description"]).strip(),
            quantity=int(row["quantity"]) if not pd.isna(row["quantity"]) else 0,
            unit_price=unit_price,
            total=float(row["total"]) if not pd.isna(row["total"]) else 0.0,
            tax_rate=float(row["tax_rate"]) if not pd.isna(row["tax_rate"]) else 0.0,
            tax_amount=float(row["tax_amount"]) if not pd.isna(row["tax_amount"]) else 0.0,
        )
        line_items.append(item)

    return Invoice(
        invoice_id=str(meta.get("invoice_id", "")),
        vendor_name=str(meta.get("vendor_name", "")),
        vendor_gstin=meta.get("vendor_gstin"),
        buyer_name=str(meta.get("buyer_name", "")),
        buyer_gstin=meta.get("buyer_gstin"),
        invoice_date=str(meta.get("invoice_date", "")),
        due_date=str(meta.get("due_date", "")) if meta.get("due_date") else None,
        line_items=line_items,
        subtotal=meta.get("subtotal", 0.0),
        total_tax=float(meta.get("total_tax", 0.0)),
        grand_total=float(meta.get("grand_total", 0.0)),
        payment_terms=meta.get("payment_terms"),
        approval_threshold=float(meta.get("approval_threshold", 0.0)) if meta.get("approval_threshold") else None,
    )


def load_ledger(filepath: str = DEFAULT_LEDGER_PATH) -> Ledger:
    df = pd.read_excel(filepath, sheet_name="Ledger_Entries")
    entries = []
    for _, row in df.iterrows():
        entry = LedgerEntry(
            entry_id=str(row["entry_id"]).strip(),
            date=str(row["date"]).strip(),
            ref=str(row["ref"]).strip(),
            vendor=str(row["vendor"]).strip(),
            desc=str(row["desc"]).strip(),
            debit=float(row["debit"]) if not pd.isna(row["debit"]) else 0.0,
            credit=float(row["credit"]) if not pd.isna(row["credit"]) else 0.0,
            tax=float(row["tax"]) if not pd.isna(row["tax"]) else 0.0,
            total=float(row["total"]) if not pd.isna(row["total"]) else 0.0,
        )
        entries.append(entry)
    return Ledger(entries=entries)


def load_all_documents(invoice_path=DEFAULT_INVOICE_PATH, ledger_path=DEFAULT_LEDGER_PATH):
    return load_invoice(invoice_path), load_ledger(ledger_path)


def invoice_to_dict(invoice: Invoice) -> Dict[str, Any]:
    return invoice.model_dump()


def ledger_to_dict(ledger: Ledger) -> Dict[str, Any]:
    return ledger.model_dump()
