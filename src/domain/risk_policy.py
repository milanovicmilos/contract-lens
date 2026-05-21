"""
Risk policy definitions for CUAD clause categories.

Each policy maps a clause category to:
- default_risk: baseline risk level when the category is present (Low/Medium/High)
- high_risk_keywords: terms that escalate risk to High when detected in the clause text
- justification_template: human-readable rationale

The 41 default policies below are derived from CUAD category semantics and
common legal review practice. They are starting points — a real deployment
should tune these via legal counsel review and historical incident data.
"""

from typing import Any, Dict, Optional, Tuple

DEFAULT_POLICIES: Dict[str, Dict[str, Any]] = {
    # --- Metadata (low-risk identification) ---
    "Document Name": {
        "high_risk_keywords": [],
        "default_risk": "Low",
        "rationale": "Contract title identification; minimal risk impact.",
    },
    "Parties": {
        "high_risk_keywords": ["shell company", "subsidiary of unknown"],
        "default_risk": "Low",
        "rationale": "Counterparty identification; verify entity legitimacy.",
    },
    "Agreement Date": {
        "high_risk_keywords": [],
        "default_risk": "Low",
        "rationale": "Contract effective period marker.",
    },
    "Effective Date": {
        "high_risk_keywords": ["retroactive"],
        "default_risk": "Low",
        "rationale": "When obligations begin; retroactive effective dates increase risk.",
    },
    "Expiration Date": {
        "high_risk_keywords": ["perpetual", "no expiration"],
        "default_risk": "Low",
        "rationale": "Contract end; perpetual contracts shift to higher risk.",
    },
    # --- Renewal & Termination ---
    "Renewal Term": {
        "high_risk_keywords": ["automatic", "auto-renew", "indefinite"],
        "default_risk": "Medium",
        "rationale": "Auto-renewal exposes party to indefinite commitments.",
    },
    "Notice Period To Terminate Renewal": {
        "high_risk_keywords": ["180 days", "one year", "12 months"],
        "default_risk": "Medium",
        "rationale": "Long notice periods make termination harder.",
    },
    "Termination For Convenience": {
        "high_risk_keywords": ["no termination", "irrevocable", "cannot terminate"],
        "default_risk": "Medium",
        "rationale": "Absence of convenience termination locks parties in.",
    },
    # --- Governance & Jurisdiction ---
    "Governing Law": {
        "high_risk_keywords": ["Russia", "China", "North Korea", "Iran", "Unknown"],
        "default_risk": "Medium",
        "rationale": "Foreign sanctioned jurisdictions create enforcement risk.",
    },
    # --- Competitive Restrictions (high risk by nature) ---
    "Most Favored Nation": {
        "high_risk_keywords": ["worldwide", "unlimited", "any customer"],
        "default_risk": "High",
        "rationale": "MFN clauses constrain pricing flexibility.",
    },
    "Competitive Restriction Exception": {
        "high_risk_keywords": [],
        "default_risk": "Medium",
        "rationale": "Carve-outs to competitive restrictions; verify scope.",
    },
    "Non-Compete": {
        "high_risk_keywords": ["worldwide", "perpetual", "unlimited", "global"],
        "default_risk": "High",
        "rationale": "Non-compete clauses restrict future business opportunities.",
    },
    "Exclusivity": {
        "high_risk_keywords": ["sole", "exclusive", "worldwide"],
        "default_risk": "High",
        "rationale": "Exclusivity grants reduce market flexibility.",
    },
    "No-Solicit Of Customers": {
        "high_risk_keywords": ["worldwide", "indefinite", "all customers"],
        "default_risk": "Medium",
        "rationale": "Restrictions on customer outreach post-contract.",
    },
    "No-Solicit Of Employees": {
        "high_risk_keywords": ["worldwide", "indefinite"],
        "default_risk": "Medium",
        "rationale": "Restrictions on hiring; check duration and scope.",
    },
    "Non-Disparagement": {
        "high_risk_keywords": ["any statement", "perpetual"],
        "default_risk": "Medium",
        "rationale": "Limits public commentary; review chilling-effect concerns.",
    },
    # --- Rights of refusal & control ---
    "Rofr/Rofo/Rofn": {
        "high_risk_keywords": ["all assets", "any transaction"],
        "default_risk": "Medium",
        "rationale": "First refusal rights constrain future deals.",
    },
    "Change Of Control": {
        "high_risk_keywords": ["immediate termination", "automatic"],
        "default_risk": "High",
        "rationale": "CoC triggers can derail M&A activity.",
    },
    "Anti-Assignment": {
        "high_risk_keywords": ["no assignment", "void", "prohibited"],
        "default_risk": "Medium",
        "rationale": "Strict anti-assignment limits flexibility.",
    },
    # --- Commercial terms ---
    "Revenue/Profit Sharing": {
        "high_risk_keywords": ["all revenue", "100%", "gross"],
        "default_risk": "Medium",
        "rationale": "Verify the calculation basis and audit rights.",
    },
    "Price Restrictions": {
        "high_risk_keywords": ["cannot reduce", "minimum price"],
        "default_risk": "Medium",
        "rationale": "Pricing constraints can trigger antitrust scrutiny.",
    },
    "Minimum Commitment": {
        "high_risk_keywords": ["take-or-pay", "guaranteed"],
        "default_risk": "Medium",
        "rationale": "Minimum purchase obligations create financial exposure.",
    },
    "Volume Restriction": {
        "high_risk_keywords": ["cap", "maximum", "ceiling"],
        "default_risk": "Medium",
        "rationale": "Volume caps limit business growth.",
    },
    # --- IP & License (high stakes for tech contracts) ---
    "Ip Ownership Assignment": {
        "high_risk_keywords": ["all ip", "work for hire", "irrevocable"],
        "default_risk": "High",
        "rationale": "Permanent IP transfers; verify carve-outs for pre-existing IP.",
    },
    "Joint Ip Ownership": {
        "high_risk_keywords": ["jointly owned", "shared"],
        "default_risk": "Medium",
        "rationale": "Joint IP creates licensing and exploitation complexity.",
    },
    "License Grant": {
        "high_risk_keywords": ["irrevocable", "perpetual", "sublicensable"],
        "default_risk": "Medium",
        "rationale": "Broad license grants reduce IP control.",
    },
    "Non-Transferable License": {
        "high_risk_keywords": ["non-transferable", "personal use only"],
        "default_risk": "Low",
        "rationale": "Non-transferable licenses limit downstream value.",
    },
    "Affiliate License-Licensor": {
        "high_risk_keywords": [],
        "default_risk": "Low",
        "rationale": "Affiliate license scope; verify entity definitions.",
    },
    "Affiliate License-Licensee": {
        "high_risk_keywords": [],
        "default_risk": "Low",
        "rationale": "Affiliate license scope; verify entity definitions.",
    },
    "Unlimited/All-You-Can-Eat-License": {
        "high_risk_keywords": ["unlimited", "all-you-can-eat", "no cap"],
        "default_risk": "High",
        "rationale": "Unlimited usage clauses can produce significant revenue loss.",
    },
    "Irrevocable Or Perpetual License": {
        "high_risk_keywords": ["irrevocable", "perpetual"],
        "default_risk": "High",
        "rationale": "Irrevocable licenses permanently encumber IP.",
    },
    "Source Code Escrow": {
        "high_risk_keywords": ["release upon", "trigger"],
        "default_risk": "Medium",
        "rationale": "Escrow release triggers; review trigger conditions.",
    },
    # --- Post-termination & audit ---
    "Post-Termination Services": {
        "high_risk_keywords": ["indefinite", "transition assistance"],
        "default_risk": "Medium",
        "rationale": "Post-termination obligations extend liability.",
    },
    "Audit Rights": {
        "high_risk_keywords": ["unlimited", "any time", "without notice"],
        "default_risk": "Medium",
        "rationale": "Audit scope; check notice and frequency limits.",
    },
    # --- Liability & damages (always high risk) ---
    "Uncapped Liability": {
        "high_risk_keywords": ["uncapped", "unlimited liability", "no limit"],
        "default_risk": "High",
        "rationale": "Uncapped liability creates unbounded financial risk.",
    },
    "Cap On Liability": {
        "high_risk_keywords": ["unlimited", "uncapped"],
        "default_risk": "Medium",
        "rationale": "Verify cap is reasonable relative to contract value.",
    },
    "Liquidated Damages": {
        "high_risk_keywords": ["punitive", "treble", "exceeds actual damages"],
        "default_risk": "Medium",
        "rationale": "Liquidated damages must reflect reasonable estimate of harm.",
    },
    # --- Warranty & insurance ---
    "Warranty Duration": {
        "high_risk_keywords": ["lifetime", "perpetual"],
        "default_risk": "Medium",
        "rationale": "Long warranty periods extend liability tail.",
    },
    "Insurance": {
        "high_risk_keywords": ["no insurance required"],
        "default_risk": "Medium",
        "rationale": "Insurance requirements protect against counterparty default.",
    },
    # --- Remedies & beneficiaries ---
    "Covenant Not To Sue": {
        "high_risk_keywords": ["all claims", "perpetual"],
        "default_risk": "High",
        "rationale": "Covenants not to sue waive future remedies.",
    },
    "Third Party Beneficiary": {
        "high_risk_keywords": ["any third party", "broad class"],
        "default_risk": "Medium",
        "rationale": "Third-party beneficiary rights expand enforcement.",
    },
}


class RiskPolicy:
    """
    Defines the rules for scoring risks across CUAD clause categories.

    The default policy ships with rules for all 41 CUAD categories. Users may
    override individual categories or supply a complete custom policy dict.
    """

    def __init__(self, rules: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize the risk policy.

        Args:
            rules: A mapping of category name to its policy dict (with keys
                'high_risk_keywords', 'default_risk', optionally 'rationale').
                If None or empty, the full default 41-category policy is used.
                Partial dicts are merged on top of defaults so callers can
                override only the categories they care about.
        """
        if rules:
            merged = {**DEFAULT_POLICIES}
            merged.update(rules)
            self.rules = merged
        else:
            self.rules = dict(DEFAULT_POLICIES)

    def assess_risk(self, category: str, text: str) -> Tuple[str, float, str]:
        """
        Assess risk for a clause via keyword matching against the policy.

        Args:
            category: The clause category (e.g., "Governing Law").
            text: The extracted clause text.

        Returns:
            Tuple of (risk_level, confidence_score, justification).
        """
        if category not in self.rules:
            return (
                "Low",
                0.1,
                f"No specific risk policy defined for '{category}'; assumed low risk.",
            )

        policy = self.rules[category]
        text_lower = text.lower()

        for kw in policy.get("high_risk_keywords", []):
            if kw.lower() in text_lower:
                return (
                    "High",
                    0.9,
                    f"High-risk keyword '{kw}' detected in {category} clause. "
                    f"{policy.get('rationale', '')}".strip(),
                )

        default_risk = policy.get("default_risk", "Medium")
        score_map = {"Low": 0.3, "Medium": 0.5, "High": 0.8}
        return (
            default_risk,
            score_map.get(default_risk, 0.5),
            f"Standard {category} clause. {policy.get('rationale', '')}".strip(),
        )
