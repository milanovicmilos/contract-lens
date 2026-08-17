"""
Keyword / regex baseline classifier for ContractLens benchmarking.

Implements ``IClassifier`` using hand-crafted regular expressions, one
pattern set per CUAD category.  This is the "lower bound" baseline: a
simple heuristic that any rule-based system can achieve without any ML.

Design rules:
- Patterns are written to be *conservative*: prefer fewer false positives
  over more recall (mirrors how the tuned-threshold DeBERTa model is
  evaluated).
- The output follows the same ``Dict[category, float]`` contract as
  HFClassifier — 1.0 for a hit, 0.0 for a miss — so the orchestrator
  can use it interchangeably with the neural model.
- Patterns are intentionally not cherry-picked; they use only terminology
  that unambiguously signals the category concept.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from src.application.interfaces.iclassifier import IClassifier

# Each entry: (category_name, list_of_regex_patterns)
# A category is triggered when *any* pattern matches (logical OR).
# Patterns are compiled case-insensitively.
_CATEGORY_PATTERNS: List[Tuple[str, List[str]]] = [
    (
        "Document Name",
        [
            r"\bthis\s+(?:agreement|contract|license|arrangement)\b",
            r"\b(?:supply|service|distribution|partnership|license)\s+agreement\b",
            r"^(?:master\s+)?(?:software|services?|license)\s+agreement",
        ],
    ),
    (
        "Parties",
        [
            r"\bby\s+and\s+between\b",
            r"\bhereinafter\s+referred\s+to\s+as\b",
            r"\((?:the\s+)?[\"']?(?:company|licensor|licensee|vendor|supplier|customer|client)[\"']?\)",
            r"\bparties?\s+to\s+this\s+agreement\b",
        ],
    ),
    (
        "Agreement Date",
        [
            r"\bas\s+of\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\b",
            r"\bentered\s+into\s+(?:as\s+of\s+)?\w+\s+\d{1,2},?\s+\d{4}\b",
            r"\bdated\s+(?:as\s+of\s+)?\w+\s+\d{1,2},?\s+\d{4}\b",
            r"\beffective\s+(?:date\s+)?(?:january|february|march|april|may|june|july|august|september|october|november|december)",
        ],
    ),
    (
        "Effective Date",
        [
            r"\beffective\s+date\b",
            r"\bcomes?\s+into\s+(?:effect|force)\b",
            r"\beffective\s+upon\b",
            r"\bdate\s+(?:this\s+agreement\s+)?(?:becomes?\s+)?effective\b",
        ],
    ),
    (
        "Expiration Date",
        [
            r"\bexpir(?:es?|ation)\s+date\b",
            r"\bterm\s+(?:shall\s+)?(?:end|expire|conclude)\b",
            r"\binitial\s+term\s+(?:of|shall)\b",
            r"\bterm\s+of\s+(?:this\s+)?(?:agreement|contract)\s+(?:shall\s+)?(?:be|expire|end)\b",
        ],
    ),
    (
        "Renewal Term",
        [
            r"\bautomatically\s+renew\b",
            r"\brenewal\s+term\b",
            r"\brenew(?:ed|al)\s+for\s+(?:an?\s+)?(?:additional|successive|further)\b",
            r"\bsuccess(?:ive|ful)\s+(?:one-year|annual|renewal)\s+term\b",
        ],
    ),
    (
        "Notice Period To Terminate Renewal",
        [
            r"\bwritten\s+notice\b.{0,100}\bterminat\b.{0,100}\brenew\b",
            r"\bdays['']?\s+(?:prior\s+)?(?:written\s+)?notice.{0,80}(?:not\s+)?renew\b",
            r"\bnotice\s+(?:of\s+)?non-?renewal\b",
            r"\bnotify.{0,80}(?:not\s+)?(?:intend(?:s)?\s+to\s+)?renew\b",
        ],
    ),
    (
        "Governing Law",
        [
            r"\bgoverned\s+by\b",
            r"\blaws?\s+of\s+the\s+(?:state|province|country)\s+of\b",
            r"\bjurisdiction\s+(?:of|shall\s+be)\b",
            r"\bchoice\s+of\s+law\b",
            r"\bapplicable\s+law\s+(?:shall\s+be|is|will\s+be)\b",
        ],
    ),
    (
        "Most Favored Nation",
        [
            r"\bmost[- ]favou?red\s+nation\b",
            r"\bMFN\b",
            r"\bmost[- ]favou?red\s+(?:customer|pricing|terms)\b",
        ],
    ),
    (
        "Competitive Restriction Exception",
        [
            r"\bexception\s+to\b.{0,60}\bcompetit\b",
            r"\bcompetitive\s+restriction\s+exception\b",
            r"\bnotwithstanding\b.{0,80}\bnon[- ]compet\b",
            r"\bpermitted\s+compet\b",
        ],
    ),
    (
        "Non-Compete",
        [
            r"\bnon[- ]compet\b",
            r"\bnot\s+(?:to\s+)?(?:directly\s+or\s+indirectly\s+)?(?:engage|compete|participate)\b.{0,80}\bcompet",
            r"\bshall\s+not\s+(?:own|operate|work\s+for)\b.{0,60}\bcompet",
            r"\brestrictive\s+covenant\b",
        ],
    ),
    (
        "Exclusivity",
        [
            r"\bexclusive\s+(?:right|license|supplier|vendor|distributor|provider|territory)\b",
            r"\bexclusivity\b",
            r"\bsole\s+and\s+exclusive\b",
            r"\bexclusive\s+(?:basis|arrangement)\b",
        ],
    ),
    (
        "No-Solicit Of Customers",
        [
            r"\bsolicit\b.{0,80}\bcustomer\b",
            r"\bno[- ]solicit\b.{0,60}\bcustomer\b",
            r"\bnot\s+(?:to\s+)?solicit\b.{0,80}\bclient\b",
            r"\bsolicitation\s+of\s+(?:any\s+)?(?:customer|client)\b",
        ],
    ),
    (
        "No-Solicit Of Employees",
        [
            r"\bsolicit\b.{0,80}\bemployee\b",
            r"\bno[- ]solicit\b.{0,60}\bemployee\b",
            r"\bnot\s+(?:to\s+)?(?:hire|solicit|recruit)\b.{0,80}\bemployee\b",
            r"\bsolicitation\s+of\s+(?:any\s+)?employee\b",
        ],
    ),
    (
        "Non-Disparagement",
        [
            r"\bdisparage\b",
            r"\bnon[- ]disparagement\b",
            r"\bnegative\s+(?:statement|comment|remark)\b.{0,60}\bother\s+party\b",
            r"\bnot\s+(?:to\s+)?make\b.{0,80}\bnegative\b.{0,60}\bstatement\b",
        ],
    ),
    (
        "Termination For Convenience",
        [
            r"\bterminate\b.{0,60}\bfor\s+convenience\b",
            r"\bterminat(?:e|ion)\s+for\s+convenience\b",
            r"\bwithout\s+cause\b.{0,60}\bterminate\b",
            r"\bterminate\b.{0,60}\bwithout\s+cause\b",
        ],
    ),
    (
        "Rofr/Rofo/Rofn",
        [
            r"\bright\s+of\s+first\s+refusal\b",
            r"\bright\s+of\s+first\s+offer\b",
            r"\bright\s+of\s+first\s+negotiation\b",
            r"\bROFR\b",
            r"\bROFO\b",
            r"\bROFN\b",
        ],
    ),
    (
        "Change Of Control",
        [
            r"\bchange\s+of\s+control\b",
            r"\bchange\s+in\s+control\b",
            r"\bacquisition\b.{0,80}\b(?:terminat|assign|consent)\b",
            r"\bmerger\b.{0,80}\b(?:terminat|assign|consent)\b",
        ],
    ),
    (
        "Anti-Assignment",
        [
            r"\bnot\s+(?:be\s+)?(?:permitted\s+to\s+)?assign\b.{0,80}\bwithout\b.{0,60}\bconsent\b",
            r"\bshall\s+not\s+assign\b",
            r"\banti[- ]assignment\b",
            r"\bno\s+assignment\b",
            r"\bnot\s+transfer(?:rable|able)?\b.{0,60}\bright\b",
            r"\bassign\b.{0,100}\bwithout\b.{0,60}\b(?:written\s+)?consent\b",
        ],
    ),
    (
        "Revenue/Profit Sharing",
        [
            r"\brevenue\s+shar(?:e|ing)\b",
            r"\bprofit\s+shar(?:e|ing)\b",
            r"\bnet\s+revenue\s+split\b",
            r"\bcommission\b.{0,60}\bpercentage\b.{0,60}\brevenue\b",
        ],
    ),
    (
        "Price Restrictions",
        [
            r"\bprice\s+restriction\b",
            r"\bminimum\s+(?:resale\s+)?price\b",
            r"\bnot\s+(?:to\s+)?(?:charge|sell|offer)\b.{0,80}\bbelow\b.{0,60}\bprice\b",
            r"\bmaximum\s+(?:retail|resale)\s+price\b",
            r"\bprice[- ]fix\b",
        ],
    ),
    (
        "Minimum Commitment",
        [
            r"\bminimum\s+(?:annual|quarterly|monthly)?\s*(?:purchase|order|quantity|volume|commit)\b",
            r"\bminimum\s+commitment\b",
            r"\bcommit(?:s|ted)?\s+to\s+purchase\b",
            r"\bguaranteed\s+(?:minimum|volume)\b",
        ],
    ),
    (
        "Volume Restriction",
        [
            r"\bvolume\s+restriction\b",
            r"\bmaximum\s+(?:volume|quantity|units)\b",
            r"\bnot\s+(?:to\s+)?(?:exceed|purchase|order)\b.{0,60}\b(?:units|volume)\b",
            r"\bpurchase\s+limit\b",
        ],
    ),
    (
        "Ip Ownership Assignment",
        [
            r"\bwork(?:s)?\s+(?:made|created)\s+for\s+hire\b",
            r"\bwork[- ]for[- ]hire\b",
            r"\bassign(?:s|ed)?\b.{0,60}\bintellectual\s+property\b",
            r"\ball\s+(?:right,?\s+title\s+and\s+)?interest\b.{0,60}\bIP\b",
            r"\bIP\s+(?:shall\s+)?vest\b.{0,80}\b(?:in|with)\s+(?:the\s+)?(?:company|licensor|client)\b",
        ],
    ),
    (
        "Joint Ip Ownership",
        [
            r"\bjointly\s+own\b",
            r"\bjoint\s+(?:IP\s+)?ownership\b",
            r"\bco[- ]own(?:ership)?\b",
            r"\bequal\s+(?:undivided\s+)?(?:interest|share)\b.{0,60}\bintellectual\s+property\b",
        ],
    ),
    (
        "License Grant",
        [
            r"\bhereby\s+grants?\b",
            r"\bgrant(?:s|ed)?\b.{0,60}\blicense\s+to\b",
            r"\bright\s+to\s+(?:use|reproduce|distribute|sublicense)\b",
            r"\blicense\s+grant\b",
            r"\bgrant(?:ing)?\s+(?:a|an)\s+(?:non-?exclusive|exclusive|limited|worldwide)\s+(?:right|license)\b",
        ],
    ),
    (
        "Non-Transferable License",
        [
            r"\bnon[- ]transferable\b",
            r"\bnot\s+transferable\b",
            r"\bmay\s+not\s+(?:be\s+)?transfer(?:red)?\b.{0,60}\blicense\b",
            r"\blicense\b.{0,60}\bnon[- ]transferable\b",
        ],
    ),
    (
        "Affiliate License-Licensor",
        [
            r"\blicensor(?:'s)?\s+affiliate\b",
            r"\baffiliate(?:s)?\s+of\s+(?:the\s+)?licensor\b",
            r"\baffiliate\b.{0,60}\blicens(?:or|e)\b.{0,60}\bright\b",
        ],
    ),
    (
        "Affiliate License-Licensee",
        [
            r"\blicensee(?:'s)?\s+affiliate\b",
            r"\baffiliate(?:s)?\s+of\s+(?:the\s+)?licensee\b",
            r"\blicensee\b.{0,60}\baffiliate\b.{0,60}\bright\b",
        ],
    ),
    (
        "Unlimited/All-You-Can-Eat-License",
        [
            r"\bunlimited\s+(?:use|access|license|seat)\b",
            r"\ball[- ]you[- ]can[- ]eat\b",
            r"\bno\s+(?:volume|seat|usage)\s+limit\b",
            r"\bunlimited\s+number\s+of\s+(?:user|seat|license)\b",
        ],
    ),
    (
        "Irrevocable Or Perpetual License",
        [
            r"\birrevocable\b",
            r"\bperpetual\s+(?:license|right)\b",
            r"\bin\s+perpetuity\b",
            r"\blicense\b.{0,60}\bperpetual\b",
        ],
    ),
    (
        "Source Code Escrow",
        [
            r"\bsource\s+code\s+escrow\b",
            r"\bescrow\b.{0,80}\bsource\s+code\b",
            r"\bcode\s+escrow\b",
            r"\bescrow\s+agent\b.{0,80}\bsource\b",
        ],
    ),
    (
        "Post-Termination Services",
        [
            r"\bpost[- ]termination\b",
            r"\bafter\b.{0,40}\btermination\b.{0,60}\bservice\b",
            r"\btransition\s+(?:service|assistance|period)\b",
            r"\bwind[- ]down\s+(?:service|period)\b",
        ],
    ),
    (
        "Audit Rights",
        [
            r"\baudit\s+right\b",
            r"\bright\s+to\s+audit\b",
            r"\binspect\b.{0,60}\bbooks?\b",
            r"\bbook(?:s)?\s+and\s+records?\b.{0,60}\bexamine\b",
            r"\bindependent\s+audit\b",
        ],
    ),
    (
        "Uncapped Liability",
        [
            r"\buncapped\b",
            r"\bnot\s+(?:be\s+)?cap(?:ped)?\b.{0,60}\bliabilit\b",
            r"\bunlimited\s+liabilit\b",
            r"\bno\s+(?:limit|cap)\b.{0,60}\bliabilit\b",
        ],
    ),
    (
        "Cap On Liability",
        [
            r"\bcap(?:ped)?\b.{0,60}\bliabilit\b",
            r"\bliabilit\b.{0,60}\bcap(?:ped|s)?\b",
            r"\bnot\s+exceed\b.{0,100}\$[\d,]+",
            r"\bmaximum\s+(?:aggregate\s+)?liabilit\b",
            r"\blimit(?:ation)?\s+of\s+liabilit\b",
            r"\baggregate\s+liabilit\b.{0,60}\bnot\s+exceed\b",
        ],
    ),
    (
        "Liquidated Damages",
        [
            r"\bliquidated\s+damage\b",
            r"\bpre[- ]determined\s+damage\b",
            r"\bagreed[- ]upon\s+damage\b",
            r"\bpenalt(?:y|ies)\b.{0,60}\bbreach\b",
        ],
    ),
    (
        "Warranty Duration",
        [
            r"\bwarrant(?:y|ies)\b.{0,60}\b(?:\d+|one|two|three|six|twelve)\b.{0,30}\b(?:year|month|day)\b",
            r"\bwarranty\s+period\b",
            r"\bwarrants\b.{0,60}\bfor\b.{0,40}\b(?:\d+|one|two|three|six|twelve)\b",
            r"\bwarranty\s+(?:term|duration)\b",
        ],
    ),
    (
        "Insurance",
        [
            r"\binsurance\b",
            r"\bliability\s+insurance\b",
            r"\bgeneral\s+(?:commercial\s+)?liability\b",
            r"\bpolicy\b.{0,60}\bcoverage\b",
            r"\binsured\b",
        ],
    ),
    (
        "Covenant Not To Sue",
        [
            r"\bcovenant\s+not\s+to\s+sue\b",
            r"\bnot\s+(?:to\s+)?(?:bring|institute|commence|file)\b.{0,80}\b(?:suit|claim|action|proceeding)\b",
            r"\brelease\b.{0,60}\bclaim\b",
            r"\bwaive\b.{0,60}\bright\s+to\s+sue\b",
        ],
    ),
    (
        "Third Party Beneficiary",
        [
            r"\bthird[- ]party\s+beneficiar\b",
            r"\bno\s+third[- ](?:party\s+)?beneficiar\b",
            r"\bintended\s+beneficiar\b",
            r"\bthird\s+parties\s+(?:are\s+)?(?:not\s+)?(?:intended|entitled)\b",
        ],
    ),
]

# Compile all patterns once at module load time.
_COMPILED: List[Tuple[str, List[re.Pattern]]] = [
    (cat, [re.compile(p, re.IGNORECASE | re.DOTALL) for p in patterns])
    for cat, patterns in _CATEGORY_PATTERNS
]

# Category name → index in the 41-category ordering (for API compatibility)
_CAT_NAMES = [cat for cat, _ in _CATEGORY_PATTERNS]


class KeywordClassifier(IClassifier):
    """
    Regex keyword baseline that implements IClassifier.

    Returns 1.0 when any pattern for a category matches the text, 0.0
    otherwise.  The binary output is used as a deterministic threshold-free
    baseline during evaluation: a prediction is positive iff score == 1.0.
    """

    def __init__(self, patterns: Optional[List[Tuple[str, List[str]]]] = None):
        """
        Args:
            patterns: Optional override for the built-in category patterns.
                      Each tuple is (category_name, list_of_regex_strings).
                      When None, the built-in CUAD-derived patterns are used.
        """
        if patterns is not None:
            self._compiled = [
                (cat, [re.compile(p, re.IGNORECASE | re.DOTALL) for p in pats])
                for cat, pats in patterns
            ]
        else:
            self._compiled = _COMPILED

    def classify(self, text: str) -> Dict[str, float]:
        if not text or not text.strip():
            return {}

        result: Dict[str, float] = {}
        for category, compiled_patterns in self._compiled:
            hit = any(pat.search(text) for pat in compiled_patterns)
            result[category] = 1.0 if hit else 0.0
        return result
