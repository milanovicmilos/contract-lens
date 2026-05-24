"""
CUAD dataset loading and preprocessing pipeline.

Loads CUAD dataset (SQuAD format), maps clauses with master_clauses.csv,
and provides utilities for creating train/val/test splits.
"""

import csv
import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ClauseCategory:
    """Represents a clause category in CUAD."""

    question_id: str  # e.g., "Q1", "Q2", etc.
    category_name: str  # e.g., "Document Name", "Parties", etc.
    question_text: str  # Full question text
    is_binary: bool  # True if Yes/No classification, False if extraction
    typical_answers: List[str] = None  # Examples: ["Yes", "No"] or ["Date", "Party name"]

    def __post_init__(self) -> None:
        if self.typical_answers is None:
            self.typical_answers = []


@dataclass
class ContractExample:
    """Single contract with CUAD annotations."""

    contract_id: str
    filename: str
    title: str
    context: str  # Full contract text
    qas: List[Dict[str, Any]]  # List of {question, id, answers, is_impossible}
    csv_row: Optional[Dict[str, str]] = None  # Corresponding CSV row


class CUADDataset:
    """
    Loader for CUAD dataset.

    Handles:
    - Loading CUAD_v1.json in SQuAD format
    - Parsing master_clauses.csv
    - Mapping between them
    - Creating train/val/test splits
    """

    # CUAD has 41 clause categories
    CLAUSE_CATEGORIES = {
        "Document Name": "Q1",
        "Parties": "Q2",
        "Agreement Date": "Q3",
        "Effective Date": "Q4",
        "Expiration Date": "Q5",
        "Renewal Term": "Q6",
        "Notice Period To Terminate Renewal": "Q7",
        "Governing Law": "Q8",
        "Most Favored Nation": "Q9",
        "Competitive Restriction Exception": "Q10",
        "Non-Compete": "Q11",
        "Exclusivity": "Q12",
        "No-Solicit Of Customers": "Q13",
        "No-Solicit Of Employees": "Q14",
        "Non-Disparagement": "Q15",
        "Termination For Convenience": "Q16",
        "Rofr/Rofo/Rofn": "Q17",
        "Change Of Control": "Q18",
        "Anti-Assignment": "Q19",
        "Revenue/Profit Sharing": "Q20",
        "Price Restrictions": "Q21",
        "Minimum Commitment": "Q22",
        "Volume Restriction": "Q23",
        "Ip Ownership Assignment": "Q24",
        "Joint Ip Ownership": "Q25",
        "License Grant": "Q26",
        "Non-Transferable License": "Q27",
        "Affiliate License-Licensor": "Q28",
        "Affiliate License-Licensee": "Q29",
        "Unlimited/All-You-Can-Eat-License": "Q30",
        "Irrevocable Or Perpetual License": "Q31",
        "Source Code Escrow": "Q32",
        "Post-Termination Services": "Q33",
        "Audit Rights": "Q34",
        "Uncapped Liability": "Q35",
        "Cap On Liability": "Q36",
        "Liquidated Damages": "Q37",
        "Warranty Duration": "Q38",
        "Insurance": "Q39",
        "Covenant Not To Sue": "Q40",
        "Third Party Beneficiary": "Q41",
    }

    def __init__(self, dataset_dir: Path = None) -> None:
        """
        Initialize dataset loader.

        Args:
            dataset_dir: Path to CUAD_v1 directory. Defaults to CUAD_v1/ in project root.
        """
        if dataset_dir is None:
            dataset_dir = Path(__file__).parent.parent.parent / "CUAD_v1"

        self.dataset_dir = dataset_dir
        self.json_path = dataset_dir / "CUAD_v1.json"
        self.csv_path = dataset_dir / "master_clauses.csv"

        logger.info(f"Initialized CUADDataset: {self.dataset_dir}")

    def load(self) -> Tuple[List[ContractExample], List[ClauseCategory]]:
        """
        Load and merge CUAD JSON and CSV data.

        Returns:
            Tuple of (contracts, categories) where:
            - contracts: List of ContractExample objects
            - categories: List of ClauseCategory objects
        """
        logger.info("Loading CUAD dataset...")

        # Load JSON (SQuAD format)
        contracts = self._load_squad_json()
        logger.info(f"✅ Loaded {len(contracts)} contracts from CUAD_v1.json")

        # Load CSV
        csv_data = self._load_master_clauses_csv()
        logger.info(f"✅ Loaded {len(csv_data)} rows from master_clauses.csv")

        # Merge JSON and CSV
        contracts = self._merge_with_csv(contracts, csv_data)

        # Extract categories
        categories = self._extract_categories(contracts)

        return contracts, categories

    def _load_squad_json(self) -> List[ContractExample]:
        """Load CUAD_v1.json in SQuAD format."""
        if not self.json_path.exists():
            raise FileNotFoundError(f"CUAD JSON not found: {self.json_path}")

        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        contracts = []
        articles = data.get("data", [])

        for article_idx, article in enumerate(articles):
            title = article.get("title", f"contract_{article_idx}")
            paragraphs = article.get("paragraphs", [])

            for para in paragraphs:
                context = para.get("context", "")
                qas = para.get("qas", [])

                # Create contract ID from title
                contract_id = self._sanitize_id(title)

                contract = ContractExample(
                    contract_id=contract_id,
                    filename=title,
                    title=title,
                    context=context,
                    qas=qas,
                )

                contracts.append(contract)

        return contracts

    def _load_master_clauses_csv(self) -> Dict[str, Dict[str, str]]:
        """
        Load master_clauses.csv and create lookup by filename.

        Returns:
            Dictionary mapping filename to CSV row.
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Master clauses CSV not found: {self.csv_path}")

        csv_data = {}

        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get("Filename", "").strip()
                if filename:
                    # Normalize filename for matching
                    normalized_filename = self._sanitize_filename(filename)
                    csv_data[normalized_filename] = row

        return csv_data

    def _merge_with_csv(
        self,
        contracts: List[ContractExample],
        csv_data: Dict[str, Dict[str, str]],
    ) -> List[ContractExample]:
        """
        Merge contract JSON with CSV data.

        Args:
            contracts: List of contracts loaded from JSON.
            csv_data: CSV data indexed by filename.

        Returns:
            Contracts with CSV data attached.
        """
        matched_count = 0

        for contract in contracts:
            # Try to find matching CSV row
            normalized_title = self._sanitize_filename(contract.title)

            # Exact match
            if normalized_title in csv_data:
                contract.csv_row = csv_data[normalized_title]
                matched_count += 1
            else:
                # Fuzzy match: try to find by partial filename
                for csv_filename, csv_row in csv_data.items():
                    if self._fuzzy_filename_match(normalized_title, csv_filename):
                        contract.csv_row = csv_row
                        matched_count += 1
                        break

        logger.info(f"Matched {matched_count}/{len(contracts)} contracts with CSV data")
        return contracts

    def _extract_categories(self, contracts: List[ContractExample]) -> List[ClauseCategory]:
        """Extract and analyze clause categories from contracts."""
        categories = []

        for category_name, q_id in self.CLAUSE_CATEGORIES.items():
            # Find examples from contracts
            examples = []
            for contract in contracts:
                for qa in contract.qas:
                    if q_id in qa.get("id", ""):
                        answers = qa.get("answers", [])
                        if answers:
                            for ans in answers:
                                text = ans.get("text", "").strip()
                                if text:
                                    examples.append(text)
                        break
                if len(examples) >= 3:
                    break

            # Determine if binary or extraction
            is_binary = any(ans.lower() in ("yes", "no") for ans in examples[:3])

            category = ClauseCategory(
                question_id=q_id,
                category_name=category_name,
                question_text=f"Highlight parts related to {category_name}",
                is_binary=is_binary,
                typical_answers=examples[:5],
            )

            categories.append(category)

        return categories

    def analyze_label_distribution(self, contracts: List[ContractExample]) -> Dict[str, Any]:
        """
        Analyze distribution of labels across dataset.

        Returns:
            Dictionary with statistics about label distribution.
        """
        stats = {
            "total_contracts": len(contracts),
            "total_qa_entries": 0,
            "total_answers": 0,
            "per_category": defaultdict(lambda: {"total": 0, "answered": 0}),
        }

        for contract in contracts:
            for qa in contract.qas:
                stats["total_qa_entries"] += 1

                # Extract category from ID
                qa_id = qa.get("id", "")
                if "_" in qa_id:
                    category = qa_id.split("_")[1]
                    stats["per_category"][category]["total"] += 1

                    answers = qa.get("answers", [])
                    if answers and answers[0].get("text"):
                        stats["total_answers"] += 1
                        stats["per_category"][category]["answered"] += 1

        # Convert to regular dict
        stats["per_category"] = dict(stats["per_category"])

        return stats

    @staticmethod
    def _sanitize_id(text: str) -> str:
        """Sanitize text for use as ID."""
        return text.replace(" ", "_").replace("/", "_").lower()[:50]

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """Normalize filename for matching."""
        return filename.strip().lower().replace(" ", "_")

    @staticmethod
    def _fuzzy_filename_match(name1: str, name2: str, threshold: float = 0.7) -> bool:
        """
        Fuzzy match two filenames.

        Uses simple token-based matching as heuristic.
        """
        tokens1 = set(name1.split("_"))
        tokens2 = set(name2.split("_"))

        if not tokens1 or not tokens2:
            return False

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        similarity = intersection / union if union > 0 else 0

        return similarity >= 0.6

    def create_splits(
        self,
        contracts: List[ContractExample],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple[List[ContractExample], List[ContractExample], List[ContractExample]]:
        """
        Create train/val/test splits deterministically.

        Uses contract filename hash for reproducible splits.

        Args:
            contracts: List of all contracts.
            train_ratio: Fraction for training set.
            val_ratio: Fraction for validation set.

        Returns:
            Tuple of (train, val, test) contract lists.
        """
        train_contracts = []
        val_contracts = []
        test_contracts = []

        for contract in contracts:
            # Use filename hash for reproducible split
            # MD5 used purely for deterministic test-split bucketing of filenames.
            # Not a security primitive — flagging usedforsecurity=False so bandit
            # recognises the intent.
            hash_val = int(
                hashlib.md5(contract.filename.encode(), usedforsecurity=False).hexdigest(), 16
            )
            bucket = hash_val % 100

            if bucket < train_ratio * 100:
                train_contracts.append(contract)
            elif bucket < (train_ratio + val_ratio) * 100:
                val_contracts.append(contract)
            else:
                test_contracts.append(contract)

        logger.info(
            f"Created splits: "
            f"train={len(train_contracts)}, "
            f"val={len(val_contracts)}, "
            f"test={len(test_contracts)}"
        )

        return train_contracts, val_contracts, test_contracts
