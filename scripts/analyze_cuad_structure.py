#!/usr/bin/env python3
"""
Analiza CUAD dataset strukture.
Ovo je eksplorativna skripta za razumevanje CUAD_v1.json i master_clauses.csv formata.
"""

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict


def analyze_cuad_json() -> None:
    """Analiziraj CUAD_v1.json SQuAD format."""
    print("\n" + "=" * 70)
    print("ANALIZA CUAD_v1.json (SQuAD FORMAT)")
    print("=" * 70)

    cuad_path = Path("CUAD_v1/CUAD_v1.json")

    with open(cuad_path, "r", encoding="utf-8") as f:
        cuad_data: Dict[str, Any] = json.load(f)

    # Strukturna analiza
    print("\n📊 TOP-LEVEL STRUKTURA:")
    print(f"   - Root keys: {list(cuad_data.keys())}")

    if "data" in cuad_data:
        articles = cuad_data["data"]
        print(f"   - Number of contracts (articles): {len(articles)}")

        # Statistika
        total_paragraphs = 0
        total_qas = 0
        total_answers = 0
        question_categories: Counter = Counter()

        # Prvi ugovor - detaljna analiza
        if articles:
            print("\n📄 STRUKTURA PRVOG UGOVORA (SAMPLE):")
            first_article = articles[0]
            print(f"   - Title: {first_article.get('title', 'N/A')}")
            print(f"   - Article keys: {list(first_article.keys())}")

            paragraphs = first_article.get("paragraphs", [])
            print(f"   - Paragraphs in first contract: {len(paragraphs)}")

            if paragraphs:
                first_para = paragraphs[0]
                print("\n   📋 PRVI PARAGRAF:")
                print(f"      - Context length: {len(first_para.get('context', ''))} chars")
                print(f"      - Context preview: {first_para.get('context', '')[:100]}...")
                print(f"      - Paragraph keys: {list(first_para.keys())}")

                qas = first_para.get("qas", [])
                print(f"      - QA entries: {len(qas)}")

                if qas:
                    first_qa = qas[0]
                    print("\n      ❓ PRVI QA ENTRY:")
                    print(f"         - Question: {first_qa.get('question', 'N/A')}")
                    print(f"         - ID: {first_qa.get('id', 'N/A')}")
                    print(f"         - Is impossible: {first_qa.get('is_impossible', False)}")

                    answers = first_qa.get("answers", [])
                    print(f"         - Answers: {len(answers)}")
                    if answers:
                        ans = answers[0]
                        print(f"         - Answer text: {ans.get('text', 'N/A')}")
                        print(f"         - Answer start: {ans.get('answer_start', 'N/A')}")

        # Globalna statistika kroz sve ugovore
        print("\n📈 GLOBALNA STATISTIKA (kroz sve ugovore):")
        for article in articles:
            total_paragraphs += len(article.get("paragraphs", []))
            for para in article.get("paragraphs", []):
                for qa in para.get("qas", []):
                    total_qas += 1
                    # Ekstraktuj kategoriju iz question-ID (npr. "Q14" → Indemnification)
                    qa_id = qa.get("id", "")
                    if "_" in qa_id:
                        category = qa_id.split("_")[1]
                        question_categories[category] += 1

                    answers = qa.get("answers", [])
                    if answers:
                        total_answers += 1

        print(f"   - Total paragraphs: {total_paragraphs:,}")
        print(f"   - Total QA entries: {total_qas:,}")
        print(f"   - Total answers (non-empty): {total_answers:,}")
        print(f"   - Clause categories detected: {len(question_categories)}")

        if question_categories:
            print("\n   📑 TOP 10 CLAUSE CATEGORIES:")
            for category, count in question_categories.most_common(10):
                print(f"      - {category}: {count}")


def analyze_master_clauses_csv() -> None:
    """Analiziraj master_clauses.csv."""
    print("\n" + "=" * 70)
    print("ANALIZA master_clauses.csv")
    print("=" * 70)

    csv_path = Path("CUAD_v1/master_clauses.csv")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print("\n📊 CSV STRUKTURA:")
    if rows:
        print(f"   - Number of rows: {len(rows)}")
        print(f"   - Column names: {rows[0].keys()}")

        print("\n   🔍 PRVI RED (SAMPLE):")
        first_row = rows[0]
        for key, value in list(first_row.items())[:5]:
            preview = str(value)[:60] if value else "N/A"
            print(f"      - {key}: {preview}...")

        # Analiza kolona
        print("\n   📈 ANALIZA KOLONA:")
        for key in first_row.keys():
            non_empty = sum(1 for row in rows if row.get(key))
            print(f"      - {key}: {non_empty}/{len(rows)} non-empty")


def main() -> None:
    """Glavna analiza."""
    try:
        analyze_cuad_json()
        analyze_master_clauses_csv()

        print("\n" + "=" * 70)
        print("✅ ANALIZA ZAVRŠENA")
        print("=" * 70)

    except FileNotFoundError as e:
        print(f"❌ Fajl nije pronađen: {e}")
    except json.JSONDecodeError as e:
        print(f"❌ JSON greška: {e}")


if __name__ == "__main__":
    main()
