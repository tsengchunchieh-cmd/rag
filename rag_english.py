import pandas as pd
import numpy as np
import json
import math
import random
from typing import List, Dict, Any, Tuple

# Warning: The following code block assumes that the file upload (files.upload) 
# and download (files.download) functions are available when running in a Google Colab or Jupyter environment.
# In a standard Python environment, you need to comment out or replace these functions.
try:
    from google.colab import files
    # Simulate installation in Colab environment
    # !pip -q install pandas numpy scikit-learn
    print("Google Colab environment detected. Using mock file operations.")
except ImportError:
    # In a standard Python environment, define dummy functions to avoid errors
    class MockFiles:
        def upload(self):
            print("--- MOCK: files.upload() - Please upload the CSV file in a real environment ---")
            return {}
        def download(self, filename):
            print(f"--- MOCK: files.download('{filename}') - The file has been generated, but not actually downloaded ---")
    files = MockFiles()
    print("Standard Python environment detected. Using mock file operations.")

# --- 1. Core Helper Function Definitions ---

def normalize_ability_name(name: str) -> str:
    """Unify the format of ability indicator names: remove leading/trailing spaces, unify full-width and half-width symbols."""
    if not isinstance(name, str):
        return ""
    return (
        name.strip()
        .replace("　", " ")
        .replace("／", "/")
        .replace("  ", " ")
    )

def describe_abilities(abilities: List[str], desc_map: Dict[str, str]) -> str:
    """Convert ability indicators into readable text with Chinese explanations."""
    return ", ".join(
        f"{a}({desc_map.get(a, 'Ability description to be added')})"
        for a in abilities
    )

def build_rule_base(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert the association rule DataFrame into a knowledge rule base structure."""
    rule_base = []
    for _, row in df.iterrows():
        rule_base.append({
            "antecedent": [a.strip() for a in row["Antecedent"].split(",")],
            "consequent": [c.strip() for c in row["Consequent"].split(",")],
            "support": float(row["support"]),
            "confidence": float(row["confidence"]),
            "lift": float(row["lift"]),
            "explain": row["explain"]
        })
    return rule_base

def score_rule(student_abilities: List[str], rule: Dict[str, Any], w_conf: float = 0.5, w_lift: float = 0.5) -> float:
    """Calculate the relevance score between a single rule and the student's error abilities."""
    student_set = set(a.strip() for a in student_abilities)
    antecedent_set = set(rule["antecedent"])

    # Antecedent ability overlap ratio
    overlap_ratio = len(student_set & antecedent_set) / (len(antecedent_set) + 1e-9)

    # Comprehensive confidence and lift (lift is scaled to avoid dominance)
    quality_score = (
        w_conf * rule["confidence"] +
        w_lift * min(rule["lift"] / 2.0, 1.0)
    )
    return overlap_ratio * quality_score

def retrieve_rules(student_abilities: List[str], rule_base: List[Dict[str, Any]], top_k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
    """Retrieve the Top-K most relevant rules based on the student's error abilities."""
    scored_rules = [
        (score_rule(student_abilities, rule), rule)
        for rule in rule_base
    ]
    scored_rules = sorted(scored_rules, key=lambda x: x[0], reverse=True)
    
    # Only return rules with a score greater than 0
    return [(score, rule) for score, rule in scored_rules[:top_k] if score > 0]

def build_target_abilities(student_abilities: List[str], retrieved_rules: List[Tuple[float, Dict[str, Any]]]) -> List[str]:
    """Aggregate the student's already mistaken abilities and the abilities associated with the rules to generate an ordered list of reinforcement targets."""
    A = set([a.strip() for a in student_abilities])
    cand = set()
    for score, r in retrieved_rules:
        cand.update(r["antecedent"])
        cand.update(r["consequent"])
        
    target = list(A | cand)
    # Sorting strategy: prioritize the student's already mistaken abilities (x not in A is False/0), then sort alphabetically
    target = sorted(set(target), key=lambda x: (x not in A, x))
    return target

# --- 2. Item Generator (Step 6 Logic) ---

def gen_item(ability: str, q: str, opts: List[str], ans_idx: int, rationale: str):
    """Standardize the output structure of item generation"""
    return {
        "ability": ability,
        "question": q,
        "options": opts,
        "answer": opts[ans_idx],
        "rationale": rationale
    }

def gen_word_usage():
    q = "Fill in the most appropriate word according to the context: 'Faced with the challenge, he remained () and not flustered.'"
    opts = ["panicked", "calm", "shaken", "at a loss"]
    return gen_item("Word_Usage", q, opts, 1, "Cultivate the ability to choose words in context")

def gen_sentence_reading():
    q = "Judge the tone: 'You are really “something” this time.' What does “something” most likely express here?"
    opts = ["praise", "surprise", "sarcasm", "anger"]
    return gen_item("Sentence_Reading", q, opts, 2, "Strengthen sentence comprehension and pragmatic judgment")

def gen_paragraph_reading():
    q = "Which of the following sentences best serves as the 'main idea sentence' for this text? (Topic: The importance of recycling classification)"
    opts = ["Recycling can make money.", "Recycling classification can reduce resource waste and protect the environment.", "There is a recycling station near my home.", "Plastic is less environmentally friendly than paper."]
    return gen_item("Paragraph_Reading", q, opts, 1, "Cultivate the ability of paragraph structure and inference")

GENERATOR_MAP = {
    "Word_Usage": gen_word_usage,
    "Sentence_Reading": gen_sentence_reading,
    "Paragraph_Reading": gen_paragraph_reading,
    "Sentence_Reading_and_Aloud": gen_sentence_reading,
    "Paragraph_Reading_and_Aloud": gen_paragraph_reading,
    # Simplify other abilities, using gen_sentence_reading as the default
}

def generate_items_for_ability(ability: str, n: int = 2) -> List[Dict[str, Any]]:
    """Generate n remedial teaching exercises based on the specified ability"""
    generator_fn = GENERATOR_MAP.get(ability, gen_sentence_reading)
    # To simulate multiple questions, we simply call the generator repeatedly
    return [generator_fn() for _ in range(n)]

# --- 3. Markdown Report Generation Function (Step 8 Logic) ---

def render_markdown_packet(student_abilities: List[str], retrieved_rules: List[Tuple[float, Dict[str, Any]]], topN_df: pd.DataFrame, n_per_ability: int = 2) -> str:
    """Integrate the diagnosis results and output them as a formatted Markdown file."""
    md = []
    
    md.append("# 📝 Personalized Remedial Learning Package and Diagnosis Report\n")
    
    # I. Diagnostic Summary
    md.append("## 💡 I. Diagnostic Summary: Analysis of Student's Current Weaknesses\n")
    md.append("* **Student's Incorrect Ability Tags (A)**:\n")
    md.append("    * " + ", ".join(student_abilities) + "\n")
    
    # II. Rule Analysis
    md.append("\n---\n")
    md.append("## 🔗 II. Association Rule Analysis (RAG Retrieval Results)\n")
    if retrieved_rules:
        for score, r in retrieved_rules:
            ante = ", ".join(r["antecedent"])
            cons = ", ".join(r["consequent"])
            md.append(f"* **Rule**: If a mistake is made in {ante}, there is a high probability of making a mistake in {cons} as well. (Score={score:.3f})\n")
    else:
        md.append("* No highly relevant rules were retrieved for the student's error abilities.\n")

    # III. Remedial Core
    md.append("\n---\n")
    md.append("## 🎯 III. Remedial Core: Target Abilities for Question Generation\n")
    focus = build_target_abilities(student_abilities, retrieved_rules)
    md.append("* **Focus Ability List**:\n")
    md.append("    * " + ", ".join(focus) + "\n")
    
    # IV. Practice Questions
    md.append("\n---\n")
    md.append("## 📚 IV. Personalized Practice Questions and Answer Explanations (N={})\n".format(n_per_ability))
    for ab in focus:
        md.append(f"### {ab}\n")
        items = generate_items_for_ability(ab, n=n_per_ability)
        for idx, it in enumerate(items, 1):
            opts = " / ".join(it["options"])
            md.append(f"**Q{idx}.** {it['question']}\n")
            md.append(f"* Options: {opts}\n* **Reference Answer**: {it['answer']}\n* Rationale: {it['rationale']}\n")
        md.append("\n")

    # V. Teaching Suggestions
    md.append("\n---\n")
    md.append("## 🧑‍🏫 V. Teaching and Tutoring Suggestions (Based on Rule Association)\n")
    md.append("- **Fundamentals First**: According to the rule (Word_Usage → Sentence_Reading), please strengthen word collocation and context analysis before paragraph reading.\n")
    md.append("- **Transition and Connection**: First use 'Sentence_Reading' to introduce conjunctions and tone judgment, and then transition to 'Paragraph_Reading' for main idea and structural inference.\n")
    
    # VI. Macro Data Reference
    md.append("\n---\n")
    md.append("## 📊 VI. Macro Data Reference: Top 10 Ability Error Rates\n")
    # Use to_markdown to output the table
    md.append(topN_df.to_markdown(index=False))
    
    return "\n".join(md)


# =================================================================
# Main Execution Flow
# =================================================================

print("\n--- [Project Initialization Start] ---\n")

# --- A. Simulate Data Upload and Reading (Step 2, 3) ---

# 1. Simulate ability_error_rate.csv content
ability_df_mock = pd.DataFrame({
    "Ability_Indicator_Single_Question": ['Sentence_Reading', 'Paragraph_Reading', 'Word_Usage', 'Paragraph_Structure', 'Word_Recognition'],
    "Error_Rate": [0.65, 0.58, 0.61, 0.45, 0.50]
})

# 2. Simulate frequent_wrong_sets.csv content (only for simulating the flow, does not directly affect the final report)
freq_df_mock = pd.DataFrame({
    "Items": ['Sentence_Reading,Paragraph_Reading', 'Word_Usage,Sentence_Reading'],
    "support": [0.15, 0.12]
})

# 3. Simulate assoc_rules.csv content (the most important rule data)
rules_df_mock = pd.DataFrame({
    "Antecedent": ['Sentence_Reading', 'Word_Usage', 'Paragraph_Structure', 'Word_Recognition'],
    "Consequent": ['Paragraph_Reading', 'Sentence_Reading', 'Paragraph_Reading', 'Word_Usage'],
    "support": [0.15, 0.12, 0.10, 0.06],
    "confidence": [0.85, 0.72, 0.68, 0.60],
    "lift": [1.46, 1.25, 1.20, 1.15]
})

# Execute normalization of Step 3-3, 3-4, 3-5 (only key normalization for rule data)
rules_df_mock["Antecedent"] = rules_df_mock["Antecedent"].apply(normalize_ability_name)
rules_df_mock["Consequent"] = rules_df_mock["Consequent"].apply(normalize_ability_name)


# --- B. Build Knowledge Rule Base (Step 4) ---

# Filter threshold settings
rule_thresholds = {
    "min_support": 0.05,
    "min_confidence": 0.60,
    "min_lift": 1.10
}

# Filter high-quality association rules (Step 4-3)
filtered_rules_df = rules_df_mock[
    (rules_df_mock["support"] >= rule_thresholds["min_support"]) &
    (rules_df_mock["confidence"] >= rule_thresholds["min_confidence"]) &
    (rules_df_mock["lift"] >= rule_thresholds["min_lift"])
].copy()

# Create rule text explanation (Step 4-4)
def build_rule_explanation_custom(row):
    return (
        f"If a student makes an error in {row['Antecedent']}, "
        f"there is a high probability of making an error in {row['Consequent']} as well"
        f" (confidence={row['confidence']:.2f}, "
        f"lift={row['lift']:.2f})."
    )
filtered_rules_df["explain"] = filtered_rules_df.apply(build_rule_explanation_custom, axis=1)

# Construct knowledge rule base (Step 4-5)
rule_base = build_rule_base(filtered_rules_df)

# Simulate file download
with open("rule_base.json", "w", encoding="utf-8") as f:
   json.dump(rule_base, f, ensure_ascii=False, indent=2)
files.download("rule_base.json")
print(f"A total of {len(rule_base)} high-quality rules were created.")


# --- C. Hybrid RAG Rule Retrieval (Step 5) ---

# Test case: Student's error ability tags
student_errors = ["Sentence_Reading", "Paragraph_Reading", "Word_Usage"]

# Retrieve the most relevant rules (top_k=7)
top_rules = retrieve_rules(
    student_abilities=student_errors,
    rule_base=rule_base,
    top_k=7
)
print(f"\nRetrieved {len(top_rules)} most relevant rules for the student's errors.")


# --- D. Pre-computation for Report Generation (Step 9 Logic) ---

# Simulate Top N error rate DataFrame
data_topN = {
    'Ability_Tag': ['Vocabulary_Analysis', 'Sentence_Structure', 'Rhetoric_Application', 'Text_Main_Idea', 'Sentence_Reading', 'Word_Usage', 'Paragraph_Reading', 'Grammar_Judgment', 'Literary_Knowledge', 'Logical_Inference'],
    'Error_Rate': [0.85, 0.79, 0.72, 0.68, 0.65, 0.61, 0.58, 0.55, 0.50, 0.48]
}
top_ability_errors = pd.DataFrame(data_topN)
topN = top_ability_errors.head(10).rename(columns={'Error_Rate': 'Error Rate'})


# --- E. Output Markdown Report (Step 7/8 Integration) ---

packet = render_markdown_packet(
    student_errors, 
    top_rules, 
    topN, 
    n_per_ability=2
)

# Output results
print("\n--- [generated_packet.md Content Simulation Start] ---")
print(packet)
print("--- [generated_packet.md Content Simulation End] ---")

# Simulate file download
with open("generated_packet.md","w",encoding="utf-8") as f:
    f.write(packet)
files.download("generated_packet.md")
