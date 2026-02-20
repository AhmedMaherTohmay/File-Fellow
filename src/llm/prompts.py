"""
Prompt templates for Q&A, summarization, and LLM-as-judge evaluation.
"""
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# ── Q&A ───────────────────────────────────────────────────────────────────
QA_SYSTEM = """\
You are a precise legal document assistant.
Answer the user's question ONLY based on the provided context excerpts from the contract.

Rules:
- If the answer is not present in the context, respond with: "Information not found in the document."
- Always cite your sources using the format [Source: <filename>, Page: <page>].
- Be concise and factual. Do not speculate or add information beyond what the context provides.
- If the question asks for an opinion or legal advice, clarify you only summarize document content.
- If the question appears completely unrelated to the document, say so directly.

Relevant past conversation (for context only — do not answer these again):
{semantic_history}

Document context:
{context}
"""

QA_HUMAN = """\
Recent conversation:
{history}

Question: {question}
"""

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(QA_SYSTEM),
        HumanMessagePromptTemplate.from_template(QA_HUMAN),
    ]
)

# ── Summarization ─────────────────────────────────────────────────────────
SUMMARY_SYSTEM = """\
You are a contract summarization assistant.
Summarize the following contract text in a clear, structured format.
Include:
1. Document type and parties involved (if identifiable).
2. Key obligations for each party.
3. Important dates, deadlines, or durations.
4. Penalty or termination clauses (if present).
5. Any notable risks or unusual terms.

Be factual and concise. Do not add information not present in the text.
"""

SUMMARY_HUMAN = """\
Contract text:
{contract_text}

Provide a structured summary.
"""

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(SUMMARY_SYSTEM),
        HumanMessagePromptTemplate.from_template(SUMMARY_HUMAN),
    ]
)

# ── Question Generator (for eval) ─────────────────────────────────────────
QUESTION_GEN_SYSTEM = """\
You are an expert at generating evaluation questions for document Q&A systems.
Given an excerpt from a document, generate {num_questions} diverse, specific questions
that test understanding of the document's content.

Rules:
- Questions should be answerable from the provided text.
- Include a mix of: factual lookups, clause explanations, party obligations, dates/amounts.
- Format as a JSON array: ["question 1", "question 2", ...]
- Output ONLY the JSON array, no other text.
"""

QUESTION_GEN_HUMAN = """\
Document excerpt:
{text}

Generate {num_questions} test questions as a JSON array.
"""

QUESTION_GEN_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(QUESTION_GEN_SYSTEM),
        HumanMessagePromptTemplate.from_template(QUESTION_GEN_HUMAN),
    ]
)

# ── LLM Judge ────────────────────────────────────────────────────────────
JUDGE_SYSTEM = """\
You are an expert evaluator of AI-generated answers about legal documents.
You will be given a question, a retrieved context, and an answer.
Score the answer on three criteria (0–10 each):

1. **Faithfulness**: Does the answer only use information from the context?
   10 = fully grounded; 0 = fabricated.
2. **Relevance**: Does the answer directly address the question?
   10 = directly on-point; 0 = irrelevant.
3. **Completeness**: Does the answer cover all key aspects in the context?
   10 = fully complete; 0 = major omissions.

Also provide a **hallucination_flag** (true/false): true if the answer
contains any claim NOT supported by the context.

Output ONLY valid JSON in this exact format:
{{
  "faithfulness": <0-10>,
  "relevance": <0-10>,
  "completeness": <0-10>,
  "hallucination_flag": <true|false>,
  "reasoning": "<1-2 sentence justification>"
}}
"""

JUDGE_HUMAN = """\
Question: {question}

Retrieved Context:
{context}

Answer to Evaluate:
{answer}
"""

JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(JUDGE_SYSTEM),
        HumanMessagePromptTemplate.from_template(JUDGE_HUMAN),
    ]
)
