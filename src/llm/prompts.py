"""
Prompt templates for Q&A and summarization.
"""
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# ── Q&A ───────────────────────────────────────────────────────────────────
QA_SYSTEM = """\
You are a precise legal document assistant.
Answer the user's question ONLY based on the provided context excerpts from the contract.
Rules:
- If the answer is not present in the context, respond with: "Information not found in the document."
- Always cite your sources using the format [Source: <filename>, Page: <page>, Chunk: <chunk_id>].
- Be concise and factual. Do not speculate or add information beyond what the context provides.
- If the question asks for an opinion or legal advice, clarify that you only summarize document content.

Context:
{context}
"""

QA_HUMAN = """\
Conversation history:
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
