"""The Agentic Condom: the corrections layer between lip reading and speech.

It takes what the lip reader read (plus the other readings, per-word confidences, phrase-memory
candidates and the conversation so far) and returns the line the user most likely meant, changing
only words the reader was unsure of (`gate`). `lipread.corrector.Corrector` is its LLM client.
Design, prompt and numbers: `.context/agentic-condom.md`.
"""

from lipread.agentic_condom.condom import BUDGET_MS, AgenticCondom, CondomRequest, CondomResult
from lipread.agentic_condom.gate import SURE_ABOVE, Edit

__all__ = ["BUDGET_MS", "SURE_ABOVE", "AgenticCondom", "CondomRequest", "CondomResult", "Edit"]
