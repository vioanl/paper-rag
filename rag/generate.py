from dotenv import load_dotenv
from rag.config import LLM_PROVIDER, LLM_MODEL, MAX_TOKENS

load_dotenv()

SYSTEM = """You answer questions about a set of research papers, using only the numbered sources given.
Rules:
- Use only information in the sources. Do not add outside knowledge.
- After each sentence, cite the sources it comes from, like [1] or [2][4].
- If the sources don't contain the answer, reply only: "The papers don't cover this."
- Be concise."""

def call_llm(system, prompt, model=LLM_MODEL):
    """Send one prompt to the configured LLM and return its text reply."""
    if LLM_PROVIDER == "gemini":
        from google import genai
        from google.genai import types
        client = genai.Client()  # keep a reference, or it closes mid-request
        resp = client.models.generate_content(
            model=model, contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system,
                                               max_output_tokens=MAX_TOKENS))
        return resp.text
    import anthropic
    msg = anthropic.Anthropic().messages.create(
        model=model, max_tokens=MAX_TOKENS, system=system,
        messages=[{"role": "user", "content": prompt}])
    return msg.content[0].text

def format_sources(chunks):
    return "\n\n".join(f"[{n}] ({c['paper']}, p{c['page']})\n{c['text']}"
                       for n, c in enumerate(chunks, 1))

def generate(question, chunks):
    return call_llm(SYSTEM, f"Sources:\n\n{format_sources(chunks)}\n\nQuestion: {question}")
