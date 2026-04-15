import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"


def ask_mistral(prompt: str) -> str:
    """Send a prompt to Ollama's Mistral model and return the response text."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError("❌ Cannot connect to Ollama. Please run: ollama serve")
    except requests.exceptions.Timeout:
        raise RuntimeError("❌ Ollama request timed out. The model may still be loading.")
    except Exception as e:
        raise RuntimeError(f"❌ Ollama error: {str(e)}")


def generate_questions(topic: str) -> list[str]:
    """Generate 5 viva-style questions for the given topic."""
    prompt = (
        f"You are a university examiner conducting a viva (oral exam).\n"
        f"Generate exactly 5 clear, concise viva questions on the topic: '{topic}'.\n"
        f"Format: one question per line, numbered 1 to 5. Do not include any extra text."
    )
    response = ask_mistral(prompt)
    lines = [line.strip() for line in response.split("\n") if line.strip()]
    # Remove leading numbers like "1." or "1)"
    questions = []
    for line in lines:
        if line and line[0].isdigit():
            # Strip "1. " or "1) "
            parts = line.split(" ", 1)
            if len(parts) > 1:
                questions.append(parts[1].strip())
            else:
                questions.append(line)
        else:
            questions.append(line)
    return questions[:5]  # Ensure max 5


def evaluate_answer(question: str, answer: str) -> str:
    """Evaluate a student's answer and return structured feedback with a score out of 10."""
    prompt = (
        f"You are a university examiner evaluating a student's viva answer.\n\n"
        f"Question: {question}\n"
        f"Student's Answer: {answer}\n\n"
        f"Provide a structured evaluation with exactly these sections:\n"
        f"Score: X/10\n"
        f"Strengths: [what the student did well]\n"
        f"Weaknesses: [what was missing or incorrect]\n"
        f"Tips: [one actionable suggestion to improve]"
    )
    return ask_mistral(prompt)


def improve_answer(question: str, answer: str) -> str:
    """Return an improved, model-quality answer to the viva question."""
    prompt = (
        f"You are an expert tutor. Rewrite the student's answer below into a clear, "
        f"structured, and academically accurate answer suitable for a university viva exam.\n\n"
        f"Question: {question}\n"
        f"Student's Answer: {answer}\n\n"
        f"Improved Answer:"
    )
    return ask_mistral(prompt)


def get_correct_answer(question: str) -> str:
    """Provide an ideal, model-level answer for the given viva question."""
    prompt = (
        f"You are an expert academic. Provide a clear, concise, and complete answer "
        f"to the following university viva question:\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    return ask_mistral(prompt)