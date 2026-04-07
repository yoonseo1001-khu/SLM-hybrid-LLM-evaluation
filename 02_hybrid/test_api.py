import requests

API_URL = "http://10.189.26.12:30002/v1/chat/completions"

headers = {
    "Content-Type": "application/json"
}

def generate_negative(question, answer):

    prompt = f"""
    Question: {question}
    Correct Answer: {answer}

    Generate a medically plausible but incorrect answer.
    """

    data = {
        "model": "llama-3-3-70b-chat",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    response = requests.post(API_URL, headers=headers, json=data)
    result = response.json()

    return result["choices"][0]["message"]["content"]


# 테스트
q = "What is the treatment for hypertension?"
a = "ACE inhibitors"

neg = generate_negative(q, a)
print("\nGenerated Negative:\n", neg)
