# main.py
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import faiss
import os
import numpy as np

# Load local LLM (ensure this matches the quantized GGUF you have)
llm = Llama(model_path="mistral-7b-instruct-v0.1.Q8_0.gguf", n_ctx=2048)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Hybrid data: QA + Paragraph
qa_pairs = [
    ("What is Hemath's profession?", "Hemath is a full-stack software engineer with strong JavaScript skills and moderate experience in frameworks like React."),
    ("What motivates Hemath in his career?", "Hemath seeks growth in skills, presence, impact, and character. He values quality, scalability, and building from scratch."),
    ("Why did Hemath leave his previous company?", "He left because the environment wasn't conducive to learning or growth."),
    ("What role is Hemath stepping into now?", "He is joining a larger company as a Software Engineer 2."),
    ("What does Hemath believe about humor and respect?", "He sometimes feels his humor makes others take him less seriously, and he's working on being more respectable while staying authentic."),
    ("Is Hemath introverted or extroverted?", "Hemath is an introvert."),
    ("Which communities is Hemath actively part of?", "He is involved in JS Lovers and Think Digital."),
    ("What role does Hemath play in the Think Digital community?", "He mentors students and guides them on their learning journey."),
    ("How does Hemath prefer to communicate thoughtfully?", "He prefers email over chat for deep and mindful interactions."),
    ("What is Hemath's goal with his personal brand?", "To share knowledge, inspire others, and build a niche, loyal audience."),
    ("What kind of content does Hemath want to create?", "Blogs, newsletters, YouTube videos, and visually engaging content like web development tips for LinkedIn."),
    ("What makes Hemath's approach to content unique?", "He focuses on practical and often overlooked technical concepts with a strong design sense."),
    ("What are Hemath's learning goals for this year?", "He plans to learn data science and is currently studying calculus."),
    ("What are Hemath's fitness goals?", "To jog 100 km per week and reduce belly fat through a structured plan."),
    ("What is Hemath's work schedule?", "He works from 11:30 AM to 8:30 PM to align with his Italian team."),
    ("How many hours per week does Hemath want to freelance?", "At least 20 hours per week."),
    ("What is one of Hemath's top weekend priorities?", "Teaching students every Saturday evening."),
    ("What is SnapNostr?", "It's a privacy-focused screenshot generator for the Nostr platform, built by Hemath."),
    ("What principles does Hemath follow in building tools like SnapNostr?", "Ethical, minimalist tech principles — he avoids using heavy analytics."),
    ("What kind of projects is Hemath building?", "npm packages, reusable Flutter components, and event-based tools like QR code generators and networking games."),
    ("What award did Hemath recently win?", "The “Code DJ” award in his current company."),
    ("What is Hemath's nickname?", "DJ Hemath."),
]

# Load main document
with open("data.txt", "r", encoding="utf-8") as f:
    data = f.read()

# Simple paragraph chunking
def chunk_text(text, max_length=500, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+max_length]
        chunks.append(" ".join(chunk))
        i += max_length - overlap
    return chunks

paragraph_chunks = chunk_text(data)

# Combine both
all_chunks = [(q, a, "qa") for q, a in qa_pairs] + [(None, chunk, "paragraph") for chunk in paragraph_chunks]

# Embeddings + Indexing
texts = [a for _, a, _ in all_chunks]
vectors = embedder.encode(texts, convert_to_numpy=True)
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

# Save metadata
metadata = all_chunks

# Ask questions
while True:
    query = input("\nAsk about Hemath (or type 'exit'): ")
    if query.lower() == 'exit':
        break

    q_embedding = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(q_embedding, k=5)

    context_chunks = [metadata[i][1] for i in I[0]]
    context = "\n".join(context_chunks)

    prompt = f"""[INST] You are Hemath's assistant. Answer the question only based on the given information.
    
Context:
{context}

Question: {query}
[/INST]
"""

    response = llm(prompt, max_tokens=512, stop=["</s>"])
    print("\n🧠 Answer:", response["choices"][0]["text"].strip())