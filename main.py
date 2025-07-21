import os
import subprocess
import requests
import numpy as np
from typing import List, Dict, Any
from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import re
import json
import time
import sys
import shutil

# ---- CONFIG ----
GEMINI_CLI_PATH = "/usr/local/bin/gemini"
SERPAPI_KEY = "7f4e29155d2a6208f75d35db72d9000eae2a169735e77f9c172995949e035fd8"
EMBEDDING_MODEL = "all-mpnet-base-v2"

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

def call_gemini_cli(prompt: str) -> str:
    try:
        result = subprocess.run(
            [GEMINI_CLI_PATH, "--prompt", prompt],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"[Error] Gemini CLI failed:\n{e.stderr.strip()}"

def get_embedding(text: str) -> np.ndarray:
    return embedding_model.encode([text])[0]

# --- Smarter Heading-aware Chunking ---
def pdf_to_heading_chunks(pdf_path, min_heading_fontsize=13, context_paragraphs=2):
    """
    Uses PyMuPDF to extract heading-based chunks from a PDF.
    Each chunk is a heading + its following N paragraphs (until next heading).
    """
    doc = fitz.open(pdf_path)
    chunks = []
    current_heading = None
    current_chunk = ""
    paragraph_buffer = []
    last_heading_fontsize = None

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                line_text = " ".join([span["text"] for span in line["spans"]]).strip()
                if not line_text:
                    continue
                # Heuristic: treat as heading if font size is large or text is ALL CAPS or matches heading pattern
                max_fontsize = max([span["size"] for span in line["spans"]])
                is_heading = (
                    max_fontsize >= min_heading_fontsize or
                    re.match(r"^[A-Z][A-Z\s\-:]{3,}$", line_text) or
                    line_text.endswith(":")
                )
                if is_heading:
                    # Save previous chunk if exists
                    if current_heading and (current_chunk or paragraph_buffer):
                        chunk_text = current_heading + "\n" + "\n".join(paragraph_buffer[:context_paragraphs])
                        chunks.append(chunk_text.strip())
                        paragraph_buffer = []
                    current_heading = line_text
                    current_chunk = ""
                    last_heading_fontsize = max_fontsize
                else:
                    paragraph_buffer.append(line_text)
                    # If buffer is too big, flush as a chunk (for long sections without headings)
                    if len(paragraph_buffer) >= context_paragraphs and current_heading:
                        chunk_text = current_heading + "\n" + "\n".join(paragraph_buffer[:context_paragraphs])
                        chunks.append(chunk_text.strip())
                        paragraph_buffer = paragraph_buffer[context_paragraphs:]
    # Add any trailing chunk
    if current_heading and (current_chunk or paragraph_buffer):
        chunk_text = current_heading + "\n" + "\n".join(paragraph_buffer[:context_paragraphs])
        chunks.append(chunk_text.strip())
    return chunks

def multi_pdf_to_heading_chunks(pdf_paths, min_heading_fontsize=13, context_paragraphs=2):
    all_chunks = []
    for pdf_path in pdf_paths:
        if os.path.exists(pdf_path):
            chunks = pdf_to_heading_chunks(pdf_path, min_heading_fontsize, context_paragraphs)
            all_chunks.extend(chunks)
    return all_chunks

# --- 1. Classifier Agent ---
def classifier_agent(state: dict) -> dict:
    prompt = (
        "You are an expert HR AI assistant. Classify the following user query as one of: "
        "'screening_data_engg' (for data engineering resume screening intent), "
        "'screening_sde' (for SDE resume screening intent), "
        "'policy' (for HR policy feedback intent), "
        "'remote' (for remote work policy FAQ intent). "
        "Respond with only 'screening_data_engg', 'screening_sde', 'policy', or 'remote' and nothing else.\n\n"
        "Examples:\n"
        "Query: Please shortlist resumes for the data engineering role.\nAnswer: screening_data_engg\n"
        "Query: Please shortlist resumes for the SDE role.\nAnswer: screening_sde\n"
        "Query: What is the feedback on the HR policy?\nAnswer: policy\n"
        "Query: Can I work remotely from another country for a month?\nAnswer: remote\n"
        f"Query: {state['query']}\n"
    )
    result = call_gemini_cli(prompt).strip().lower()
    if result == "screening_data_engg":
        decision = "screening_data_engg"
    elif result == "screening_sde":
        decision = "screening_sde"
    elif result == "policy":
        decision = "policy"
    elif result == "remote":
        decision = "remote"
    else:
        decision = "policy"  # default fallback
    state["route"] = decision
    state["classifier_output"] = result
    return state

def read_pdfs_from_folder(folder_path: str) -> Dict[str, str]:
    pdf_texts = {}
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, fname)
            try:
                with fitz.open(pdf_path) as doc:
                    text = " ".join([page.get_text() for page in doc])
                pdf_texts[fname] = text
            except Exception as e:
                print(f"Error reading {pdf_path}: {e}")
    return pdf_texts

def screening_agent(state: dict, role: str, final_top_k=10) -> dict:
    """
    Generalized screening agent for both data_engg and sde roles.
    1. Runs the initial screening script (data_engg.py or sde.py) to populate shortlisted_resumes/{role}/
    2. Reads those resumes, sends to Gemini for final shortlisting, and copies to shortlisted_resumes/{role}_final/
    """
    # Step 1: Run the initial screening script
    script_name = "data_engg.py" if role == "data_engg" else "sde.py"
    script_path = os.path.join("resume-screening-tool", script_name)
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        state["screening_initial_output"] = result.stdout.strip()
    except subprocess.CalledProcessError as e:
        state["screening_raw_answer"] = f"[{role}ScreeningAgent] Error running {script_name}: {e.stderr.strip()}"
        return state

    # Step 2: Final shortlisting using Gemini
    shortlisted_folder = f"shortlisted_resumes/{role}/"
    final_folder = f"shortlisted_resumes/{role}_final/"
    os.makedirs(final_folder, exist_ok=True)

    pdf_texts = read_pdfs_from_folder(shortlisted_folder)
    if not pdf_texts:
        state["screening_raw_answer"] = "[Error] No resumes found in shortlisted folder."
        return state

    prompt = (
        f"You are an expert technical recruiter. You are hiring for the role: {role}.\n"
        "You have 50 resumes (text below, separated by filename). Out of these, select the best {top_k} resumes for interview, based on the following criteria:\n"
        "1. Tier 1 college (IIT, NIT, BITS etc)\n"
        "2. Good coding profile\n"
        "3. Good (8+) cgpa\n"
        "4. Relevant internship experience (tech stack should sufficiently match with tools used by the role we're hiring for)\n"
        "5. Good projects\n\n"
        "For each selected student, output:\n"
        "- Name (from filename)\n"
        "- Filename (from filename)\n"
        "- Reason for selection (1-2 lines)\n"
        "- Score out of 10\n\n"
        "Strictly select only the best {top_k} resumes. Do not hallucinate names. Only select from the resumes provided below.\n\n"
        "Resumes:\n"
    )
    for fname, text in pdf_texts.items():
        prompt += f"\n---\nFilename: {fname}\n{text[:4000]}\n"  # Limit to 4000 chars per resume for prompt size

    prompt = prompt.replace("{top_k}", str(final_top_k))

    result = call_gemini_cli(prompt)
    selected_names = []
    for line in result.splitlines():
        m = re.search(r"Filename:\*\*([^\n]+)", line)
        if m:
            selected_names.append(m.group(1).strip())
        else:
            m2 = re.search(r"Name\s*[:\-]\s*([^\n,]+)", line)
            if m2:
                selected_names.append(m2.group(1).strip())
    if not selected_names:
        for fname in pdf_texts:
            if fname in result:
                selected_names.append(fname)
    copied = []
    for fname in pdf_texts:
        if any(n.lower() in fname.lower() for n in selected_names):
            src = os.path.join(shortlisted_folder, fname)
            dst = os.path.join(final_folder, fname)
            shutil.copy(src, dst)
            copied.append(fname)
    state["screening_raw_answer"] = result
    state["final_shortlisted"] = copied
    return state

class HRPolicyRecommendationAgent:
    def __init__(self, doc_chunks: List[Dict[str, Any]]):
        self.doc_chunks = doc_chunks

    def __call__(self, state: dict) -> dict:
        query = state["query"]
        query_emb = get_embedding(query)
        sims = [np.dot(chunk["embedding"], query_emb) / (np.linalg.norm(chunk["embedding"]) * np.linalg.norm(query_emb) + 1e-8)
                for chunk in self.doc_chunks]
        top_k = min(5, len(self.doc_chunks))
        top_idxs = np.argsort(sims)[-top_k:][::-1]
        relevant_chunks = [self.doc_chunks[i]["text"] for i in top_idxs]
        state["policy_feedback_raw"] = {
            "question": query,
            "chunks": relevant_chunks
        }
        return state

class RemoteWorkFAQAgent:
    def __init__(self, remote_chunks: List[Dict[str, Any]]):
        self.remote_chunks = remote_chunks

    def __call__(self, state: dict) -> dict:
        query = state["query"]
        query_emb = get_embedding(query)
        sims = [np.dot(chunk["embedding"], query_emb) / (np.linalg.norm(chunk["embedding"]) * np.linalg.norm(query_emb) + 1e-8)
                for chunk in self.remote_chunks]
        top_k = 3
        top_idxs = np.argsort(sims)[-top_k:][::-1]
        relevant_chunks = [self.remote_chunks[i]["text"] for i in top_idxs]
        state["remote_raw_answer"] = {
            "question": query,
            "chunks": relevant_chunks
        }
        return state

def polishing_agent(state: dict) -> dict:
    if "screening_raw_answer" in state:
        answer = state["screening_raw_answer"]
        state["final_response"] = answer
        return state
    elif "policy_feedback_raw" in state:
        feedback_data = state["policy_feedback_raw"]
        question = feedback_data["question"]
        context = "\n\n".join(feedback_data["chunks"])
        print(f"Context for policy feedback: {context}")
        prompt = (
            "You are an expert HR policy consultant. Given the user's request and the following excerpts from the current HR policy, "
            "critique the policy and suggest concrete, actionable improvements. "
            "Base your suggestions strictly on the provided excerpts—do not hallucinate or invent policy details. "
            "Use reasoning and synthesis to highlight ambiguities, missing details, or areas for clarification. "
            "Your answer should be clear, practical, and sound like advice to an HR manager. "
            "Do not include meta-commentary or instructions. Only provide the final recommendations.\n\n"
            f"HR's Request:\n{question}\n\n"
            f"Current HR Policy Excerpts:\n{context}\n"
        )
    elif "remote_raw_answer" in state:
        faq_data = state["remote_raw_answer"]
        question = faq_data["question"]
        answer = "\n\n".join(faq_data["chunks"])
        print(f"Context for remote work FAQ: {answer}")
        prompt = (
            "You are an HR remote work policy expert. Given the user's question and the following top 3 most relevant remote work policy excerpts, answer the user's question using only the information present in these excerpts. "
            "If the answer is a combination of these, combine them concisely. "
            "Do NOT add any information that is not present in the excerpts. "
            "Do NOT hallucinate or invent any policy details. "
            "If none of the excerpts answer the question, reply with 'No relevant policy found.' "
            "Your answer should be clear, friendly, and sound like a human HR manager talking to an employee. "
            "Do not mention options, alternatives, or label anything as 'option'. Only give the single, best, final answer in natural language. "
            "Do not include any meta-commentary or instructions. Just answer the user's question as if you are the HR manager.\n\n"
            f"User's Question:\n{question}\n\n"
            f"Remote Work Policy Excerpts:\n{answer}\n"
        )
    else:
        state["final_response"] = "[Error: Nothing to polish]"
        return state

    polished = call_gemini_cli(prompt).strip()
    if polished.lower().startswith("option"):
        polished = polished.split("\n", 1)[-1].strip()
    state["final_response"] = polished
    return state

def orchestrator(query: str, doc_chunks: List[Dict[str, Any]], remote_chunks: List[Dict[str, Any]], final_top_k=10) -> str:
    builder = StateGraph(dict)
    builder.add_node("classifier", classifier_agent)
    builder.add_node("screening_data_engg", lambda state: screening_agent(state, "data_engg", final_top_k=final_top_k))
    builder.add_node("screening_sde", lambda state: screening_agent(state, "sde", final_top_k=final_top_k))
    builder.add_node("policy", HRPolicyRecommendationAgent(doc_chunks))
    builder.add_node("remote", RemoteWorkFAQAgent(remote_chunks))
    builder.add_node("polish", polishing_agent)

    builder.set_entry_point("classifier")
    builder.add_conditional_edges(
        "classifier",
        lambda state: state["route"],
        {
            "screening_data_engg": "screening_data_engg",
            "screening_sde": "screening_sde",
            "policy": "policy",
            "remote": "remote"
        }
    )
    builder.add_edge("screening_data_engg", "polish")
    builder.add_edge("screening_sde", "polish")
    builder.add_edge("policy", "polish")
    builder.add_edge("remote", "polish")
    builder.add_edge("polish", END)

    graph = builder.compile()
    state = {"query": query}
    final_state = graph.invoke(state)
    return final_state["final_response"]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--serve', action='store_true', help='Run as web server')
    args = parser.parse_args()

    DOC_PDF_PATHS = [
        # "/Users/prateek/Downloads/ai-agents/HR Policy Manual - Leave Policy.pdf",
        "/Users/prateek/Downloads/ai-agents/Harassment Prevention Policy.pdf",
        # Add more HR policy PDFs here
    ]
    REMOTE_PDF_PATHS = [
        "/Users/prateek/Downloads/ai-agents/Remote Work Policy.pdf",
        # Add more remote work policy PDFs here
    ]
    MIN_HEADING_FONTSIZE = 13
    CONTEXT_PARAGRAPHS = 4
    doc_texts = multi_pdf_to_heading_chunks(DOC_PDF_PATHS, min_heading_fontsize=MIN_HEADING_FONTSIZE, context_paragraphs=CONTEXT_PARAGRAPHS)
    remote_texts = multi_pdf_to_heading_chunks(REMOTE_PDF_PATHS, min_heading_fontsize=MIN_HEADING_FONTSIZE, context_paragraphs=CONTEXT_PARAGRAPHS)
    doc_chunks = [{"text": t, "embedding": get_embedding(t)} for t in doc_texts]
    remote_chunks = [{"text": t, "embedding": get_embedding(t)} for t in remote_texts]

    if args.serve:
        from fastapi import FastAPI, Request
        from fastapi.middleware.cors import CORSMiddleware
        import uvicorn
        app = FastAPI()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        @app.post("/chat")
        async def chat(request: Request):
            data = await request.json()
            query = data.get("query", "")
            if not query:
                return {"error": "Missing query"}
            response = orchestrator(query, doc_chunks, remote_chunks)
            return {"response": response}
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        queries = [
            "Please shortlist resumes for the data engineering role.",
            # "Please shortlist resumes for the SDE role.",
            # "Suggest how to improve our HR policy communication wrt current practices.",
            # "Is there any room for improvement in the POSH committee appeal procedure?",
            # "What is the approval process for remote work leave requests?"
            # "Can I work remotely from another country for 45 days?",
            # "What are the factors based on which eligibility for remote work or RW is determined?",
            # "What is the meaning of RW?"
        ]
        for q in queries:
            print(f"\nUser Query: {q}")
            response = orchestrator(q, doc_chunks, remote_chunks)
            print(f"AI Response:\n{response}\n")