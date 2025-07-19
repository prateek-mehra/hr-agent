# AI Agent HR Chat App

A sophisticated multi-agent HR system that uses LangGraph to orchestrate different specialized agents for various HR tasks including resume screening, policy feedback, and remote work FAQs.

## 🏗️ Architecture Overview

The system uses a **classifier-first architecture** with specialized agents:

```
User Query → Classifier Agent → Specialized Agent → Polishing Agent → Final Response
```

### Agent Types:
1. **Classifier Agent** - Routes queries to appropriate specialized agents
2. **Resume Screening Agents** (Data Engineering & SDE) - Automates resume shortlisting
3. **HR Policy Recommendation Agent** - Provides policy improvement suggestions
4. **Remote Work FAQ Agent** - Answers remote work policy questions
5. **Polishing Agent** - Refines outputs for human-readable responses

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Gemini CLI tool installed at `/usr/local/bin/gemini`
- Resume screening scripts in `resume-screening-tool/` directory

### Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure PDF paths in `main.py`:
   ```python
   DOC_PDF_PATHS = [
       "/path/to/your/Harassment Prevention Policy.pdf",
       # Add more HR policy PDFs
   ]
   REMOTE_PDF_PATHS = [
       "/path/to/your/Remote Work Policy.pdf",
       # Add more remote work policy PDFs
   ]
   ```

### Running the Application

#### Option 1: Web Server (FastAPI)
```bash
python main.py --serve
```
Server starts at http://localhost:8000

API endpoint: `POST /chat` with JSON payload `{"query": "your question"}`

#### Option 2: Streamlit UI
```bash
streamlit run chat_ui.py
```
Access at http://localhost:8501

#### Option 3: Command Line Testing
```bash
python main.py
```
Runs predefined test queries directly

## 🤖 Agentic Flows

### 1. Resume Screening Flow

**Purpose**: Automatically screen and shortlist resumes for technical positions

**Classification Triggers**: 
- "Please shortlist resumes for the data engineering role."
- "Please shortlist resumes for the SDE role."

**Process**:
1. Query classification identifies screening intent (`screening_data_engg` or `screening_sde`)
2. Executes initial screening script (`resume-screening-tool/data_engg.py` or `sde.py`)
3. Reads results from `shortlisted_resumes/{role}/` directory
4. AI-powered final shortlisting using Gemini (selects top 10)
5. Copies final selections to `shortlisted_resumes/{role}_final/`
6. Response polished for hiring manager consumption

**Screening Criteria**:
- Tier 1 college (IIT, NIT, BITS etc)
- Good coding profile
- Good (8+) CGPA
- Relevant internship experience
- Strong projects

**Test Case - Data Engineering**:
```
Input: "Please shortlist resumes for the data engineering role."

Classification: screening_data_engg

Output Example:
"Based on my analysis of the candidate pool, here are the top 10 data engineering candidates selected for interviews:

**Selected Candidates:**

1. **Rahul Sharma** (rahul_sharma_resume.pdf)
   - IIT Delhi, Computer Science, CGPA: 9.2
   - Strong experience with Apache Spark, Kafka, and AWS
   - Score: 9.5/10

2. **Priya Patel** (priya_patel_resume.pdf)
   - NIT Trichy, Data Science, CGPA: 8.8
   - Excellent projects in ETL pipelines and data warehousing
   - Score: 9.2/10

[... continues for all 10 candidates]

These candidates demonstrate strong technical foundations, relevant project experience, and academic excellence suitable for our data engineering requirements."
```

**Test Case - SDE Role**:
```
Input: "Please shortlist resumes for the SDE role."

Classification: screening_sde

Output Example:
"Here are the top 10 SDE candidates recommended for technical interviews:

**Selected Candidates:**

1. **Amit Kumar** (amit_kumar_resume.pdf)
   - IIT Bombay, Computer Science, CGPA: 9.4
   - Strong DSA skills, multiple internships at top tech companies
   - Score: 9.7/10

2. **Sneha Gupta** (sneha_gupta_resume.pdf)
   - BITS Pilani, Computer Science, CGPA: 8.9
   - Impressive full-stack projects and competitive programming achievements
   - Score: 9.3/10

[... detailed breakdown of all candidates]

All selected candidates show exceptional problem-solving abilities and hands-on development experience."
```

### 2. HR Policy Feedback Flow

**Purpose**: Analyze existing HR policies and provide improvement recommendations

**Classification Triggers**: 
- "What is the feedback on the HR policy?"
- "Suggest how to improve our HR policy communication"
- "Is there any room for improvement in the POSH committee appeal procedure?"

**Process**:
1. Query classified as `policy` intent
2. Vector similarity search through HR policy document chunks
3. Retrieves top 5 most relevant policy sections
4. AI analysis identifies gaps and improvement areas
5. Structured recommendations generated

**Test Case - Policy Communication**:
```
Input: "Suggest how to improve our HR policy communication wrt current practices."

Classification: policy

Context Retrieved: [Top 5 relevant chunks from Harassment Prevention Policy.pdf]

Output Example:
"Based on my analysis of your current HR policy documentation, here are key recommendations to enhance policy communication:

**Critical Improvements Needed:**

1. **Digital-First Approach**: Your current policy relies heavily on physical distribution. Implement a centralized digital portal where employees can access real-time policy updates.

2. **Acknowledgment Tracking**: There's no mechanism to confirm employees have read and understood policy changes. Add digital acknowledgment requirements with tracking.

3. **Regular Training Sessions**: Current approach lacks regular reinforcement. Schedule quarterly policy briefings and Q&A sessions.

4. **Feedback Mechanisms**: No structured way for employees to ask policy questions. Create dedicated helpdesk or FAQ system.

5. **Multilingual Support**: Consider providing key policies in regional languages for better comprehension.

**Implementation Priority**: Start with the digital portal and acknowledgment system, as these will provide the foundation for other improvements."
```

**Test Case - POSH Committee Appeals**:
```
Input: "Is there any room for improvement in the POSH committee appeal procedure?"

Classification: policy

Output Example:
"After reviewing your POSH committee appeal procedures, I've identified several areas for strengthening:

**Key Recommendations:**

1. **Timeline Clarity**: The current policy doesn't specify exact timeframes for appeal responses. Recommend adding specific timelines (e.g., 15 days for acknowledgment, 30 days for resolution).

2. **External Review Option**: Consider adding provision for external expert review in complex cases to ensure complete objectivity.

3. **Support System**: The policy lacks details on counseling and support services for complainants during the appeal process.

4. **Documentation Standards**: Enhance requirements for detailed documentation at each stage of the appeal process.

5. **Training Requirements**: Specify mandatory annual training hours for committee members on updated legal requirements.

These improvements will strengthen both process integrity and employee confidence in the system."
```

### 3. Remote Work FAQ Flow

**Purpose**: Answer employee questions about remote work policies using company documentation

**Classification Triggers**: 
- "Can I work remotely from another country for a month?"
- "What are the factors based on which eligibility for remote work is determined?"
- "What is the meaning of RW?"

**Process**:
1. Query classified as `remote` intent
2. Vector search through remote work policy document chunks
3. Retrieves top 3 most relevant sections
4. Direct policy-based answers without hallucination
5. Human-friendly response formatting

**Test Case - International Remote Work**:
```
Input: "Can I work remotely from another country for a month?"

Classification: remote

Context Retrieved: [Top 3 relevant chunks from Remote Work Policy.pdf]

Output Example:
"Based on our remote work policy, working from another country for a month requires special approval since it exceeds our standard duration limits.

**Process for your request:**
1. Submit request at least 30 days in advance through the HR portal
2. Obtain approval from your direct manager and HR
3. Ensure tax and legal compliance documentation
4. Confirm adequate internet connectivity and workspace setup

**Additional requirements:**
- Valid work authorization in the destination country
- Signed international remote work agreement
- Regular check-ins with your team during overlapping hours

Please contact HR to initiate the approval process for your specific situation."
```

**Test Case - Eligibility Criteria**:
```
Input: "What are the factors based on which eligibility for remote work or RW is determined?"

Classification: remote

Output Example:
"Remote work eligibility is determined based on these key factors:

**Job Role Compatibility:**
- Nature of work can be performed independently
- Minimal requirement for physical presence
- Role doesn't involve handling confidential physical documents

**Performance Criteria:**
- Consistent performance ratings of 'Meets Expectations' or above
- No disciplinary actions in the past 12 months
- Demonstrated self-management skills

**Technical Requirements:**
- Reliable internet connection with adequate bandwidth
- Dedicated workspace with proper ergonomics
- Access to required software and security tools

**Duration and Approval:**
- Short-term requests: Manager approval required
- Extended periods: HR and senior management approval
- International locations: Additional legal and tax clearance

Your eligibility will be assessed based on all these factors during the application process."
```

**Test Case - Policy Definitions**:
```
Input: "What is the meaning of RW?"

Classification: remote

Output Example:
"RW stands for 'Remote Work' in our company policy documentation. 

It refers to any work arrangement where employees perform their job duties from a location other than the traditional office environment, including:
- Home-based work
- Co-working spaces
- Temporary locations during business travel
- International locations (with special approval)

RW arrangements can be temporary (few days) or extended (several weeks/months) based on business needs and approval levels."
```

## 🛠️ Technical Implementation

### Key Technologies
- **LangGraph**: Agent orchestration and workflow management
- **FastAPI**: Web server for API endpoints
- **Streamlit**: Simple web UI for testing
- **SentenceTransformers**: Vector embeddings (`all-mpnet-base-v2`)
- **PyMuPDF**: PDF processing and text extraction
- **Gemini CLI**: Large language model integration

### Document Processing Pipeline
```python
# Intelligent heading-aware PDF chunking
def pdf_to_heading_chunks(pdf_path, min_heading_fontsize=13, context_paragraphs=2):
    # Detects headings based on font size and formatting
    # Groups content under each heading with context paragraphs
    # Returns structured chunks for better semantic search
```

### Vector Search Implementation
```python
# Semantic similarity search
query_embedding = get_embedding(query)
similarities = [cosine_similarity(query_emb, chunk_emb) for chunk_emb in doc_embeddings]
top_chunks = get_top_k_chunks(similarities, k=5)  # policy: 5, remote: 3
```

### Configuration Parameters
```python
GEMINI_CLI_PATH = "/usr/local/bin/gemini"          # Gemini CLI location
SERPAPI_KEY = "your_api_key"                       # Search API (unused currently)
EMBEDDING_MODEL = "all-mpnet-base-v2"              # Sentence transformer model
MIN_HEADING_FONTSIZE = 13                          # PDF heading detection threshold
CONTEXT_PARAGRAPHS = 4                             # Paragraphs per chunk
```

## 📁 File Structure

```
ai-agent-hr-chat/
├── main.py                    # Core orchestrator, agents, and FastAPI server
├── chat_ui.py                # Streamlit web interface
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore patterns
├── package-lock.json        # Node.js lock file (legacy)
├── resume-screening-tool/   # Resume processing scripts (external dependency)
│   ├── data_engg.py        # Data engineering screening logic
│   └── sde.py              # SDE screening logic
└── shortlisted_resumes/    # Generated during screening process
    ├── data_engg/          # Initial data engineering shortlist
    ├── data_engg_final/    # Final data engineering shortlist  
    ├── sde/                # Initial SDE shortlist
    └── sde_final/          # Final SDE shortlist
```

## 🔧 Agent Architecture Details

### State Management
```python
# Shared state object passed between agents
state = {
    "query": "user input",
    "route": "classifier decision", 
    "screening_raw_answer": "screening results",
    "policy_feedback_raw": {"question": "...", "chunks": [...]},
    "remote_raw_answer": {"question": "...", "chunks": [...]},
    "final_response": "polished output"
}
```

### Agent Flow Graph
```python
builder = StateGraph(dict)
builder.add_node("classifier", classifier_agent)
builder.add_node("screening_data_engg", lambda state: screening_agent(state, "data_engg"))
builder.add_node("screening_sde", lambda state: screening_agent(state, "sde"))
builder.add_node("policy", HRPolicyRecommendationAgent(doc_chunks))
builder.add_node("remote", RemoteWorkFAQAgent(remote_chunks))
builder.add_node("polish", polishing_agent)

# Conditional routing based on classification
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
```

## 📊 Usage Examples

### API Usage (FastAPI)
```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"query": "Please shortlist resumes for the data engineering role."}'
```

### Streamlit UI Usage
1. Start the UI: `streamlit run chat_ui.py`
2. Enter query in the text input
3. Click the ➤ button or press Enter
4. View the conversation history with user messages on right, agent responses on left

### Direct Testing
Modify the test queries in `main.py`:
```python
queries = [
    "Please shortlist resumes for the data engineering role.",
    "Suggest how to improve our HR policy communication wrt current practices.",
    "Can I work remotely from another country for 45 days?",
]
```

## 🚨 Dependencies & Requirements

### External Dependencies
- **Resume Screening Tool**: Requires `resume-screening-tool/` directory with `data_engg.py` and `sde.py` scripts
- **Gemini CLI**: Must be installed and accessible at `/usr/local/bin/gemini`
- **PDF Documents**: HR policy and remote work policy PDFs must exist at configured paths

### Python Requirements
```
fastapi          # Web server framework
uvicorn         # ASGI server
langgraph       # Agent orchestration
sentence-transformers  # Vector embeddings
numpy           # Numerical computations
PyMuPDF         # PDF processing
requests        # HTTP client
streamlit       # Web UI framework
```

## 🔒 Error Handling & Edge Cases

### Resume Screening Errors
- Missing screening scripts → Error message returned
- No resumes in shortlisted folder → "No resumes found" message
- Subprocess failures → Stderr captured and returned

### Document Processing Errors
- Missing PDF files → Graceful skip with warning
- Corrupted PDFs → Error logged, processing continues
- Empty documents → Handled with empty chunk lists

### LLM Integration Errors
- Gemini CLI failures → Error message with stderr details
- Timeout issues → Subprocess timeout handling
- Invalid responses → Fallback error messages

## 🚀 Extending the System

### Adding New Agent Types

1. **Create Agent Function**:
```python
def new_agent_type(state: dict) -> dict:
    query = state["query"]
    # Your agent logic here
    result = process_query(query)
    state["new_agent_output"] = result
    return state
```

2. **Update Classifier**:
```python
# In classifier_agent function, add new classification option
prompt = (
    "Classify as one of: 'screening_data_engg', 'screening_sde', 'policy', 'remote', 'new_type'"
    # Add examples for new type
)
```

3. **Register in Orchestrator**:
```python
builder.add_node("new_type", new_agent_type)
builder.add_conditional_edges(
    "classifier", 
    lambda state: state["route"],
    {
        # ... existing routes
        "new_type": "new_type"
    }
)
builder.add_edge("new_type", "polish")  # Connect to polishing agent
```

### Customizing Document Sources
```python
# Update paths in main.py
DOC_PDF_PATHS = [
    "/path/to/your/custom_hr_policy.pdf",
    "/path/to/additional/policy_docs.pdf",
]

REMOTE_PDF_PATHS = [
    "/path/to/your/remote_work_policy.pdf",
    "/path/to/flexible_work_guidelines.pdf", 
]
```

## 📈 Performance & Scalability

### Current Limitations
- **Single-threaded**: Processes one query at a time
- **Memory-based**: All embeddings stored in RAM
- **No Caching**: Embeddings recomputed on each restart
- **File-based**: Resume processing relies on file system operations

### Optimization Opportunities
- Implement embedding caching with persistence
- Add async processing for concurrent queries
- Use database storage for document chunks and metadata
- Implement response caching for common queries
- Add batch processing for resume screening

## 🤝 Development Guidelines

### Code Structure
- Keep agent functions pure (input state → output state)
- Use descriptive variable names and clear comments
- Handle all error cases with meaningful messages
- Follow existing prompt engineering patterns

### Testing New Agents
1. Add test queries to the `queries` list in `main.py`
2. Run `python main.py` to test without web server
3. Verify classification works correctly
4. Check that polishing agent handles new output format
5. Test edge cases and error conditions

### Contributing
1. Follow the existing agent pattern for consistency
2. Add comprehensive error handling
3. Update this README with new functionality
4. Test with actual documents and realistic queries
5. Ensure backward compatibility with existing agents

This system provides a solid foundation for HR automation while remaining extensible for future enhancements and integrations.