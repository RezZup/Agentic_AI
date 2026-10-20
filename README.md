# RAG Experiments

Hands-on experiments with Retrieval-Augmented Generation (RAG) in Python.
This repository compares traditional vector-based retrieval with a
reasoning-based, vectorless workflow and includes examples for document
ingestion, multiple retrieval backends, agentic RAG, LLM integration, and RAG
evaluation.

The project is primarily notebook-driven. It is intended for learning,
experimentation, and prototyping rather than as a production-ready service.

## What is included

### 1. Basic vector RAG

The notebooks in [`BasicRAG/notebook/`](D:/ProjectsGit/ai-ml/RAG/BasicRAG/notebook)
cover the conventional RAG workflow:

1. Load documents such as PDFs, text, and structured data.
2. Split documents into searchable chunks.
3. Generate embeddings.
4. Store embeddings in a vector store.
5. Retrieve relevant context for a user question.
6. Generate an answer with an LLM grounded in the retrieved context.

The repository includes experiments with:

- ChromaDB
- FAISS
- Typesense
- Hugging Face embedding models
- LangChain retrievers and document loaders
- LangGraph agentic workflows
- MongoDB-backed retrieval
- OpenAI-compatible, Google Gemini, and Groq model integrations

### 2. Vectorless / PageIndex RAG

[`VectorlessRAG/notebook/vectorlessRAG.ipynb`](D:/ProjectsGit/ai-ml/RAG/VectorlessRAG/notebook/vectorlessRAG.ipynb)
implements a reasoning-based RAG workflow using PageIndex.

Instead of chunking a PDF and selecting chunks using vector similarity, the
workflow:

1. Uploads a PDF to PageIndex.
2. Builds a hierarchical tree that follows the document structure.
3. Uses an LLM to reason over the tree and select relevant section IDs.
4. Retrieves the selected sections.
5. Generates a grounded response with traceable page and section context.

The notebook also demonstrates:

- Inspecting the generated document tree
- Testing retrieval with multiple questions
- Expert-guided retrieval through domain-specific instructions
- Direct PageIndex Chat API usage
- A self-hosted/open-source direction for the LLM layer

The central comparison is:

```text
Traditional vector RAG:
question -> chunks -> embeddings -> similarity search -> answer

PageIndex RAG:
question + document tree -> LLM reasoning -> relevant sections -> answer
```

Vector similarity is useful, but similarity does not always equal relevance.
The PageIndex approach preserves the document's natural hierarchy and is
particularly useful for structured, professional, legal, financial, and
technical documents.

### 3. RAG evaluation

[`BasicRAG/notebook/rag_evaluation.ipynb`](D:/ProjectsGit/ai-ml/RAG/BasicRAG/notebook/rag_evaluation.ipynb)
demonstrates an evaluation workflow with LangSmith:

1. Create or load a question-and-answer dataset.
2. Run the RAG application over the dataset.
3. Score generated answers with LLM-as-a-judge evaluators.
4. Inspect answer relevance, answer correctness, and retrieval quality.
5. Review experiment results in LangSmith.

### 4. Direct model/API testing

[`apitest.ipynb`](D:/ProjectsGit/ai-ml/RAG/apitest.ipynb) is a small
OpenAI-compatible API test notebook. It:

- Loads the Bazaarlink API key from the environment.
- Lists models available through Bazaarlink.
- Sends a chat-completion request.
- Verifies the model and endpoint configuration independently of LangChain.

This is useful for isolating API connectivity, authentication, and model-name
problems before integrating the model into a larger RAG pipeline.

## Repository layout

```text
.
├── BasicRAG/
│   ├── data/
│   │   ├── books.jsonl
│   │   └── vector_store/
│   └── notebook/
│       ├── LLM.ipynb
│       ├── RagBot.ipynb
│       ├── agenticRag.ipynb
│       ├── agenticRagWithMongodb.ipynb
│       ├── documents.ipynb
│       ├── rag_evaluation.ipynb
│       └── typesense.ipynb
├── VectorlessRAG/
│   └── notebook/
│       └── vectorlessRAG.ipynb
├── src/
│   └── rag/
│       └── __init__.py
├── apitest.ipynb
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

The committed vector-store files are local experiment artifacts. New local
vector stores, virtual environments, Python caches, and `.env` files are
ignored by Git.

## Requirements

- Python **3.14 or newer**
- A Jupyter-capable editor or environment, such as VS Code with the Jupyter
  extension
- Internet access for hosted model, PageIndex, LangSmith, and external
  database examples
- API keys only for the notebooks and integrations you use

Python support is declared in [`pyproject.toml`](D:/ProjectsGit/ai-ml/RAG/pyproject.toml)
as `>=3.14`.

## Installation

### Option A: `uv`

Install the dependencies from the lock file:

```powershell
uv sync
```

Activate the environment in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Option B: `pip`

Create and activate a virtual environment:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Register the environment as a Jupyter kernel if needed:

```powershell
python -m ipykernel install --user --name rag --display-name "Python (RAG)"
```

Select the `Python (RAG)` kernel in VS Code before running a notebook.

## Environment variables

Create a local `.env` file in the repository root. Do not commit this file or
paste secret values into notebooks.

```env
# Bazaarlink OpenAI-compatible API
BAZARRLINK=your_bazaarlink_api_key

# PageIndex PDF indexing and/or Chat API
PAGEINDEX_API_KEY=your_pageindex_api_key

# LangSmith tracing and evaluation
LANGSMITH_API_KEY=your_langsmith_api_key
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=rag-experiments

# Optional provider integrations used by selected notebooks
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key

# Optional MongoDB example
MONGODB_URI=mongodb+srv://username:password@cluster.example.mongodb.net/
```

The current notebooks load the Bazaarlink credential with:

```python
import os
from dotenv import load_dotenv

load_dotenv()
BAZZAAR_API_KEY = os.getenv("BAZARRLINK")
```

The variable name is `BAZARRLINK`. Keep this name consistent with the
notebooks, or update the code and `.env` file together.

## Bazaarlink API configuration

Bazaarlink exposes an OpenAI-compatible endpoint:

```text
https://api.bazaarlink.ai/v1
```

### Direct OpenAI client

```python
from openai import OpenAI

client = OpenAI(
    api_key=BAZZAAR_API_KEY,
    base_url="https://api.bazaarlink.ai/v1",
    timeout=60,
    max_retries=2,
)

models = client.models.list()
for model in models.data:
    print(model.id)

response = client.chat.completions.create(
    model="qwen3.7-flash",
    messages=[
        {
            "role": "user",
            "content": "What is Retrieval-Augmented Generation?",
        }
    ],
    temperature=0,
)

print(response.choices[0].message.content)
```

Use a model ID exactly as returned by `client.models.list()`. Examples seen in
the project include:

```text
qwen3.7-flash
qwen/qwen3.7-flash:free
deepseek-v4-flash
qwen3.7-plus
```

Model availability can change. Listing models is the authoritative way to
check the current IDs for the configured account and endpoint.

### LangChain client

```python
from langchain.chat_models import init_chat_model

llm = init_chat_model(
    model="qwen3.7-flash",
    model_provider="openai",
    api_key=BAZZAAR_API_KEY,
    base_url="https://api.bazaarlink.ai/v1",
    temperature=0,
    timeout=60,
    max_retries=2,
)

answer = llm.invoke("Explain the difference between vector RAG and PageIndex RAG.")
print(answer.content)
```

Pass `temperature` directly rather than inside `model_kwargs` to avoid the
LangChain warning about explicitly declared parameters.

## Running the notebooks

Open the repository in VS Code, select the project virtual environment, and
run the notebooks individually.

Recommended order:

1. **API connectivity:** [`apitest.ipynb`](D:/ProjectsGit/ai-ml/RAG/apitest.ipynb)
2. **Document loading and embeddings:** [`documents.ipynb`](D:/ProjectsGit/ai-ml/RAG/BasicRAG/notebook/documents.ipynb)
3. **Basic chatbot/RAG:** [`RagBot.ipynb`](D:/ProjectsGit/ai-ml/RAG/BasicRAG/notebook/RagBot.ipynb)
4. **Agentic workflows:** [`agenticRag.ipynb`](D:/ProjectsGit/ai-ml/RAG/BasicRAG/notebook/agenticRag.ipynb)
5. **Vectorless retrieval:** [`vectorlessRAG.ipynb`](D:/ProjectsGit/ai-ml/RAG/VectorlessRAG/notebook/vectorlessRAG.ipynb)
6. **Evaluation:** [`rag_evaluation.ipynb`](D:/ProjectsGit/ai-ml/RAG/BasicRAG/notebook/rag_evaluation.ipynb)

The notebooks are independent experiments. You do not have to run every
notebook in this order, but the API test should be run first when diagnosing
model access problems.

## Vectorless RAG workflow

The PageIndex notebook uses the `attention_is_all_you_need.pdf` document path
as an example. Before running it, confirm that the file exists locally or
change `PDF_PATH` to a document available on your machine.

The workflow is:

```python
from pageindex import PageIndexClient

pi_client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
result = pi_client.submit_document(PDF_PATH)
doc_id = result["doc_id"]
```

PageIndex processing is asynchronous. The notebook polls until the document
tree is ready, then uses the tree for retrieval. The LLM tree-search function
returns selected node IDs, and the pipeline fetches the corresponding section
content before generating an answer.

Expert-guided retrieval adds domain rules to the retrieval prompt rather than
requiring embedding-model fine-tuning. For example, a rule can direct queries
about a particular metric or topic toward a specific section of a document.

## Conventional RAG workflow

The BasicRAG notebooks explore the standard retrieval pipeline:

```text
source documents
    -> document loaders
    -> text splitting
    -> embeddings
    -> vector store
    -> retriever
    -> prompt with retrieved context
    -> LLM answer
```

Different notebooks demonstrate different storage and orchestration choices:

- [`documents.ipynb`](D:/ProjectsGit/ai-ml/RAG/BasicRAG/notebook/documents.ipynb):
  document loading and preparation
- [`LLM.ipynb`](D:/ProjectsGit/ai-ml/RAG/BasicRAG/notebook/LLM.ipynb): LLM
  integration experiments
- [`RagBot.ipynb`](D:/ProjectsGit/ai-ml/RAG/BasicRAG/notebook/RagBot.ipynb):
  chatbot and retrieval chains
- [`agenticRag.ipynb`](D:/ProjectsGit/ai-ml/RAG/BasicRAG/notebook/agenticRag.ipynb):
  agentic retrieval and LangGraph orchestration
- [`agenticRagWithMongodb.ipynb`](D:/ProjectsGit/ai-ml/RAG/BasicRAG/notebook/agenticRagWithMongodb.ipynb):
  MongoDB-oriented retrieval
- [`typesense.ipynb`](D:/ProjectsGit/ai-ml/RAG/BasicRAG/notebook/typesense.ipynb):
  Typesense search integration

## Evaluation workflow

The evaluation notebook uses a dataset of questions and expected answers,
executes the RAG application for each question, and applies judge-based
evaluators. Configure LangSmith tracing before running evaluation:

```python
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGSMITH_API_KEY"]
os.environ["LANGCHAIN_PROJECT"] = "rag-experiments"
```

Do not expose LangSmith or model-provider keys in notebook output, commits,
screenshots, or issue reports.

## Troubleshooting

### `ConnectTimeout` or `APITimeoutError`

Confirm that the client uses the Bazaarlink hostname exactly:

```text
https://api.bazaarlink.ai/v1
```

Do not use the old or incorrect value:

```text
https://api.bazaar.ai/v1
```

Also check internet access, VPN/proxy rules, firewall settings, and increase
the client timeout:

```python
client = OpenAI(
    api_key=BAZZAAR_API_KEY,
    base_url="https://api.bazaarlink.ai/v1",
    timeout=60,
    max_retries=2,
)
```

### `401 Unauthorized`

The API key is missing, invalid, expired, or loaded under a different
environment-variable name. Verify the value is present without printing the
secret:

```python
if not BAZZAAR_API_KEY:
    raise RuntimeError("BAZARRLINK is not configured")
```

### `404`, `400`, or model errors

Model IDs are case-sensitive and provider-specific. List the available models
and copy an ID exactly:

```python
for model in client.models.list().data:
    print(model.id)
```

For example, `qwen3.7-flash` and
`qwen/qwen3.7-flash:free` are different IDs. Do not assume that a model ID
from another provider is valid on Bazaarlink.

### LangChain temperature warning

Use:

```python
temperature=0
```

instead of:

```python
model_kwargs={"temperature": 0}
```

### PageIndex document failures

- Confirm `PAGEINDEX_API_KEY` is loaded.
- Confirm `PDF_PATH` points to an existing PDF.
- Wait for asynchronous tree processing to finish.
- Reuse the returned `doc_id` for subsequent tree and retrieval operations.

### Notebook uses the wrong Python environment

In VS Code, select the interpreter from `.venv` and then select the same
environment as the notebook kernel. A package installed in one interpreter is
not automatically available in another.

## Security and data handling

- Keep `.env` files and API keys local.
- Never commit credentials, database connection strings, or private documents.
- Review hosted-provider data-retention and privacy policies before uploading
  sensitive documents.
- Treat generated answers as model output and validate them against the
  retrieved source material.
- Use strict access controls for MongoDB and hosted PageIndex resources.

## Development notes

The project metadata is defined in [`pyproject.toml`](D:/ProjectsGit/ai-ml/RAG/pyproject.toml),
and the pip-compatible dependency list is in
[`requirements.txt`](D:/ProjectsGit/ai-ml/RAG/requirements.txt). The package
source currently provides the `rag` package scaffold in
[`src/rag/`](D:/ProjectsGit/ai-ml/RAG/src/rag).

There is currently no automated test suite or production application entry
point covering all notebooks. Validation is performed by executing the
relevant notebook cells and checking API, retrieval, and evaluation outputs.

## License

No license is currently declared in the repository metadata. Confirm the
intended licensing terms before distributing or reusing the project.
