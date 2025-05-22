📄 Document Research & Theme Identification Chatbot
An interactive web-based chatbot that enables users to upload scanned images and PDF documents, perform semantic search, extract relevant answers, and identify major themes across a corpus of documents.

🚀 Project Overview
This project was developed as part of an internship task focused on Document Research & Theme Identification using LLMs. The goal is to help users research across a large set of documents, extract relevant insights, and synthesize key themes with precise citations.

Key features:

Upload PDFs and scanned images

OCR for image text extraction

LLM-powered theme synthesis (via Groq + LLaMA 3.1)

Semantic similarity search (via Chroma + HuggingFace Embeddings)

Structured, cited responses in tabular and summarized formats

📁 Folder Structure
bash
Copy
Edit
chatbot_theme_identifier/
├── uploads/                 # Uploaded documents (PDFs/Images)
├── chroma_db/              # Persistent vector store (ChromaDB)
├── templates/
│   └── index.html          # Frontend interface
├── app.py                  # Main Flask backend
├── requirements.txt        # Required Python packages
└── .env                    # Environment variables (API keys, etc.)
🧠 Key Functionalities
1. 📤 Document Upload
Supports PDF, JPG, PNG, JPEG.

Scanned image text extracted via Tesseract OCR.

PDFs parsed using PyPDF2.

2. 🧠 Semantic Search with Vector DB
Text from documents split into 1000-token chunks using LangChain’s RecursiveCharacterTextSplitter.

Chunks embedded with sentence-transformers/paraphrase-MiniLM-L6-v2.

Stored and queried using Chroma vector store.

3. 💬 Query Handling
Users can submit a query via POST /query.

Top matching document chunks retrieved.

Tabular display of answers with doc_id and chunk citation.

4. 🌐 Theme Identification
Retrieved results passed to LLaMA 3.1 via Groq.

Synthesizes key themes across all documents.

Themes returned with supporting evidence and citations.

🛠️ Tech Stack
Layer	Tools Used
Backend	Flask, Python
OCR	Tesseract
LLM	Groq + LLaMA 3.1
Vector DB	ChromaDB
Embeddings	HuggingFace Transformers
PDF Parsing	PyPDF2, pdf2image
Frontend	HTML (Jinja2 via Flask)
Deployment	(To be added: e.g. Render/Railway)

🔐 Setup & Installation
Clone the repository

bash
Copy
Edit
git clone https://github.com/Chourey481/Yash-chourey-wasserstoff-AiInternTask/.git
cd Yash-chourey-wasserstoff-AiInternTask/
Create a virtual environment

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Set up Tesseract

Windows users: Install Tesseract OCR and update the path in app.py.

Configure Environment Variables
Create a .env file:

env
Copy
Edit
GROQ_API_KEY=your_groq_api_key
Run the server

bash
Copy
Edit
python app.py
🧪 Sample Workflow
Go to the homepage.

Upload a PDF or image document.

Submit a natural language query.

Receive:

Relevant chunks in a tabular format with citations

Synthesized theme-based response using Groq

🧠 Credits & References
Tesseract OCR

LangChain

Groq + LLaMA

Chroma

HuggingFace Transformers
 
