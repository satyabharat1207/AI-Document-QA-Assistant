# 🤖 AI Documents Q&A Assistant

![AI Documents Q&A Assistant Banner](https://github.com/satyabharat1207/AI-Document-QA-Assistant/blob/main/banner.png)

This project is a lightweight AI Document Q&A Assistant, a Streamlit web app that helps you interact with your documents. Simply upload files like PDFs or DOCX, and the app processes them into a searchable format.

You can then ask questions about the uploaded content and receive accurate, cited answers. The assistant also includes a general chat mode, so you can ask it anything—it's not limited to just your documents.

---

## ✨ Features
- 📂 Upload and process multiple documents
- ⚡ Ask questions and get answers from your docs
- 📝 Summarize documents in one click
- 💬 General chat mode (not limited to docs)
- 💾 Save & download chat history
- 🎨 Modern UI with gradient background and styled chat bubbles
- 🐳 Docker-ready for easy deployment on any server
---

## 🛠️ Tech Stack
- **Backend**: Python  
- **Libraries**: LangChain, Groq API, Google Generative AI, FAISS (vector DB)  
- **Frontend**: Streamlit  
- **Others**: OCR for scanned documents, Docker, AWS deployment
---

## ⚙️ Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/satyabharat1207/AI-Documents-QA-Assistant.git
cd AI-Documents-QA-Assistant
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate    # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
Create a `.env` file and add your API keys:  
```Copy example env file
cp .env.example .env   # Linux/Mac
copy .env.example .env # Windows

```

### 5. Run Application
```bash
streamlit run app.py
```

### 6. Docker Deployment
Build Docker image
```bash
docker build -t chatbot-image
```
Run Docker container
Build Docker image
```bash
docker run -d --name myapp -p 5000:8501 chatbot-image
```
Access the app at: http://<your-server-ip>:8501
---

## 📂 Project Structure
```
AI-Documents-QA-Assistant/
├── app.py             # Main Streamlit app
├── requirements.txt   # Dependencies
├── .env.example       # API key template
├── Dockerfile         # Container deployment
├── .gitignore         # Ignore sensitive files
└── README.md          # Project documentation
```

## 🚀 Roadmap

- 🔍 Add support for more file formats (txt, md, pptx)
- 🧠 Improve RAG integration with advanced LLMs (OpenAI / Anthropic / Gemini)
- 🌍 Deploy on Streamlit Cloud, AWS, or Hugging Face Spaces
- 📝 Add user authentication & multi-user support
  
---

## 📜 License
This project is licensed under the MIT License.
