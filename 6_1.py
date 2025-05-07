# uvicorn sales_service_AI.6_1:app --reload
import os
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import smtplib
from email.message import EmailMessage

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Paths and setup
TICKET_DIR = 'tickets'
VECTORSTORE_PATH = "vectorstore"
EMBEDDING_MODEL_PATH = "models/all-MiniLM-L6-v2"

os.makedirs(TICKET_DIR, exist_ok=True)

# Initialize LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    api_key="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"
)

# Load vector store and embedding model (RAG)
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_PATH,
    model_kwargs={"local_files_only": True}
)
vectorstore = FAISS.load_local(VECTORSTORE_PATH, embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# FastAPI app
app = FastAPI(title="Winger IT Support Chat API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Models
class StartChatRequest(BaseModel):
    ticket_id: str | None = None

class ChatRequest(BaseModel):
    ticket_id: str
    message: str

# Ticket management
def generate_ticket_id():
    date_part = datetime.now().strftime("%Y%m%d")
    base = f"WI{date_part}"
    existing = [
        fname for fname in os.listdir(TICKET_DIR)
        if fname.startswith(f"#{base}") and fname.endswith(".json")
    ]
    next_serial = len(existing) + 1
    return f"#" + base + f"{next_serial:04d}"

def ticket_path(ticket_id):
    return os.path.join(TICKET_DIR, f"{ticket_id}.json")

def load_chat_history(ticket_id):
    path = ticket_path(ticket_id)
    if os.path.exists(path):
        with open(path, 'r') as f:
            messages = json.load(f)
    else:
        messages = []

    return [
        SystemMessage(content="You are a helpful customer support assistant, developed by Winger IT Solutions.")
    ] + [
        HumanMessage(content=m["user"]) if m["role"] == "user"
        else AIMessage(content=m["ai"])
        for m in messages
    ]

def save_chat_history(ticket_id, history):
    messages = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "user": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "ai", "ai": msg.content})
    with open(ticket_path(ticket_id), 'w') as f:
        json.dump(messages, f, indent=2)

# Routes
@app.post("/start_chat")
def start_chat(req: StartChatRequest):
    ticket_id = req.ticket_id or generate_ticket_id()
    
    if not os.path.exists(ticket_path(ticket_id)):
        chat_history = [SystemMessage(content="You are a helpful customer support assistant, developed by Winger IT Solutions.")]
        save_chat_history(ticket_id, chat_history)
    else:
        chat_history = load_chat_history(ticket_id)
    
    history_dicts = []
    for msg in chat_history[1:]:
        if isinstance(msg, HumanMessage):
            history_dicts.append({"role": "user", "user": msg.content})
        elif isinstance(msg, AIMessage):
            history_dicts.append({"role": "ai", "ai": msg.content})
    
    return {"ticket_id": ticket_id, "chat": history_dicts}

@app.post("/chat")
def chat(req: ChatRequest):
    ticket_id = req.ticket_id
    if not os.path.exists(ticket_path(ticket_id)):
        raise HTTPException(status_code=404, detail="Ticket ID not found")

    # Step 1: Load history and append user message
    chat_history = load_chat_history(ticket_id)
    chat_history.append(HumanMessage(content=req.message))

    # Check for escalation trigger
#################################

    # if req.message.strip().lower() == "contact human":
    #     send_email_with_ticket(ticket_id, chat_history)
    #     chat_history.append(AIMessage(content="Thank you. A Human Assistant will contact you within 24 to 48 hours."))
    #     save_chat_history(ticket_id, chat_history)
    #     return {
    #         "response": "Thank you. A Human Assistant will contact you within 24 to 48 hours.",
    #         "retrieved_docs": []
    #     }

    if req.message.strip().lower() == "contact human":
    # Load the full chat history
        chat_history = load_chat_history(ticket_id)

    # Format history as plain conversation text
        convo_text = ""
        for msg in chat_history:
            if isinstance(msg, HumanMessage):
                convo_text += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
             convo_text += f"Assistant: {msg.content}\n"

    # Ask LLM to summarize the conversation
        summary_prompt = f"Summarize this customer support conversation for a human assistant:\n\n{convo_text}"
        summary = llm.invoke(summary_prompt).content.strip()

    # Compose and send email with summary + attached JSON
        subject = f"Ticket Escalation: {ticket_id}"
        body = f"""
📬 Escalation Notice - Ticket ID: {ticket_id}

A user has requested human assistance. Here is a summary of the conversation so far:

{summary}

The complete chat log is attached.
"""
    # Save latest message
        chat_history.append(AIMessage(content="Thank you. A Human Assistant will contact you within 24 to 48 hours."))
        save_chat_history(ticket_id, chat_history)

    # Send email
        send_email_with_attachment(
            subject=subject,
            body=body,
            to="faisal.shaikh@wingerit.in",
            filename=f"{ticket_id}.json",
            file_content=json.dumps([msg.dict() for msg in chat_history], indent=2)
        )

        return {
        "response": "Thank you. A Human Assistant will contact you within 24 to 48 hours.",
        "retrieved_docs": []
    }



#################################

    # Step 2: Get context using RAG (FAISS + embeddings)
    retrieved_docs = retriever.invoke(req.message)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # Step 3: Add RAG context as a system message (temporary, not saved)
    chat_with_context = chat_history.copy()
    chat_with_context.insert(1, SystemMessage(content=f"Relevant context:\n{context}"))

    # Step 4: Invoke LLM
    response = llm.invoke(chat_with_context)

    # Step 5: Save history without the temporary context message
    chat_history.append(response)
    save_chat_history(ticket_id, chat_history)

    # Step 6: Return response + docs for display
    return {
        "response": response.content,
        "retrieved_docs": [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            } for doc in retrieved_docs
        ]
    }


# "mado bkup lbbp hfwj"

def send_email_with_attachment(subject, body, to, filename, file_content):
    sender = "shaikfa66@gmail.com"
    password = "mado bkup lbbp hfwj"  # Make sure you're using an App Password

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to
    msg.set_content(body)

    msg.add_attachment(
        file_content.encode("utf-8"),
        maintype="application",
        subtype="json",
        filename=filename
    )

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(sender, password)
        smtp.send_message(msg)



from fastapi.responses import JSONResponse
import os

@app.get("/tickets/")
def list_tickets():
    files = os.listdir("tickets")
    json_files = [f for f in files if f.endswith(".json")]
    return JSONResponse(content=json_files)



def send_email_with_ticket(ticket_id, history):
    sender_email = "shaikfa66@gmail.com"
    receiver_email = "faisal.shaikh@wingerit.in"
    subject = f"Human Support Requested - Ticket {ticket_id}"
    password = "mado bkup lbbp hfwj"  # ⚠️ Do not use real password in plaintext

    summary_lines = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            summary_lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            summary_lines.append(f"AI: {msg.content}")
    
    summary_text = "\n".join(summary_lines)

    # Create email
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg.set_content(f"A human assistant was requested.\n\nTicket ID: {ticket_id}\n\nSummary:\n{summary_text}")

    # Attach the JSON file
    ticket_file_path = ticket_path(ticket_id)
    with open(ticket_file_path, 'rb') as f:
        ticket_json = f.read()
    msg.add_attachment(ticket_json, maintype='application', subtype='json', filename=f"{ticket_id}.json")

    # Send email
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(sender_email, password)
        smtp.send_message(msg)
