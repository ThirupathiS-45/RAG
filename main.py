from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai
import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import json
from datetime import datetime
import PyPDF2
import io

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize ChromaDB for RAG - Timetable Memory System
# Store database persistently in the time_manager folder
import os
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "chromadb_storage")

# Create persistent ChromaDB client
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Collection for storing successful timetables and user patterns
try:
    timetable_memory = chroma_client.get_or_create_collection(
        name="timetable_memory",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    )
    
    # Collection for study materials (optional)
    study_materials = chroma_client.get_or_create_collection(
        name="study_materials",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    )
    print(f"✅ ChromaDB initialized successfully at: {CHROMA_DB_PATH}")
except Exception as e:
    # Fallback to default embedding if sentence-transformers fails
    print(f"Warning: Using default embeddings due to: {e}")
    timetable_memory = chroma_client.get_or_create_collection(name="timetable_memory")
    study_materials = chroma_client.get_or_create_collection(name="study_materials")
    print(f"✅ ChromaDB initialized with default embeddings at: {CHROMA_DB_PATH}")

# FastAPI app
app = FastAPI(title="Smart Time Management Assistant with RAG")

# Pydantic model for input
class TaskInput(BaseModel):
    query: str  # Natural language input like "tomorrow maths exam, day after tomorrow project submission"

@app.post("/generate_timetable/")
async def generate_timetable(task_input: TaskInput):
    try:
        # Retrieve relevant timetable patterns and past successful schedules
        retrieved_context = timetable_memory.query(
            query_texts=[task_input.query],
            n_results=5
        )
        
        # Prepare context from retrieved documents
        context_info = ""
        if retrieved_context['documents'] and retrieved_context['documents'][0]:
            context_info = "Here are similar past timetables and patterns that worked well for you:\n\n"
            for doc in retrieved_context['documents'][0]:
                context_info += f"- {doc}\n"
            context_info += "\nUse these insights to create a better personalized schedule.\n"
        
        prompt = f"""
        You are a smart AI time management assistant. Today is October 6, 2025.
        
        The user has given you this natural language input: "{task_input.query}"
        
        {context_info}
        
        Your task:
        1. Parse the input to identify tasks and their relative dates (tomorrow, day after tomorrow, next week, etc.)
        2. Convert relative dates to actual dates starting from today (October 6, 2025)
        3. Create a detailed hour-by-hour timetable from today until all deadlines are met
        4. Schedule study/preparation time for each task based on its complexity and deadline
        5. Include breaks, meals, and realistic time allocations
        6. Prioritize tasks that are due sooner
        7. Use the retrieved context to provide personalized recommendations
        
        **IMPORTANT: Return the timetable in TABLE FORMAT using markdown tables.**
        
        Format each day like this:
        
        ## Monday, October 6, 2025
        | Time | Activity | Task/Subject | Notes |
        |------|----------|--------------|-------|
        | 7:00 AM - 8:00 AM | Morning Routine | - | Breakfast, get ready |
        | 8:00 AM - 10:00 AM | Study Session | Maths Quiz Prep | Focus on weak areas |
        | 10:00 AM - 10:15 AM | Break | - | Short break |
        | 10:15 AM - 12:00 PM | Study Session | Maths Quiz Prep | Practice problems |
        
        Continue this table format for each day until all deadlines are met.
        Include realistic time for meals, breaks, and sleep.
        Make it actionable and specific with clear time slots.
        """

        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        
        # Store this timetable for future learning and reference
        timetable_memory.add(
            documents=[f"Query: {task_input.query}\nGenerated Timetable: {response.text[:500]}..."],
            metadatas=[{
                "type": "generated_timetable", 
                "query": task_input.query,
                "date": datetime.now().isoformat(),
                "rating": 0  # Will be updated when user provides feedback
            }],
            ids=[f"timetable_{datetime.now().timestamp()}"]
        )

        return {"timetable": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_study_material/")
async def upload_study_material(file: UploadFile = File(...), subject: str = "general"):
    """Upload PDF study materials, syllabi, or notes to enhance RAG context"""
    try:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Read PDF content
        content = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        
        # Extract text from all pages
        text_content = ""
        for page in pdf_reader.pages:
            text_content += page.extract_text() + "\n"
        
        # Split into chunks for better retrieval
        chunks = [text_content[i:i+1000] for i in range(0, len(text_content), 800)]
        
        # Store in ChromaDB
        for i, chunk in enumerate(chunks):
            study_materials.add(
                documents=[chunk],
                metadatas=[{
                    "type": "study_material",
                    "subject": subject,
                    "filename": file.filename,
                    "chunk": i,
                    "upload_date": datetime.now().isoformat()
                }],
                ids=[f"{file.filename}_{subject}_{i}_{datetime.now().timestamp()}"]
            )
        
        return {"message": f"Successfully uploaded {file.filename} with {len(chunks)} chunks", "subject": subject}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_study_note/")
async def add_study_note(note: str, subject: str = "general", topic: str = ""):
    """Add text-based study notes or important information"""
    try:
        study_materials.add(
            documents=[note],
            metadatas=[{
                "type": "study_note",
                "subject": subject,
                "topic": topic,
                "date": datetime.now().isoformat()
            }],
            ids=[f"note_{subject}_{datetime.now().timestamp()}"]
        )
        
        return {"message": "Study note added successfully", "subject": subject, "topic": topic}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search_materials/")
async def search_materials(query: str, n_results: int = 5):
    """Search through stored study materials and notes"""
    try:
        results = study_materials.query(
            query_texts=[query],
            n_results=n_results
        )
        
        return {
            "query": query,
            "results": results['documents'][0] if results['documents'] else [],
            "metadata": results['metadatas'][0] if results['metadatas'] else []
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rate_timetable/")
async def rate_timetable(timetable_id: str, rating: int, feedback: str = ""):
    """Rate a timetable's effectiveness (1-5 stars) for future learning"""
    try:
        if rating < 1 or rating > 5:
            raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
        
        # Store feedback for future timetable generation
        timetable_memory.add(
            documents=[f"Timetable feedback: Rating {rating}/5. {feedback}"],
            metadatas=[{
                "type": "timetable_feedback",
                "timetable_id": timetable_id,
                "rating": rating,
                "feedback": feedback,
                "date": datetime.now().isoformat()
            }],
            ids=[f"feedback_{timetable_id}_{datetime.now().timestamp()}"]
        )
        
        return {"message": f"Thank you! Rated {rating}/5 stars", "timetable_id": timetable_id}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_user_preference/")
async def add_user_preference(preference_type: str, preference_value: str, description: str = ""):
    """Store user preferences like 'best_study_time': '9AM-11AM' or 'break_duration': '15 minutes'"""
    try:
        timetable_memory.add(
            documents=[f"User preference: {preference_type} = {preference_value}. {description}"],
            metadatas=[{
                "type": "user_preference",
                "preference_type": preference_type,
                "preference_value": preference_value,
                "description": description,
                "date": datetime.now().isoformat()
            }],
            ids=[f"pref_{preference_type}_{datetime.now().timestamp()}"]
        )
        
        return {"message": "Preference saved", "type": preference_type, "value": preference_value}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_memory_stats/")
async def get_memory_stats():
    """Check what's stored in ChromaDB memory"""
    try:
        # Get count of documents in each collection
        timetable_count = timetable_memory.count()
        materials_count = study_materials.count()
        
        # Get recent timetables
        recent_timetables = timetable_memory.query(
            query_texts=["recent timetable"],
            n_results=3
        )
        
        return {
            "timetable_memory_count": timetable_count,
            "study_materials_count": materials_count,
            "recent_timetables": recent_timetables['documents'][0] if recent_timetables['documents'] else [],
            "recent_metadata": recent_timetables['metadatas'][0] if recent_timetables['metadatas'] else [],
            "storage_location": "In-memory ChromaDB (persistent across app restarts)"
        }
    
    except Exception as e:
        return {"error": str(e), "message": "ChromaDB collections might not be initialized yet"}

@app.get("/test_memory/")
async def test_memory():
    """Test if RAG is remembering previous timetables"""
    try:
        # Search for any stored timetables
        all_timetables = timetable_memory.query(
            query_texts=["timetable schedule exam"],
            n_results=10
        )
        
        # Search for user preferences
        preferences = timetable_memory.query(
            query_texts=["user preference study time"],
            n_results=5
        )
        
        return {
            "memory_test_results": {
                "stored_timetables": len(all_timetables['documents'][0]) if all_timetables['documents'] else 0,
                "stored_preferences": len(preferences['documents'][0]) if preferences['documents'] else 0,
                "sample_timetables": all_timetables['documents'][0][:2] if all_timetables['documents'] else [],
                "sample_preferences": preferences['documents'][0] if preferences['documents'] else []
            },
            "status": "RAG memory is working" if all_timetables['documents'] and all_timetables['documents'][0] else "No timetables stored yet"
        }
    
    except Exception as e:
        return {"error": str(e)}
