import os
import base64
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY not found in environment variables.")
    exit(1)

genai.configure(api_key=api_key)

pdf_path = "req/secret project mine.pdf"
if not os.path.exists(pdf_path):
    print(f"Error: PDF file not found at {pdf_path}")
    exit(1)

print(f"Reading and base64-encoding PDF ({pdf_path})...")
with open(pdf_path, "rb") as f:
    pdf_bytes = f.read()
b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
print(f"Encoded size: {len(b64_pdf)} characters.")

print("Sending inline PDF payload to Gemini Flash Latest...")
model = genai.GenerativeModel("gemini-flash-latest")

prompt = (
    "You are an expert systems analyst and UX psychologist. "
    "Analyze the attached requirements document in detail. "
    "1. Extract all functional requirements, features, specifications, and rules.\n"
    "2. Extract all UI/UX design guidelines, styling choices, and psychological UX designs.\n"
    "   Specifically inspect terms about user patience, waiting states, and loading states "
    "(e.g., showing 'data is getting fetched' while they wait because they cannot wait).\n"
    "3. Format the final output into a clean, structured Markdown requirements document."
)

try:
    response = model.generate_content([
        {
            "mime_type": "application/pdf",
            "data": b64_pdf
        },
        prompt
    ])
    print("\n=== EXTRACTED REQUIREMENTS ===\n")
    print(response.text)
    
    # Save the output to a text file for reference
    with open("extracted_requirements.md", "w", encoding="utf-8") as f:
        f.write(response.text)
    print("\nSaved requirements to extracted_requirements.md")
    
except Exception as e:
    print(f"Error during content generation: {e}")
