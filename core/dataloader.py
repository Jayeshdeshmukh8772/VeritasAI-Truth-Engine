"""
VeritasAI Ingestion & Lightweight Vector RAG Engine.
Supports scraping URLs and reading files, chunking text via recursive character or
semantic distance metrics, and performing vector search via numpy-based cosine similarity.
Uses the already loaded SentenceTransformer from HallucinationDetector to conserve RAM.
"""

import re
import numpy as np
import httpx
from typing import List, Dict, Optional, Tuple


class TextChunker:
    """Methods for splitting large text sources into manageable grounding context blocks."""

    @staticmethod
    def recursive_character_split(text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
        """
        Split text recursively by paragraph, sentence, then word boundaries.
        
        Args:
            text: Raw document text
            chunk_size: Target characters per chunk
            overlap: Character overlap between contiguous chunks
            
        Returns:
            List of text chunk strings
        """
        if len(text) <= chunk_size:
            return [text]

        # Candidate splitters in order of granularity
        splitters = ["\n\n", "\n", ". ", " ", ""]
        chunks = []
        
        # A simple sliding window recursive splitting logic
        words = text.split(" ")
        current_chunk = []
        current_len = 0
        
        for word in words:
            word_len = len(word) + 1
            if current_len + word_len > chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                # Retain overlap words
                overlap_words = current_chunk[-max(1, int(overlap / 6)):] if len(current_chunk) > 5 else []
                current_chunk = overlap_words + [word]
                current_len = sum(len(w) + 1 for w in current_chunk)
            else:
                current_chunk.append(word)
                current_len += word_len
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return [c for c in chunks if len(c.strip()) > 30]

    @staticmethod
    def semantic_split(text: str, encoder, threshold: float = 0.35) -> List[str]:
        """
        Split text at sentence boundaries where semantic topic drift is detected.
        
        Args:
            text: Raw document text
            encoder: Shared SentenceTransformer instance
            threshold: Cosine distance boundary to trigger chunk break
            
        Returns:
            List of text chunk strings
        """
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        if len(sentences) <= 2:
            return sentences

        # Encode all sentences
        embeddings = encoder.encode(sentences)
        normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        chunks = []
        current_chunk_sentences = [sentences[0]]
        
        # Compare cosine distance between adjacent sentences: 1 - cosine_sim
        for i in range(len(sentences) - 1):
            sim = np.dot(normalized[i], normalized[i+1])
            dist = 1.0 - sim
            
            if dist > threshold and len(" ".join(current_chunk_sentences)) > 150:
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = [sentences[i+1]]
            else:
                current_chunk_sentences.append(sentences[i+1])
                
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))
            
        return chunks


class LightweightVectorStore:
    """In-memory, numpy-accelerated vector store to prevent heavyweight database RAM overhead."""

    def __init__(self) -> None:
        self.documents: List[Dict] = []  # Contains dicts: {"text": str, "source": str, "embedding": np.ndarray}

    def add_documents(self, chunks: List[str], source: str, encoder) -> None:
        """
        Encode and store text chunks with metadata.
        
        Args:
            chunks: List of split text strings
            source: Label indicating origin (e.g. filename, URL)
            encoder: Shared SentenceTransformer
        """
        if not chunks:
            return
            
        embeddings = encoder.encode(chunks)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Avoid division by zero
        normalized_embeddings = embeddings / norms
        
        for text, emb in zip(chunks, normalized_embeddings):
            self.documents.append({
                "text": text,
                "source": source,
                "embedding": emb
            })

    def retrieve(self, query: str, encoder, k: int = 3) -> List[Tuple[str, str, float]]:
        """
        Retrieve top-k matching documents using cosine similarity.
        
        Args:
            query: User search query
            encoder: Shared SentenceTransformer
            k: Count of documents to return
            
        Returns:
            List of tuples: (text, source, similarity_score)
        """
        if not self.documents:
            return []

        # Encode query
        q_emb = encoder.encode([query])[0]
        q_norm = np.linalg.norm(q_emb)
        if q_norm == 0:
            return []
        q_emb = q_emb / q_norm

        results = []
        for doc in self.documents:
            sim = float(np.dot(doc["embedding"], q_emb))
            results.append((doc["text"], doc["source"], sim))
            
        # Sort by similarity descending
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:k]

    def clear(self) -> None:
        """Reset the vector database."""
        self.documents = []


class DataLoader:
    """Ingestion controller for extracting text contents from files or URLs."""

    @staticmethod
    async def load_url(url: str) -> str:
        """Fetch URL content and extract clean text using standard python parsers."""
        headers = {"User-Agent": "VeritasAI RAG Bot/1.0"}
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            html = resp.text
            
            # Remove scripts, styles and get text (minimal regex-based HTML clean)
            clean = re.sub(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', '', html, flags=re.I)
            clean = re.sub(r'<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>', '', clean, flags=re.I)
            clean = re.sub(r'<[^>]+>', ' ', clean)
            clean = re.sub(r'\s+', ' ', clean).strip()
            
            # Truncate to a reasonable character count to protect RAM
            return clean[:30000]

    @staticmethod
    def load_file(file_content: bytes, file_name: str) -> str:
        """Load text from raw file uploads (Text, Markdown, or PDF)."""
        ext = file_name.split(".")[-1].lower()
        
        if ext in ["txt", "md", "markdown", "py", "json"]:
            return file_content.decode("utf-8", errors="ignore")
        elif ext == "pdf":
            try:
                import pypdf
                import io
                reader = pypdf.PdfReader(io.BytesIO(file_content))
                text_parts = []
                for page in reader.pages[:15]:  # Limit pages to prevent RAM spikes
                    t = page.extract_text()
                    if t:
                        text_parts.append(t)
                return "\n".join(text_parts)
            except ImportError:
                # Basic fallback if pypdf is missing: try to extract printable text
                text_content = file_content.decode("ascii", errors="ignore")
                # Strip typical binary headers/trailers
                cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', ' ', text_content)
                return f"[PDF Ingest Note: pypdf not installed. Basic extraction used]\n\n" + cleaned[:5000]
        else:
            return file_content.decode("utf-8", errors="ignore")
