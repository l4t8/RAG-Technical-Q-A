import os
import sys
import shutil
import hashlib
import json
import re
import time
from typing import List, Optional

# Importamos la librería oficial para listar modelos disponibles
import google.generativeai as genai

from dotenv import load_dotenv

# --- Document Loading & Splitting ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Vector Store & Embeddings (Actualizados a v0.2) ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# --- LLMs ---
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Retrievers ---
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# --- Chains & Prompts ---
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# CAMBIO: Usamos 'gemini-pro' como fallback seguro. 
# Si tu consola muestra que tienes acceso a 'gemini-1.5-flash', cámbialo aquí.
GEMINI_MODEL = "gemini-2.5-flash" 

class DocumentProcessor:
    """Handles loading and splitting of raw files."""
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def usual_process_pdf(self, pdf_path="2509.01092v2.pdf"):
        # print(f"Processing {pdf_path}...")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        chunks = self.text_splitter.split_documents(docs)
        # print(f"--> Split into {len(chunks)} chunks.")
        return chunks

class VectorStoreManager:
    """Handles interactions with ChromaDB."""

    def __init__(self, persist_directory=CHROMA_PATH):
        self.persist_directory = persist_directory
        self.embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

        self.db = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_function,
            collection_name="metapaper_database"
        )

    def _generate_chunk_id(self, chunk):
        source = chunk.metadata.get('source', 'unknown_source')
        page = chunk.metadata.get('page', '0')
        content_hash = hashlib.md5(chunk.page_content.encode('utf-8')).hexdigest()[:6]
        return f"{source}:{page}:{content_hash}"

    def add_documents(self, chunks):
        ids = [self._generate_chunk_id(chunk) for chunk in chunks]
        self.db.add_documents(documents=chunks, ids=ids)
        print(f"--> Upserted {len(chunks)} documents.")

    def search(self, query, k=3):
        results = self.db.similarity_search(query, k=k)
        return results
    
    def reset_database(self):
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            print("Database deleted.")
        else:
            print("No database found to delete.")

class RAGPipelines:
    """
    Gestor de pipelines.
    """
    def __init__(self, vector_db, all_chunks, model_name=GEMINI_MODEL):
        # DEBUG: Confirmar qué modelo se está cargando realmente
        print(f"--> LOADING MODEL: {model_name}")
        
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
        self.vector_db = vector_db
        self.all_chunks = all_chunks
        self._init_retrievers()

    def _init_retrievers(self):
        self.dense_retriever = self.vector_db.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 4}
        )
        
        self.bm25_retriever = BM25Retriever.from_documents(self.all_chunks)
        self.bm25_retriever.k = 4
        
        self.hybrid_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.dense_retriever],
            weights=[0.5, 0.5]
        )

    def _get_prompt(self, with_context=True):
        if with_context:
            template = """
            You are a technical assistant analyzing a research paper. Use the following pieces of retrieved context to answer the multiple-choice question.
            
            CONTEXT:
            {context}
            
            QUESTION:
            {question}
            
            OPTIONS:
            A) {option_a}
            B) {option_b}
            C) {option_c}
            D) {option_d}
            
            INSTRUCTIONS:
            1. Select the correct option (A, B, C, or D).
            2. Explain your reasoning briefly.
            3. Cite the 'source' (e.g., filename) and 'page' from the metadata provided in the context if possible.
            4. Return the output in the following JSON format:
            {{
                "answer": "A", 
                "reasoning": "Because...", 
                "source": "filename: page X"
            }}
            """
        else:
            template = """
            You are a technical assistant. Answer the following multiple-choice question based on your internal knowledge.
            
            QUESTION:
            {question}
            
            OPTIONS:
            A) {option_a}
            B) {option_b}
            C) {option_c}
            D) {option_d}
            
            INSTRUCTIONS:
            1. Select the correct option (A, B, C, or D).
            2. Return the output in the following JSON format:
            {{
                "answer": "A", 
                "reasoning": "Because..."
            }}
            """
        return ChatPromptTemplate.from_template(template)

    def get_baseline_pipeline(self):
        prompt = self._get_prompt(with_context=False)
        chain = prompt | self.llm
        return chain

    def get_bm25_pipeline(self):
        prompt = self._get_prompt(with_context=True)
        chain = create_retrieval_chain(self.bm25_retriever, create_stuff_documents_chain(self.llm, prompt))
        return chain

    def get_dense_pipeline(self):
        prompt = self._get_prompt(with_context=True)
        chain = create_retrieval_chain(self.dense_retriever, create_stuff_documents_chain(self.llm, prompt))
        return chain
    
    def get_hybrid_pipeline(self):
        prompt = self._get_prompt(with_context=True)
        chain = create_retrieval_chain(self.hybrid_retriever, create_stuff_documents_chain(self.llm, prompt))
        return chain

class Evaluator:
    def __init__(self, dataset_path):
        self.dataset = self._load_dataset(dataset_path)
    
    def _load_dataset(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data 

    def _clean_json_response(self, text):
        text = text.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'"answer":\s*"([A-D])"', text, re.IGNORECASE)
            if match:
                return {"answer": match.group(1).upper(), "reasoning": text}
            return {"answer": "Error", "reasoning": text}

    def evaluate_pipeline(self, pipeline, pipeline_name, limit=None):
        print(f"\n==========================================")
        print(f"--- Evaluating: {pipeline_name} ---")
        print(f"==========================================")
        
        correct_count = 0
        results = []
        questions_to_process = self.dataset[:limit] if limit else self.dataset
        
        for i, q in enumerate(questions_to_process):
            input_data = {
                "question": q["question"],
                "option_a": q["answers"]["A"],
                "option_b": q["answers"]["B"],
                "option_c": q["answers"]["C"],
                "option_d": q["answers"]["D"]
            }
            
            ground_truth = q["correct_answer"].strip().upper()
            start_time = time.time()
            
            try:
                if pipeline_name == "Baseline":
                    response = pipeline.invoke(input_data)
                    response_text = response.content
                    source_docs = []
                else:
                    rag_input = {"input": q["question"], **input_data}
                    response = pipeline.invoke(rag_input)
                    response_text = response["answer"]
                    source_docs = [doc.metadata for doc in response.get("context", [])]

                parsed_res = self._clean_json_response(response_text)
                predicted_answer = parsed_res.get("answer", "").strip().upper()
                
                is_correct = predicted_answer == ground_truth
                if is_correct:
                    correct_count += 1
                
                elapsed = time.time() - start_time
                results.append({
                    "id": i,
                    "pipeline": pipeline_name,
                    "correct": is_correct,
                    "predicted": predicted_answer,
                    "ground_truth": ground_truth,
                    "latency": elapsed,
                    "retrieved_metadata": source_docs
                })
                
                icon = '✅' if is_correct else '❌'
                print(f"Q{i+1}: {icon} | Pred: {predicted_answer} | Real: {ground_truth} | {elapsed:.2f}s")
                
            except Exception as e:
                print(f"Error en Q{i+1}: {e}")
                results.append({"id": i, "error": str(e)})

        accuracy = (correct_count / len(questions_to_process)) * 100
        print(f"\n>>> Resultados Finales {pipeline_name}: Accuracy {accuracy:.2f}%")
        return accuracy, results

if __name__ == "__main__":
    # --- DEBUGGING DE MODELOS ---
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("⚠️ ADVERTENCIA: No se encontró GOOGLE_API_KEY en variables de entorno.")
        else:
            genai.configure(api_key=api_key)
            print("\n--- CONSULTANDO MODELOS DISPONIBLES EN TU CUENTA ---")
            print("(Si esto falla, tu API Key podría estar mal configurada)")
            available_models = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    # Limpiamos el prefijo 'models/' para mostrar el nombre corto
                    short_name = m.name.replace("models/", "")
                    available_models.append(short_name)
                    print(f" - {short_name}")
            print("----------------------------------------------------\n")
            
            # Verificación automática
            if GEMINI_MODEL not in available_models and f"models/{GEMINI_MODEL}" not in [m.name for m in genai.list_models()]:
                print(f"⚠️ ADVERTENCIA: El modelo configurado '{GEMINI_MODEL}' NO aparece en tu lista.")
                print("   Por favor, cambia la variable GEMINI_MODEL en el código por uno de la lista de arriba.\n")

    except Exception as e:
        print(f"Error listando modelos: {e}")
        print("Continuando con la ejecución...\n")

    # --- INICIO DEL PROGRAMA PRINCIPAL ---
    p = DocumentProcessor()
    
    if os.path.exists("2509.01092v2.pdf"):
         chunks = p.usual_process_pdf()
    else:
         print("⚠️ PDF no encontrado. Asegúrate de que el archivo existe.")
         chunks = []

    M = VectorStoreManager()
    
    if chunks:
        rag_manager = RAGPipelines(vector_db=M.db, all_chunks=chunks)
        
        # 3. Pipelines
        # get_baseline_pipeline
        # get_bm25_pipeline
        # get_dense_pipeline
        chain_hybrid = rag_manager.get_dense_pipeline()
        
        # 4. Evaluar
        if os.path.exists("ModelizaciónEmpresaUCMData.json"):
            evaluator = Evaluator("ModelizaciónEmpresaUCMData.json")
            score_hybrid, _ = evaluator.evaluate_pipeline(chain_hybrid, "Dense", limit=50)
        else:
            print("⚠️ Archivo JSON del dataset no encontrado.")