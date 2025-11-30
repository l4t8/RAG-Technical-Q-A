import os
import sys
import shutil
import hashlib
import json
import re
import time
from typing import List, Optional
import logging
import pandas as pd
from datetime import datetime


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
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
PDF_PATH = "2509.01092v2.pdf"
DATASET_PATH = "ModelizaciónEmpresaUCMData.json"
GEMINI_MODEL = "gemini-2.0-flash-thinking-exp-01-21"
DATAFRAME_PATH = GEMINI_MODEL + ".csv"

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
        logging.debug(f"--> LOADING MODEL: {model_name}")
        
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

    def _get_document_prompt(self):
        """
        This template defines how each retrieved chunk looks inside the {context} variable.
        We inject the 'source' and 'page' metadata here so the LLM can see it.
        """
        return PromptTemplate(
            input_variables=["page_content", "source", "page"],
            template="Content: {page_content}\nSource Reference: {source}, Page: {page}\n--------------------------------\n"
        )

    def _get_prompt(self, with_context=True):
        if with_context:
            template = """
            You are a technical assistant analyzing a research paper. Use the following pieces of retrieved context to answer the multiple-choice question.
            
            CONTEXT (includes content and source metadata):
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
            3. EXTRACT EVIDENCE: Copy the **exact sentence or phrase** from the context that justifies your answer.
            4. Cite the metadata (Source/Page) associated with that text.
            
            Return the output in the following JSON format:
            {{
                "answer": "A", 
                "reasoning": "Because [explanation]...", 
                "quote": "[Exact text found in the context]",
                "source": "filename: [source] page [page]"
            }}
            """
        else:
            # Baseline prompt (No changes needed)
            template = """
            You are a technical assistant. Answer the following multiple-choice question based on your internal knowledge.
            QUESTION: {question}
            OPTIONS: A) {option_a} B) {option_b} C) {option_c} D) {option_d}
            INSTRUCTIONS: Return JSON: {{"answer": "A", "reasoning": "..."}}
            """
        return ChatPromptTemplate.from_template(template)

    def get_baseline_pipeline(self):
        prompt = self._get_prompt(with_context=False)
        chain = prompt | self.llm
        return chain

    def get_bm25_pipeline(self):
        prompt = self._get_prompt(with_context=True)
        # We pass the document_prompt to include metadata in the context string
        combine_docs_chain = create_stuff_documents_chain(
            self.llm, 
            prompt, 
            document_prompt=self._get_document_prompt()
        )
        chain = create_retrieval_chain(self.bm25_retriever, combine_docs_chain)
        return chain

    def get_dense_pipeline(self):
        prompt = self._get_prompt(with_context=True)
        combine_docs_chain = create_stuff_documents_chain(
            self.llm, 
            prompt, 
            document_prompt=self._get_document_prompt()
        )
        chain = create_retrieval_chain(self.dense_retriever, combine_docs_chain)
        return chain
    
    def get_hybrid_pipeline(self):
        prompt = self._get_prompt(with_context=True)
        combine_docs_chain = create_stuff_documents_chain(
            self.llm, 
            prompt, 
            document_prompt=self._get_document_prompt()
        )
        chain = create_retrieval_chain(self.hybrid_retriever, combine_docs_chain)
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
        
        # Initialize default structure
        result = {
            "answer": "Error", 
            "reasoning": text, 
            "source": "None", 
            "quote": "No quote provided"
        }

        try:
            # Try strict JSON parsing first
            data = json.loads(text)
            result.update(data)
        except json.JSONDecodeError:
            # Regex Fallback
            match_ans = re.search(r'"answer":\s*"([A-D])"', text, re.IGNORECASE)
            if match_ans: result["answer"] = match_ans.group(1).upper()
            
            match_src = re.search(r'"source":\s*"(.*?)"', text, re.IGNORECASE)
            if match_src: result["source"] = match_src.group(1)

            # Extract the quote using regex (non-greedy)
            match_qt = re.search(r'"quote":\s*"(.*?)"', text, re.DOTALL)
            if match_qt: result["quote"] = match_qt.group(1)
            
        return result

    def _format_retrieved_metadata(self, docs):
        if not docs: return "None"
        refs = []
        for d in docs:
            page = d.get("page", "?")
            source = d.get("source", "unknown").split("/")[-1]
            refs.append(f"{source}:p{page}")
        return ", ".join(refs)

    def evaluate_pipeline(self, pipeline, pipeline_name, limit=None):
        logging.info(f"{'='*60}")
        logging.info(f"--- EVALUATING: {pipeline_name} ---")
        logging.info(f"{'='*60}")
        
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
                if "Baseline" in pipeline_name:
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
                
                cited_source = parsed_res.get("source", "-")
                cited_quote = parsed_res.get("quote", "-")

                is_correct = predicted_answer == ground_truth
                if is_correct: correct_count += 1
                
                elapsed = time.time() - start_time
                retrieved_str = self._format_retrieved_metadata(source_docs)
                
                icon = '✅' if is_correct else '❌'
                logging.info(f"Q{i+1}: {icon} | Pred: {predicted_answer} | Real: {ground_truth} | {elapsed:.2f}s")
                
                if "Baseline" not in pipeline_name:
                    # Print the Quote cleanly
                    clean_quote = cited_quote.replace("\n", " ")[:100] + "..." if len(cited_quote) > 100 else cited_quote
                    logging.info(f"   ❝ Quote: \"{clean_quote}\"")
                    logging.info(f"   📄 Cited:  {cited_source}")
                    logging.info(f"   🔎 Pool:   [{retrieved_str}]")
                
                results.append({
                    "question_number": i+1,
                    "pipeline": pipeline_name,
                    "correct": is_correct,
                    "predicted": predicted_answer,
                    "ground_truth": ground_truth,
                    "quote": cited_quote,
                    "source": cited_source,
                    "latency": elapsed
                })
                
            except Exception as e:
                logging.error(f"Error Q{i+1}: {e}")
                results.append({"id": i, "error": str(e)})

        accuracy = (correct_count / len(questions_to_process)) * 100
        logging.info(f"\n>>> FINAL SCORE [{pipeline_name}]: Accuracy {accuracy:.2f}%")
        return accuracy, results

class RAGExperiment:
    """
    Clase principal que orquesta la configuración, chequeo de modelos
    y ejecución de los experimentos.
    """
    def __init__(self, pdf_path=PDF_PATH, dataset_path=DATASET_PATH, model_name=GEMINI_MODEL):
        self.pdf_path = pdf_path
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.chunks = []
        self.rag_manager = None
        
        # Check inicial de la API
        self.check_api_and_models()

    def check_api_and_models(self):
        """Verifica la API Key y muestra los modelos disponibles."""
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                print("⚠️ ADVERTENCIA: No se encontró GOOGLE_API_KEY en variables de entorno.")
                return

            genai.configure(api_key=api_key)
            available_models = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    short_name = m.name.replace("models/", "")
                    available_models.append(short_name)
            
            if self.model_name not in available_models and f"models/{self.model_name}" not in [m.name for m in genai.list_models()]:
                logging.error(f"⚠️ ADVERTENCIA: El modelo configurado '{self.model_name}' NO aparece en tu lista.")
                logging.error(f"   Modelos disponibles: {available_models} ...")
            # else: print(f"✅ Modelo '{self.model_name}' disponible y verificado.")
            # print("----------------------------------------------------\n")
        except Exception as e:
            logging.error(f"Error verificando modelos: {e}")

    def load_resources(self):
        """Carga el PDF y prepara la base de datos vectorial."""
        logging.debug("--> Iniciando carga de recursos...")
        p = DocumentProcessor()
        
        if os.path.exists(self.pdf_path):
             self.chunks = p.usual_process_pdf(self.pdf_path)
        else:
             logging.error(f"⚠️ PDF no encontrado en {self.pdf_path}.")
             self.chunks = []

        # Inicializa Vector Store
        M = VectorStoreManager()
        if self.chunks:
            # Si hay chunks, inicializamos el manager de pipelines
            # Nota: VectorStoreManager ya maneja la persistencia, 
            # así que no necesitamos re-insertar si ya existen, 
            # pero RAGPipelines necesita los chunks para BM25.
            M.add_documents(self.chunks)
            self.rag_manager = RAGPipelines(vector_db=M.db, all_chunks=self.chunks, model_name=self.model_name)
        else:
            logging.error("❌ No se pudieron cargar chunks. No se puede iniciar el sistema RAG.")

    def run(self, pipeline_type="baseline", limit=50):
        """
        Ejecuta la evaluación para el tipo de pipeline seleccionado.
        Opciones: 'baseline', 'bm25', 'dense', 'hybrid'
        """
        if not self.rag_manager:
            self.load_resources()
            
        if not self.rag_manager:
            logging.error("❌ Error: RAG Manager no inicializado.")
            return

        # Selección del pipeline
        match pipeline_type.lower():
            case "baseline":
                pipeline = self.rag_manager.get_baseline_pipeline()
                name = "Baseline (Zero-shot)"
            case "bm25":
                pipeline = self.rag_manager.get_bm25_pipeline()
                name = "BM25 RAG"
            case "dense":
                pipeline = self.rag_manager.get_dense_pipeline()
                name = "Dense RAG"
            case "hybrid":
                pipeline = self.rag_manager.get_hybrid_pipeline()
                name = "Hybrid RAG"
            case _:
                print(f"Tipo de pipeline '{pipeline_type}' no reconocido.")
                return

        # Evaluación
        if os.path.exists(self.dataset_path):
            evaluator = Evaluator(self.dataset_path)
            score, results = evaluator.evaluate_pipeline(pipeline, name, limit=limit)
            return score, results
        else:
            print(f"⚠️ Archivo JSON del dataset no encontrado en {self.dataset_path}.")
            return 0

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    # Instanciamos el experimento
    model = GEMINI_MODEL

    """
    'gemini-2.5-flash' best for processing huge loads
    'gemini-2.5-pro' best for general reasoning
    'gemini-2.0-flash-thinking-exp-01-21' best for hard logic
    """

    experiment = RAGExperiment(model_name=model)

    # "baseline","bm25", "dense", "hybrid"
    pipelines_lst = [
                     "baseline",
                     "bm25", 
                     "dense",
                     "hybrid"
                    ]

    for pipeline_type in pipelines_lst:

        acc, results = experiment.run(pipeline_type=pipeline_type, limit=50)

        [logging.debug(i) for i in results]
        # Ejemplo para correr otro pipeline inmediatamente:
        # experiment.run(pipeline_type="hybrid", limit=50)

        new_results = pd.DataFrame(results)

        new_results['run_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if os.path.exists(DATAFRAME_PATH):
            try:
                existing_results = pd.read_csv(DATAFRAME_PATH)
                global_results = pd.concat([existing_results, new_results], ignore_index=True)

            except pd.errors.EmptyDataError:
                global_results = new_results
        else:
            logging.info(f"🆕 Creating new master log: {DATAFRAME_PATH}")
            global_results = new_results

        # Drop exact duplicates to prevent spamming if you re-run the same code accidentally
        # (We exclude timestamp from this check so re-running the exact same test later counts as new)
        subset_cols = [c for c in new_results.columns if c != 'run_timestamp']
        updated_df = new_results.drop_duplicates(subset=subset_cols)

        global_results.to_csv(DATAFRAME_PATH, index=False)
        logging.info(f"Global results updated. Total rows: {len(updated_df)}")

