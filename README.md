🧠 Sistema Experimental RAG: Análisis del Framework REFRAG

Este proyecto implementa un sistema de Retrieval-Augmented Generation (RAG) diseñado para analizar documentos técnicos complejos (específicamente el paper "REFRAG: Rethinking RAG based Decoding" de Meta) y evaluar diferentes estrategias de recuperación de información.

El objetivo principal es comparar empíricamente cómo diferentes métodos de recuperación (Keyword, Semántico e Híbrido) impactan en la capacidad de un LLM para responder preguntas de opción múltiple con alta precisión.

🚀 Características

Ingesta de Documentos: Procesamiento de PDFs técnicos utilizando PyPDFLoader y RecursiveCharacterTextSplitter para una segmentación inteligente.

Vector Store: Implementación persistente con ChromaDB.

Modelos:

LLM: Google Gemini 1.5 Flash (vía langchain-google-genai).

Embeddings: HuggingFace all-MiniLM-L6-v2.

Arquitectura Modular: Soporte para 4 pipelines de evaluación distintos:

(A) Baseline: LLM sin contexto (Zero-shot).

(B) BM25: Búsqueda basada en palabras clave (Keyword Search).

(C) Dense Retrieval: Búsqueda semántica por similitud vectorial.

(D) Hybrid RAG: Ensemble Retriever (BM25 + Dense) con pesos ajustables.

Evaluación Automatizada: Sistema de evaluación que compara las predicciones contra un ground truth en formato JSON, midiendo precisión y latencia.

🛠️ Requisitos Previos

Python 3.11+

Conda (Recomendado para la gestión de entornos)

Una API Key de Google AI Studio (para usar Gemini).

📦 Instalación

Clona el repositorio:

git clone [https://github.com/tu-usuario/tu-repositorio.git](https://github.com/tu-usuario/tu-repositorio.git)
cd tu-repositorio


Configura el entorno:
Hemos preparado un archivo environment.yml para una instalación limpia y compatible multiplataforma.

conda env create -f environment.yml
conda activate langchain_env


Variables de Entorno:
Crea un archivo .env en la raíz del proyecto y añade tu clave API:

GOOGLE_API_KEY=tu_clave_api_aqui


⚙️ Estructura del Proyecto

├── chroma_db/                  # Base de datos vectorial (se genera automáticamente)
├── rag_system.py               # Script principal (Lógica RAG y Evaluación)
├── ModelizaciónEmpresaUCMData.json  # Dataset de preguntas y respuestas
├── 2509.01092v2.pdf            # Paper de investigación (Input)
├── environment.yml             # Dependencias del proyecto
└── README.md                   # Documentación


▶️ Uso

El script principal gestiona tanto la ingesta de documentos como la evaluación.

Ejecutar la evaluación:
Por defecto, el script está configurado para evaluar el pipeline Híbrido.

python rag_system.py


Cambiar de Estrategia:
Para probar otros métodos (BM25, Dense, Baseline), edita las líneas finales de rag_system.py:

# En el bloque if __name__ == "__main__":

# Para usar BM25:
chain = rag_manager.get_bm25_pipeline()
score, _ = evaluator.evaluate_pipeline(chain, "BM25 RAG")

# Para usar Dense Retrieval:
# chain = rag_manager.get_dense_pipeline()


📊 Metodología de Evaluación

El sistema utiliza un conjunto de datos (ModelizaciónEmpresaUCMData.json) que contiene preguntas difíciles de opción múltiple sobre el paper. El evaluador:

Recupera el contexto relevante (o nada, en el caso del Baseline).

Construye un prompt con instrucciones estrictas.

Solicita al LLM la respuesta y la cita de la fuente.

Compara la respuesta (A, B, C, D) con la correcta y calcula el Accuracy.

📚 Tecnologías Utilizadas

LangChain v0.2 - Orquestación de LLMs.

Chroma - Base de datos vectorial open-source.

Google Gemini API - Modelo Generativo.

HuggingFace - Modelos de Embeddings.

Rank-BM25 - Algoritmo de ranking probabilístico.

📄 Referencia

El documento analizado en este proyecto es:

REFRAG: Rethinking RAG based Decoding (Meta SuperIntelligence Labs).

Hecho con ❤️ usando Python y LangChain.
