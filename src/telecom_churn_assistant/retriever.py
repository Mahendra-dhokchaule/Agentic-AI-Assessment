import json
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv

from .guardrails import mask_personal_data, verified_customer_data
from .sample_data import CUSTOMERS

load_dotenv()

CHROMA_DIRECTORY = Path(__file__).parents[2] / ".chroma"
COLLECTION_NAME = "telecom_customers"
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL",
    "http://127.0.0.1:11434",
)
RETRIEVAL_LIMIT = 3

_vector_store: Chroma | None = None


def get_customer_data(customer_id: str) -> dict | None:
    return CUSTOMERS.get(customer_id)


def _customer_documents() -> tuple[list[Document], list[str]]:
    documents: list[Document] = []
    document_ids: list[str] = []

    for customer_id, customer in CUSTOMERS.items():
        safe_customer = verified_customer_data(customer)
        for section, value in safe_customer.items():
            masked_value = mask_personal_data(value)

            section_json = json.dumps(
                {section: masked_value},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            documents.append(
                Document(
                    page_content=section_json,
                    metadata={
                        "customer_id": customer_id,
                        "section": section,
                    },
                )
            )
            document_ids.append(f"{customer_id}-{section}")

    return documents, document_ids


def _get_vector_store() -> Chroma:
    global _vector_store

    if _vector_store is not None:
        return _vector_store

    documents, document_ids = _customer_documents()
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL,
        ),
        persist_directory=str(CHROMA_DIRECTORY),
    )
    existing = store.get(include=["documents"])
    existing_documents = dict(zip(existing["ids"], existing["documents"]))
    expected_documents = dict(
        zip(document_ids, (doc.page_content for doc in documents))
    )

    if existing_documents != expected_documents:
        if existing_documents:
            store.delete_collection()
            store = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=OllamaEmbeddings(
                    model=EMBEDDING_MODEL,
                    base_url=OLLAMA_BASE_URL,
                ),
                persist_directory=str(CHROMA_DIRECTORY),
            )
        store.add_documents(documents, ids=document_ids)

    _vector_store = store
    return store


def retrieve_customer_context(customer_id: str, query: str) -> list[Document]:
    if not query:
        query = customer_id

    documents = _get_vector_store().similarity_search(
        query,
        k=RETRIEVAL_LIMIT,
        filter={"customer_id": customer_id},
    )
    unexpected_customer_ids = {
        document.metadata.get("customer_id")
        for document in documents
        if document.metadata.get("customer_id") != customer_id
    }
    if unexpected_customer_ids:
        raise RuntimeError(
            "Customer retrieval returned records outside the requested "
            f"customer scope: {sorted(unexpected_customer_ids)}."
        )
    if not documents:
        raise RuntimeError(
            f"No indexed customer context found for {customer_id}."
        )
    return documents
