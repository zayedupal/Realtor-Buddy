import os
from typing import List, Optional

import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from vertexai.vision_models import MultiModalEmbeddingResponse, MultiModalEmbeddingModel, Image
from langchain_google_vertexai import VertexAIEmbeddings

import requests
from tqdm import tqdm

from dotenv import load_dotenv

load_dotenv()

gemini_text_embedding_function = VertexAIEmbeddings(model_name=os.environ['TEXT_EMBEDDING_MODEL'])

def embed_text(
        texts: List[str] = ["banana muffins? ", "banana bread? banana muffins?"],
        task: str = "RETRIEVAL_DOCUMENT",
        batch_size: int = 20
) -> List[List[float]]:
    """Embeds texts with a pre-trained, foundational model.
    There's a limit of 200 texts per call """
    model = TextEmbeddingModel.from_pretrained(os.environ["TEXT_EMBEDDING_MODEL"])
    inputs = [TextEmbeddingInput(text, task) for text in texts]

    texts_count = len(texts)
    total_iterations = texts_count//batch_size
    leftover = texts_count % batch_size
    embeddings = []

    for i in tqdm(range(total_iterations), desc="generating text embeddings"):
        cur_inputs = inputs[i*batch_size:(i+1)*batch_size]
        cur_embeddings = model.get_embeddings(cur_inputs)
        embeddings.extend([emb.values for emb in cur_embeddings])

    if leftover:
        cur_inputs = inputs[total_iterations * batch_size:]
        cur_embeddings = model.get_embeddings(cur_inputs)
        embeddings.extend([emb.values for emb in cur_embeddings])
    return embeddings

def embed_text_multimodal(
        text,
) -> List[List[float]]:
    """Embeds texts with a pre-trained, foundational model.
    There's a limit of 200 texts per call """
    model = MultiModalEmbeddingModel.from_pretrained(os.environ["MULTIMODAL_EMBEDDING_MODEL"])

    embeddings = model.get_embeddings(
        contextual_text=text,
        dimension=1408
    )

    return embeddings.text_embedding


def get_image_embeddings_from_url(
        image_path: str,
        contextual_text: Optional[str] = None,
) -> MultiModalEmbeddingResponse:
    """Example of how to generate multimodal embeddings from image and text.

    Args:
        project_id: Google Cloud Project ID, used to initialize vertexai
        location: Google Cloud Region, used to initialize vertexai
        image_path: Path to image (local or Google Cloud Storage) to generate embeddings for.
        contextual_text: Text to generate embeddings for.
    """

    vertexai.init(project=os.environ["GCP_PROJECT_ID"], location=os.environ["GCP_PROJECT_LOC"])

    model = MultiModalEmbeddingModel.from_pretrained(os.environ["MULTIMODAL_EMBEDDING_MODEL"])
    url_content = requests.get(image_path).content

    image = Image(url_content)

    embeddings = model.get_embeddings(
        image=image,
        contextual_text=contextual_text,
        dimension=1408
    )
    # print(f"Image Embedding: {embeddings.image_embedding}")
    # print(f"Text Embedding: {embeddings.text_embedding}")

    return image, embeddings.image_embedding

def get_image_embeddings_from_file(
        image_path: str,
        contextual_text: Optional[str] = None,
):
    """Example of how to generate multimodal embeddings from image and text.

    Args:
        project_id: Google Cloud Project ID, used to initialize vertexai
        location: Google Cloud Region, used to initialize vertexai
        image_path: Path to image (local or Google Cloud Storage) to generate embeddings for.
        contextual_text: Text to generate embeddings for.
    """

    vertexai.init(project=os.environ["GCP_PROJECT_ID"], location=os.environ["GCP_PROJECT_LOC"])

    model = MultiModalEmbeddingModel.from_pretrained(os.environ["MULTIMODAL_EMBEDDING_MODEL"])
    image = Image.load_from_file(image_path)

    embeddings = model.get_embeddings(
        image=image,
        contextual_text=contextual_text,
        dimension=1408
    )
    # print(f"Image Embedding: {embeddings.image_embedding}")
    # print(f"Text Embedding: {embeddings.text_embedding}")

    return image, embeddings.image_embedding

def get_multimodal_embeddings_from_url(
        image_path: str,
        contextual_text: Optional[str] = None,
) -> MultiModalEmbeddingResponse:
    """Example of how to generate multimodal embeddings from image and text.

    Args:
        project_id: Google Cloud Project ID, used to initialize vertexai
        location: Google Cloud Region, used to initialize vertexai
        image_path: Path to image (local or Google Cloud Storage) to generate embeddings for.
        contextual_text: Text to generate embeddings for.
    """

    vertexai.init(project=os.environ["GCP_PROJECT_ID"], location=os.environ["GCP_PROJECT_LOC"])

    model = MultiModalEmbeddingModel.from_pretrained(os.environ["MULTIMODAL_EMBEDDING_MODEL"])
    url_content = requests.get(image_path).content

    image = Image(url_content)

    embeddings = model.get_embeddings(
        image=image,
        contextual_text=contextual_text,
        dimension=1408
    )
    # print(f"Image Embedding: {embeddings.image_embedding}")
    print(f"Text Embedding: {embeddings.text_embedding}")

    return image, embeddings.image_embedding, embeddings.text_embedding

def get_multimodal_embeddings_from_file(
        image_path: str,
        contextual_text: Optional[str] = None,
) -> MultiModalEmbeddingResponse:
    """Example of how to generate multimodal embeddings from image and text.

    Args:
        project_id: Google Cloud Project ID, used to initialize vertexai
        location: Google Cloud Region, used to initialize vertexai
        image_path: Path to image (local or Google Cloud Storage) to generate embeddings for.
        contextual_text: Text to generate embeddings for.
    """

    vertexai.init(project=os.environ["GCP_PROJECT_ID"], location=os.environ["GCP_PROJECT_LOC"])

    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
    image = Image.load_from_file(image_path)

    embeddings = model.get_embeddings(
        image=image,
        contextual_text=contextual_text,
        dimension=1408
    )
    # print(f"Image Embedding: {embeddings.image_embedding}")
    # print(f"Text Embedding: {embeddings.text_embedding}")

    return image, embeddings.image_embedding, embeddings.text_embedding

def get_image_embeddings_from_upload(
        image_path: str,
        contextual_text: Optional[str] = None,
) -> MultiModalEmbeddingResponse:
    """Example of how to generate multimodal embeddings from image and text.

    Args:
        project_id: Google Cloud Project ID, used to initialize vertexai
        location: Google Cloud Region, used to initialize vertexai
        image_path: Path to image (local or Google Cloud Storage) to generate embeddings for.
        contextual_text: Text to generate embeddings for.
    """

    vertexai.init(project=os.environ["GCP_PROJECT_ID"], location=os.environ["GCP_PROJECT_LOC"])

    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
    image = Image(image_path)

    embeddings = model.get_embeddings(
        image=image,
        contextual_text=contextual_text,
        dimension=1408
    )
    # print(f"Image Embedding: {embeddings.image_embedding}")
    # print(f"Text Embedding: {embeddings.text_embedding}")

    return image, embeddings.image_embedding
