from pathlib import Path
import json

import faiss
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from torchvision import transforms
from torchvision.models import vit_b_16


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "federated"
    / "fedavg_vit"
    / "vit_fedavg_global_model.pt"
)

INDEX_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "rag"
    / "medical_knowledge.faiss"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "rag"
    / "metadata.json"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "COVID-19",
    "NORMAL",
    "PNEUMONIA",
]

IMAGE_SIZE = 224
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_vit():
    model = vit_b_16(weights=None)

    model.heads.head = torch.nn.Linear(
        model.heads.head.in_features,
        len(CLASS_NAMES),
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    return model


def load_rag():
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    index = faiss.read_index(str(INDEX_PATH))

    metadata = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )

    return embedding_model, index, metadata


def predict(model, image_path):
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)

    confidence, predicted_index = torch.max(
        probabilities,
        dim=1,
    )

    predicted_class = CLASS_NAMES[predicted_index.item()]

    return (
        predicted_class,
        confidence.item(),
        probabilities[0].cpu().numpy(),
    )


def retrieve_knowledge(
    embedding_model,
    index,
    metadata,
    query,
    top_k=3,
):
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embedding = query_embedding.astype("float32")

    scores, indices = index.search(
        query_embedding,
        min(top_k, index.ntotal),
    )

    results = []

    for score, index_id in zip(scores[0], indices[0]):
        if index_id == -1:
            continue

        results.append(
            {
                "score": float(score),
                "source": metadata[index_id]["source"],
                "text": metadata[index_id]["text"],
            }
        )

    return results


def main():
    image_path = (
        PROJECT_ROOT
        / "data"
        / "global_test"
        / "COVID-19"
        / "COVID_1005.png"
    )

    print(f"Device: {DEVICE}")
    print()
    print("Loading Federated ViT...")
    
    model = load_vit()

    print("Federated ViT loaded successfully.")
    print()

    print(f"Processing X-ray: {image_path.name}")
    print()

    predicted_class, confidence, probabilities = predict(
        model,
        image_path,
    )

    print("=== FEDXAI PREDICTION ===")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    print()

    print("Class probabilities:")

    for class_name, probability in zip(
        CLASS_NAMES,
        probabilities,
    ):
        print(
            f"  {class_name}: {probability:.4f}"
        )

    print()
    print("Loading RAG knowledge base...")

    embedding_model, index, metadata = load_rag()

    query = (
        f"Medical decision-support information relevant to "
        f"a chest radiograph predicted as {predicted_class}."
    )

    results = retrieve_knowledge(
        embedding_model,
        index,
        metadata,
        query,
        top_k=3,
    )

    print()
    print("=== RAG RETRIEVAL ===")

    for i, result in enumerate(results, start=1):
        print()
        print(f"Result {i}")
        print(f"Similarity: {result['score']:.4f}")
        print(f"Source: {result['source']}")
        print("Knowledge:")
        print(result["text"])

    print()
    print("=== PIPELINE COMPLETE ===")
    print("X-ray → Federated ViT → Prediction → RAG")


if __name__ == "__main__":
    main()