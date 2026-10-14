from pathlib import Path
import json

import faiss
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from sentence_transformers import SentenceTransformer
from torchvision import transforms
from torchvision.models import vit_b_16

from llm.client import generate
from ml.xai.vit_explain import generate_vit_attention_map


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

XAI_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "xai"

IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "global_test"
    / "COVID-19"
    / "COVID_1005.png"
)

CLASS_NAMES = [
    "COVID-19",
    "NORMAL",
    "PNEUMONIA",
]

IMAGE_SIZE = 224
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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


def load_image(image_path):
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

    tensor = transform(image).unsqueeze(0)

    return image, tensor.to(DEVICE)


def predict(model, image_tensor):
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = F.softmax(logits, dim=1)

    confidence, predicted_index = torch.max(
        probabilities,
        dim=1,
    )

    predicted_class = CLASS_NAMES[
        predicted_index.item()
    ]

    return (
        predicted_class,
        confidence.item(),
        probabilities[0].cpu(),
    )


def load_rag():
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    index = faiss.read_index(
        str(INDEX_PATH)
    )

    metadata = json.loads(
        METADATA_PATH.read_text(
            encoding="utf-8"
        )
    )

    return embedding_model, index, metadata


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

    query_embedding = query_embedding.astype(
        "float32"
    )

    scores, indices = index.search(
        query_embedding,
        min(top_k, index.ntotal),
    )

    results = []

    for score, index_id in zip(
        scores[0],
        indices[0],
    ):
        if index_id == -1:
            continue

        item = metadata[index_id]

        results.append(
            {
                "title": item["source"],
                "excerpt": item["text"],
                "source": item["source"],
                "score": float(score),
            }
        )

    return results


def save_xai_visualization(
    image,
    attention,
    predicted_class,
    confidence,
):
    XAI_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        XAI_OUTPUT_DIR
        / "full_pipeline_vit_xai.png"
    )

    plt.figure(figsize=(8, 8))

    plt.imshow(image, cmap="gray")
    plt.imshow(
        attention.numpy(),
        cmap="jet",
        alpha=0.45,
    )

    plt.title(
        f"FedXAI-Med ViT XAI\n"
        f"{predicted_class} "
        f"({confidence:.2%})"
    )

    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    return output_path


def main():
    print("=" * 70)
    print("FEDXAI-MED FULL PIPELINE")
    print("=" * 70)

    print()
    print(f"Device: {DEVICE}")

    print()
    print("Loading federated ViT...")

    model = load_vit()

    print("Federated ViT loaded successfully.")

    print()
    print(f"Processing: {IMAGE_PATH.name}")

    image, image_tensor = load_image(
        IMAGE_PATH
    )

    predicted_class, confidence, probabilities = (
        predict(
            model,
            image_tensor,
        )
    )

    print()
    print("=== MODEL PREDICTION ===")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")

    print()
    print("Class probabilities:")

    probability_dict = {}

    for class_name, probability in zip(
        CLASS_NAMES,
        probabilities,
    ):
        value = probability.item()

        probability_dict[class_name] = value

        print(
            f"  {class_name}: {value:.4f}"
        )

    print()
    print("=== XAI ===")

    attention, target_class, target_probability, _ = (
        generate_vit_attention_map(
            model,
            image_tensor,
        )
    )

    xai_path = save_xai_visualization(
        image,
        attention,
        predicted_class,
        confidence,
    )

    print(
        f"Attention map generated: {xai_path}"
    )

    print()
    print("=== RAG ===")

    embedding_model, index, metadata = load_rag()

    query = (
        f"Medical decision-support information "
        f"relevant to a chest radiograph predicted "
        f"as {predicted_class}."
    )

    retrieved = retrieve_knowledge(
        embedding_model,
        index,
        metadata,
        query,
    )

    for i, item in enumerate(
        retrieved,
        start=1,
    ):
        print()
        print(
            f"Result {i}: "
            f"{item['source']}"
        )

        print(
            f"Similarity: "
            f"{item['score']:.4f}"
        )

    print()
    print("=== LLM CLINICAL DECISION SUPPORT ===")

    context = {
        "prediction": predicted_class,
        "probability": confidence,
        "explanation": (
            "A ViT attention visualization was "
            "generated for the model prediction. "
            "The attention map is a model attribution "
            "and is not proof of disease."
        ),
        "class_probabilities": probability_dict,
        "retrieved": retrieved,
    }

    question = (
        "Provide a concise research-oriented "
        "clinical decision-support summary based "
        "only on the model output and retrieved "
        "knowledge."
    )

    result = generate(
        question=question,
        context=context,
    )

    print()
    print(result["content"])

    print()
    print(
        f"Provider: {result['provider']}"
    )

    print(
        f"LLM model: {result['model']}"
    )

    print()
    print("=" * 70)
    print("FULL PIPELINE COMPLETE")
    print("=" * 70)

    print()
    print(
        "X-ray → ViT → XAI → RAG → LLM"
    )


if __name__ == "__main__":
    main()