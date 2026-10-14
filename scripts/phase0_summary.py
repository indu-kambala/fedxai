"""Phase 0 helper: print config summary. No training."""

from backend.app.core.yaml_loader import load_yaml


def main() -> None:
    dataset = load_yaml("dataset.yaml")
    print("classes:", dataset["classes"]["names"])
    print("clients:", dataset["clients"]["count"])
    print("central raw images:", dataset["central_server_receives_raw_images"])
    print("phase: 0")


if __name__ == "__main__":
    main()
