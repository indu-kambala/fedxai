from backend.app.core.yaml_loader import load_yaml


def test_dataset_classes_are_radiographic() -> None:
    cfg = load_yaml("dataset.yaml")
    names = cfg["classes"]["names"]
    assert names == ["NORMAL", "PNEUMONIA", "COVID-19"]
    assert "common cold" not in [n.lower() for n in names]
    assert cfg["classes"]["extensible"] is True
    assert cfg["central_server_receives_raw_images"] is False
    assert cfg["splits"]["augment_before_split"] is False
    assert cfg["splits"]["prevent_patient_leakage"] is True
    assert cfg["clients"]["count"] == 5


def test_augmentation_is_conservative() -> None:
    ops = load_yaml("dataset.yaml")["augmentation"]["operations"]
    assert ops["rotation_degrees"] <= 15
    assert ops["horizontal_flip"] is False
    assert load_yaml("dataset.yaml")["augmentation"]["medically_aggressive"] is False


def test_diffusion_does_not_claim_original_medfusion() -> None:
    cfg = load_yaml("diffusion.yaml")
    assert cfg["module"]["claim_original_medfusion"] is False
    assert cfg["assume_improvement"] is False


def test_experiments_start_unevaluated() -> None:
    cfg = load_yaml("experiments.yaml")
    assert cfg["metrics_never_invented"] is True
    assert all(item["status"] == "not_evaluated" for item in cfg["matrix"])
