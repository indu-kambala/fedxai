import torch
import torch.nn.functional as F


def generate_vit_attention_map(model, image, target_class=None):
    model.eval()

    attention_maps = []

    original_attention = (
        model.encoder.layers[-1].self_attention
    )

    def attention_hook(module, inputs, output):
        if isinstance(output, tuple):
            attention_maps.append(output[1])

    original_forward = original_attention.forward

    def forward_with_attention(
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False
    ):
        output = original_forward(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            attn_mask=attn_mask,
            average_attn_weights=False,
            is_causal=is_causal
        )

        attention_maps.append(output[1])

        return output

    original_attention.forward = forward_with_attention

    with torch.no_grad():
        output = model(image)

    original_attention.forward = original_forward

    probabilities = F.softmax(output, dim=1)

    if target_class is None:
        target_class = output.argmax(dim=1).item()

    prediction_probability = (
        probabilities[0, target_class].item()
    )

    if not attention_maps:
        raise RuntimeError(
            "No attention map was captured."
        )

    attention = attention_maps[-1]

    attention = attention[0]

    attention = attention.mean(dim=0)

    attention = attention[0, 1:]

    attention = attention.reshape(14, 14)

    attention = attention.unsqueeze(0).unsqueeze(0)

    attention = F.interpolate(
        attention,
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    )

    attention = attention.squeeze()

    attention = (
        attention - attention.min()
    ) / (
        attention.max()
        - attention.min()
        + 1e-8
    )

    return (
        attention.cpu(),
        target_class,
        prediction_probability,
        probabilities[0].cpu()
    )