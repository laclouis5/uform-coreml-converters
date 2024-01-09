from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as cto
import torch
import uform

from .model_fp16 import get_fp16_model


class CML_TextEncoder(torch.nn.Module):
    def __init__(self, model: uform.TextEncoder):
        super().__init__()
        self.model = model.eval()

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.model.forward_features(input_ids, attention_mask)
        embeddings = self.model.forward_embedding(features, attention_mask)

        return features, embeddings


class CML_ImageEncoder(torch.nn.Module):
    def __init__(self, model: uform.VisualEncoder):
        super().__init__()
        self.model = model.eval()

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.model.forward_features(image)
        embeddings = self.model.forward_embedding(features)

        return features, embeddings


def convert_model(model_name: str, out_dir: Path, compression: str | None = None):
    model = get_fp16_model(f"unum-cloud/{model_name}").eval()
    image_encoder = CML_ImageEncoder(model.image_encoder).eval()
    text_encoder = CML_TextEncoder(model.text_encoder).eval()

    max_length = model.text_encoder.max_position_embeddings
    c, img_size = 3, model.image_encoder.image_size

    # (1, E)
    input_ids = torch.ones(1, max_length, dtype=torch.int32)
    attention_mask = torch.ones(1, max_length, dtype=torch.int32)

    # (1, 3, H, W)
    image = torch.ones(1, c, img_size, img_size, dtype=torch.float32)

    with torch.inference_mode():
        image_encoder = torch.jit.trace(image_encoder, image)
        text_encoder = torch.jit.trace(text_encoder, (input_ids, attention_mask))

    image_encoder = ct.convert(
        image_encoder,
        inputs=[
            ct.TensorType(
                name="image",
                shape=image.shape,
                dtype=image.numpy().dtype,
            )
        ],
        outputs=[
            ct.TensorType(name="features"),
            ct.TensorType(name="embeddings"),
        ],
        convert_to="mlprogram",
    )

    text_encoder = ct.convert(
        text_encoder,
        inputs=[
            ct.TensorType(
                name="input_ids",
                shape=input_ids.shape,
                dtype=input_ids.numpy().dtype,
            ),
            ct.TensorType(
                name="attention_mask",
                shape=attention_mask.shape,
                dtype=attention_mask.numpy().dtype,
            ),
        ],
        outputs=[
            ct.TensorType(name="features"),
            ct.TensorType(name="embeddings"),
        ],
        convert_to="mlprogram",
    )

    if compression == "palettization":
        config = cto.OpPalettizerConfig(nbits=8, weight_threshold=512)
        op_config = cto.OptimizationConfig(config)

        image_encoder = cto.palettize_weights(image_encoder, op_config)
        text_encoder = cto.palettize_weights(text_encoder, op_config)
    elif compression is None:
        pass
    else:
        raise ValueError(f"Compression method '{compression}' not supported.")

    out_dir.mkdir(exist_ok=True)
    image_encoder.save(out_dir / f"{model_name}_image-encoder.mlpackage")
    text_encoder.save(out_dir / f"{model_name}_text-encoder.mlpackage")
