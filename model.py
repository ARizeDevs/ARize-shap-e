import tempfile

import numpy as np
import PIL.Image
import torch
import trimesh
from diffusers import ShapEImg2ImgPipeline, ShapEPipeline
from diffusers.utils import export_to_ply


class Model:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = ShapEPipeline.from_pretrained(
            "openai/shap-e", torch_dtype=torch.float16
        )
        self.pipe.to(self.device)

        self.pipe_img = ShapEImg2ImgPipeline.from_pretrained(
            "openai/shap-e-img2img", torch_dtype=torch.float16
        )
        self.pipe_img.to(self.device)

    def to_glb(self, ply_path: str) -> str:
        mesh = trimesh.load(ply_path)
        rot = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        mesh = mesh.apply_transform(rot)
        rot = trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0])
        mesh = mesh.apply_transform(rot)
        mesh_path = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
        mesh.export(mesh_path.name, file_type="glb")
        return mesh_path.name

    def run_text(
        self,
        prompt: str,
        seed: int = 0,
        guidance_scale: float = 15.0,
        num_steps: int = 64,
    ) -> str:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        images = self.pipe(
            prompt,
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            output_type="mesh",
        ).images
        ply_path = tempfile.NamedTemporaryFile(suffix=".ply", delete=False, mode="w+b")
        export_to_ply(images[0], ply_path.name)
        return self.to_glb(ply_path.name)

    def run_image(
        self,
        image: PIL.Image.Image,
        seed: int = 0,
        guidance_scale: float = 3.0,
        num_steps: int = 64,
    ) -> str:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        images = self.pipe_img(
            image,
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            output_type="mesh",
        ).images
        ply_path = tempfile.NamedTemporaryFile(suffix=".ply", delete=False, mode="w+b")
        export_to_ply(images[0], ply_path.name)
        return self.to_glb(ply_path.name)
