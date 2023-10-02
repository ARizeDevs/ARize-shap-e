from potassium import Potassium, Request, Response
import torch
import uuid
import requests
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
import os
from dotenv import load_dotenv
from shap_e.util.notebooks import decode_latent_mesh

import pywavefront
from pygltflib import GLTF2, Buffer, BufferView, Accessor, Scene, Node, Mesh, Primitive


load_dotenv()

app = Potassium("my_app")


def obj_to_glb(obj_filename, glb_filename):
    # Load OBJ using pywavefront
    scene = pywavefront.Wavefront(obj_filename, collect_faces=True)

    # Create a GLTF object
    gltf = GLTF2()

    # Create buffer for storing vertex and index data
    vertices = []
    indices = []
    for name, mesh in scene.meshes.items():
        vertices.extend(mesh.vertices)
        indices.extend(mesh.faces)

    # Create a new buffer and add it to the GLTF
    buffer = Buffer(byteLength=len(vertices) * 4 + len(indices) * 2)
    gltf.buffers.append(buffer)

    # Create BufferView for vertices
    vertex_buffer_view = BufferView(
        buffer=0, byteOffset=0, byteLength=len(vertices) * 4, target=34962
    )
    gltf.bufferViews.append(vertex_buffer_view)

    # Create BufferView for indices
    index_buffer_view = BufferView(
        buffer=0,
        byteOffset=len(vertices) * 4,
        byteLength=len(indices) * 2,
        target=34963,
    )
    gltf.bufferViews.append(index_buffer_view)

    # Create Accessor for vertices
    vertex_accessor = Accessor(
        bufferView=0,
        byteOffset=0,
        componentType=5126,
        count=int(len(vertices) / 3),
        type="VEC3",
        max=list(map(max, zip(*[iter(vertices)] * 3))),
        min=list(map(min, zip(*[iter(vertices)] * 3))),
    )
    gltf.accessors.append(vertex_accessor)

    # Create Accessor for indices
    index_accessor = Accessor(
        bufferView=1,
        byteOffset=0,
        componentType=5123,
        count=int(len(indices) / 3),
        type="SCALAR",
    )
    gltf.accessors.append(index_accessor)

    # Create Mesh
    mesh = Mesh(primitives=[Primitive(attributes={"POSITION": 0}, indices=1)])
    gltf.meshes.append(mesh)

    # Create Node
    node = Node(mesh=0)
    gltf.nodes.append(node)

    # Create Scene
    scene = Scene(nodes=[0])
    gltf.scenes.append(scene)
    gltf.scene = 0

    # Convert to binary GLB
    gltf.save(glb_filename)


# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xm = load_model("transmitter", device=device)
    model = load_model("text300M", device=device)
    diffusion = diffusion_from_config(load_config("diffusion"))

    context = {"model": model, "diffusion": diffusion, "device": device, "xm": xm}

    return context


# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    # aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    # aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    prompt = request.json.get("prompt")
    signed_url = request.json.get("signedUrl")
    model = context.get("model")
    diffusion = context.get("diffusion")
    xm = context.get("xm")

    # if aws_access_key_id is None or aws_secret_access_key is None:
    #     return Response(json={"error": "AWS credentials not found"}, status=500)

    if prompt is None:
        return Response(json={"error": "prompt not found"}, status=401)

    if signed_url is None:
        return Response(json={"error": "signedUrl not found"}, status=401)

    print("Generating 3D model for: ", prompt)

    batch_size = 1
    guidance_scale = 15.0

    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )

    uuid_value = str(uuid.uuid4())
    filename = uuid_value + ".obj"

    t = decode_latent_mesh(xm, latents[0]).tri_mesh()
    with open(filename, "w") as f:
        t.write_obj(f)

    print("3D asset generated for:" + prompt)

    # with open(filename, "rb") as f:
    #     response = requests.put(signed_url, data=f)

    # if response.status_code != 200:
    #     print("Failed to upload to the signed URL")
    #     return Response(json={"error": "Failed to upload 3D model"}, status=500)

    # Convert the OBJ file to GLB
    glb_filename = uuid_value + ".glb"
    obj_to_glb(filename, glb_filename)

    with open(glb_filename, "rb") as f:
        response = requests.put(signed_url, data=f)

    if response.status_code != 200:
        print("Failed to upload to the signed URL")
        return Response(json={"error": "Failed to upload 3D model"}, status=500)

    print("Uploaded using signed URL")

    return Response(
        json={"status": "done", "fileName": glb_filename, "signedUrl": signed_url},
        status=200,
    )


if __name__ == "__main__":
    app.serve()
