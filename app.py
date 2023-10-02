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
from pygltflib import (
    GLTF2,
    Buffer,
    BufferView,
    Accessor,
    Scene,
    Node,
    Mesh,
    Primitive,
)


load_dotenv()

app = Potassium("my_app")


def obj_to_glb(obj_filename, glb_filename):
    # Load OBJ
    obj_data = pywavefront.Wavefront(obj_filename, collect_faces=True)

    # Create a GLTF object
    gltf = GLTF2()

    # Conversion process (this is a simplified example and might not handle all cases)
    # For more complex models with materials and textures, additional processing would be required.

    for name, mesh in obj_data.meshes.items():
        buffer_data = []

        # Vertices
        for vertex in mesh.vertices:
            buffer_data.extend(vertex)

        # Create buffer view for vertex data
        vertex_buffer_view = gltf.add_buffer_view(
            byteLength=len(buffer_data) * 4, target=GLTF2.TARGET_ARRAY_BUFFER
        )

        # Accessor for vertex data
        vertex_accessor = gltf.add_accessor(
            bufferView=vertex_buffer_view,
            componentType=GLTF2.COMPONENT_FLOAT,
            count=len(mesh.vertices),
            type=GLTF2.TYPE_VEC3,
            max=list(map(max, zip(*mesh.vertices))),
            min=list(map(min, zip(*mesh.vertices))),
        )

        # Indices
        indices_data = []
        for face in mesh.faces:
            indices_data.extend(face)

        # Create buffer view for indices data
        indices_buffer_view = gltf.add_buffer_view(
            byteLength=len(indices_data) * 2, target=GLTF2.TARGET_ELEMENT_ARRAY_BUFFER
        )

        # Accessor for indices data
        indices_accessor = gltf.add_accessor(
            bufferView=indices_buffer_view,
            componentType=GLTF2.COMPONENT_UNSIGNED_SHORT,
            count=len(indices_data),
            type=GLTF2.TYPE_SCALAR,
        )

        # Create mesh primitive
        primitive = {
            "attributes": {"POSITION": vertex_accessor},
            "indices": indices_accessor,
        }

        # Add mesh to GLTF
        gltf.add_mesh(primitives=[primitive])

    # Serialize the GLTF to bytes
    gltf_bytes = gltf.tobytes()

    # Convert the GLTF bytes to GLB
    with open(glb_filename, "wb") as f:
        f.write(gltf_bytes)


# def obj_to_glb(obj_filename, glb_filename):
#     # Load OBJ using pywavefront
#     scene = pywavefront.Wavefront(
#         obj_filename, create_materials=True, collect_faces=True
#     )

#     # Create a GLTF object
#     gltf = GLTF2()

#     # Extract vertices and indices
#     vertices = []
#     indices = []
#     for name, mesh in scene.meshes.items():
#         if mesh.materials and len(mesh.materials) > 0 and mesh.materials[0].vertices:
#             vertices.extend(mesh.materials[0].vertices)
#             indices.extend([i for i in range(len(vertices) // 3)])
#         else:
#             print(f"Mesh '{name}' has no valid materials or vertices. Skipping...")

#     # Create a new buffer and add it to the GLTF
#     buffer = Buffer(byteLength=len(vertices) * 4 + len(indices) * 2)
#     gltf.buffers.append(buffer)

#     # Create BufferView for vertices
#     vertex_buffer_view = BufferView(
#         buffer=0, byteOffset=0, byteLength=len(vertices) * 4, target=34962
#     )
#     gltf.bufferViews.append(vertex_buffer_view)

#     # Create BufferView for indices
#     index_buffer_view = BufferView(
#         buffer=0,
#         byteOffset=len(vertices) * 4,
#         byteLength=len(indices) * 2,
#         target=34963,
#     )
#     gltf.bufferViews.append(index_buffer_view)

#     # Create Accessor for vertices
#     vertex_accessor = Accessor(
#         bufferView=0,
#         byteOffset=0,
#         componentType=5126,
#         count=int(len(vertices) / 3),
#         type="VEC3",
#         max=list(map(max, zip(*[iter(vertices)] * 3))),
#         min=list(map(min, zip(*[iter(vertices)] * 3))),
#     )
#     gltf.accessors.append(vertex_accessor)

#     # Create Accessor for indices
#     index_accessor = Accessor(
#         bufferView=1,
#         byteOffset=0,
#         componentType=5123,
#         count=int(len(indices) / 3),
#         type="SCALAR",
#     )
#     gltf.accessors.append(index_accessor)

#     # Create Mesh
#     mesh = Mesh(primitives=[Primitive(attributes={"POSITION": 0}, indices=1)])
#     gltf.meshes.append(mesh)

#     # Create Node
#     node = Node(mesh=0)
#     gltf.nodes.append(node)

#     # Create Scene
#     scene = Scene(nodes=[0])
#     gltf.scenes.append(scene)
#     gltf.scene = 0

#     # Convert to binary GLB
#     gltf.save(glb_filename)


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

    print("Converting to GLB:" + prompt)

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
