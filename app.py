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


load_dotenv()

app = Potassium("my_app")


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
    arize_key = os.getenv("X_ARIZE_API_KEY")
    prompt = request.json.get("prompt")
    signed_url = request.json.get("signedUrl")
    model = context.get("model")
    diffusion = context.get("diffusion")
    xm = context.get("xm")

    reqKey = request.headers.get("arize")

    if reqKey:
        print("Arize key exists in the header")
        if reqKey != arize_key:
            print("Arize key is not valid")
            return Response(json={"error": "Arize key is not valid"}, status=403)
    else:
        print("Arize key does not exist in the header")
        return Response(
            json={"error": "Arize key does not exist in the header"}, status=403
        )

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

    with open(filename, "rb") as f:
        response = requests.put(signed_url, data=f)

    if response.status_code != 200:
        print("Failed to upload to the signed URL")
        return Response(json={"error": "Failed to upload 3D model"}, status=500)

    print("Uploaded using signed URL")

    return Response(
        json={"status": "done", "signedUrl": signed_url},
        status=200,
    )


if __name__ == "__main__":
    app.serve()
