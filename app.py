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

# from google.cloud import storage


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

    # # Load your GCP service account credentials
    # credentials = {
    #     "type": "service_account",
    #     "project_id": os.getenv("GCS_PROJECT_ID"),
    #     "private_key_id": os.getenv("GCS_PRIVATE_KEY_ID"),
    #     "private_key": os.getenv("GCS_PRIVATE_KEY").replace("\\n", "\n"),
    #     "client_email": os.getenv("GCS_CLIENT_EMAIL"),
    #     "client_id": os.getenv("GCS_CLIENT_ID"),
    #     "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    #     "token_uri": "https://oauth2.googleapis.com/token",
    #     "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    #     "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/local-dev-david%40nomadic-mesh-390711.iam.gserviceaccount.com",
    #     "universe_domain": "googleapis.com",
    # }

    # # Create a GCS client using the credentials
    # storage_client = storage.Client.from_service_account_info(credentials)

    # # Specify the bucket and file name
    # bucket_name = "local-dev-david"
    # directory_name = "shap_e"

    # # Create a new bucket object
    # bucket = storage_client.bucket(bucket_name)

    # blob_glb = bucket.blob(f"{directory_name}/{filename}")
    # blob_glb.upload_from_filename(filename)

    # s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    # s3.upload_file(filename, 'flow-ai-hackathon', filename)

    with open(filename, "rb") as f:
        response = requests.put(signed_url, data=f)

    if response.status_code != 200:
        print("Failed to upload to the signed URL")
        return Response(json={"error": "Failed to upload 3D model"}, status=500)

    print("Uploaded using signed URL")

    return Response(json={"status": "done"}, status=200)


if __name__ == "__main__":
    app.serve()
