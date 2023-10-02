from potassium import Potassium, Request, Response
import boto3
import torch

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
import os
from dotenv import load_dotenv
from shap_e.util.notebooks import decode_latent_mesh
from google.cloud import storage

# Load your GCP service account credentials
credentials = {
    "type": "service_account",
    "project_id": "nomadic-mesh-390711",
    "private_key_id": "a580f3898f316dfa1093e242150fa61cbd98b4d3",
    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDUuvonVsv6N4wf\n5h2KRiGsZc4W1921Xhu/ju6ynwp7aoC0vVg/jpg2CcmY3xZ4BTSNUgaG1ZK8Lhfx\nt+76DJ9jpj2rz0IqF1uoIwXCObhQj60Fd9BlYZEEfSJTcPXhSXUbybhTZRtF04R/\ntW6R0aBEtLJTVFAyrdqDmfbnIHhVviTpDF2F9FwSPLAeu3bBhdRuX5/zTZBntziY\nweW3j2PkVJDT0mHGGIHxJrjrFoxhO/Sb+QVDxq7is1d+c7uGv6Dkm3SOqgQCzZyf\nNgocY5hIStbb+2Y06fwlWDbZhpfdqeh2DEt0js/PQu3ltMkQsw6Us0MJkCE9QOBX\nKDKg0VfxAgMBAAECggEAAKcTcwUSloGevt8OzISdMDVGf+ZRBEA9+W0pxS9k2ca0\njwFk9Lp2M/W9GiiGjg8Vh4C2u4mKlARX71ZZL8gTwdtx5V3bGgwU7FfRqeQxkSE0\ntp9kyrfVOSBdZ2hBf9vA5ck9VReqgREGXzouNPkxur47bRMlVRd4k/ZoJhLA3cH4\ndf7GmSNQVJ0LHQN+O0NBGbkBotR+l1eR/HiYM8RPsf0TobzctrnQZMBRujDcBA56\nIjAKI4aSIKXbOhJ4DlLyRFAYyrF0QfmGyb5K10LgFBHMi8/lBqIB+f+eQ31BSQT5\nRvaGr7ETw78AVx/ixRlRoYXfiuUbNHNut62xnapEIQKBgQDzeL0j3SxvmG1aln4L\n0Y7t8is9CZvDOl/hL25NKg+DN55e9mKImF11Udhpd/Ps6OYrbiPelK1rPu6FCTGf\nF1cyJZJAl4QZZc9UveLoFcDAlRfb1aIbmA7WWQD4PZYD+7Dv8rSQEsVEAlYgvfuQ\nCebSvzyoc7zYpZd+1Q/RhtyHoQKBgQDfrUhXnFHfClLTRWsuSse53yRpfL8IMVps\nw9y5wPgCCJhwVKa9Io3vpxJ+3WtpYoo5VRUdDTXgSaDagb3FKe8e6yLaT3vsIuDS\nKQBD7+uKqO482laiCHsyhctVEnxxeWWIyZ8Ak1jC2qvqW+CCVi6ySXtLirvXHz7Z\n7Sj0VvuuUQKBgQDQosZx1TYe6x/KYOvidNFmVf93bqRrx7oh3eNHfKOObjroZXpK\nU0bDIj+xIXmFqo6S6O8T7ZQuMt9yYU6EZhvyfP+3Gh+5I+VnWND40Ks8XIb1ezxP\nKR6Nz/dkwmGrSCN3eyP/0hX5EYd8x1CrPdDvB7GTKJrLjBsmNK45fre8AQKBgFj/\nPXV+gSTZmrLtQAQfRPCz1G0UcX31BSGZnM1b9lH40Y6AYAeJJChitX085Gv+BTli\nnQ/+HZ2sLhBC5xr34GjQ7gEm9wuxpPp5zd06LOHy7TfRN/8omLw1d/3VaSZNQxT2\nBAnalsqQ7y5EeEPjvpi5nBEmli62A+/56P2vTlzBAoGAdoomemwc/5mmb1qInnsB\ncsIoZeoEY0V87LBYMUcnBuMEUMo435rVn6Q8PUvHZFd80xDsk8GItcpqq/hNZcOl\nA/sMDZb/7ESunc7v1YLJvmnEN7X1Y748ucF+i6NP/jtYBnphYd3zUfYlEaoDf9CD\nWM/U1jJhdVWYMc0pt7sn2YU=\n-----END PRIVATE KEY-----\n",
    "client_email": "local-dev-david@nomadic-mesh-390711.iam.gserviceaccount.com",
    "client_id": "100080683936641405232",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/local-dev-david%40nomadic-mesh-390711.iam.gserviceaccount.com",
    "universe_domain": "googleapis.com",
}

load_dotenv()

app = Potassium("my_app")


# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xm = load_model('transmitter', device=device)
    model = load_model('text300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))

    context = {
        "model": model,
        "diffusion": diffusion,
        "device": device,
        "xm": xm
    }

    return context


# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    prompt = request.json.get("prompt")
    model = context.get("model")
    diffusion = context.get("diffusion")
    xm = context.get("xm")

    if aws_access_key_id is None or aws_secret_access_key is None:
        return Response(
            json={"error": "AWS credentials not found"},
            status=500
        )

    print('Generating 3D model for: ', prompt)

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
        sigma_min=1E-3,
        sigma_max=160,
        s_churn=0,
    )

    filename = prompt + ".obj"

    t = decode_latent_mesh(xm, latents[0]).tri_mesh()
    with open(filename, 'w') as f:
        t.write_obj(f)

    print('3D asset generated for:' + prompt)

    # Create a GCS client using the credentials
    storage_client = storage.Client.from_service_account_info(credentials)
    
    # Specify the bucket and file name
    bucket_name = "local-dev-david"
    directory_name = "shap_e"

    # Create a new bucket object
    bucket = storage_client.bucket(bucket_name)

    blob_glb = bucket.blob(f"{directory_name}/{filename}")
    blob_glb.upload_from_filename(filename)
    

    # s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    # s3.upload_file(filename, 'flow-ai-hackathon', filename)

    print('Uploaded to GCS')

    return Response(
        json={"url": "sds" + filename},
        status=200
    )


if __name__ == "__main__":
    app.serve()
