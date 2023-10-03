from potassium import Potassium, Request, Response
from PIL import Image
import io
import requests
import os
from dotenv import load_dotenv
from model import Model


load_dotenv()

app = Potassium("my_app")


# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    model = Model()

    context = {
        "model": model,
    }

    return context


# @app.handler runs for every call
@app.handler("/text-to-3d")
def handler(context: dict, request: Request) -> Response:
    prompt = request.json.get("prompt")
    signed_url = request.json.get("signedUrl")
    seeds = request.json.get("seeds", 2147483647)
    guidance_scale = request.json.get("guidance_scale", 15)
    steps = request.json.get("steps", 64)
    model: Model = context.get("model")

    if prompt is None:
        return Response(json={"error": "prompt not found"}, status=401)

    if signed_url is None:
        return Response(json={"error": "signedUrl not found"}, status=401)

    print("Generating 3D model for: ", prompt)

    filename = model.run_text(prompt, seeds, guidance_scale, steps)

    print(filename)

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


@app.handler("/image-to-3d")
def image_handler(context: dict, request: Request) -> Response:
    image_url = request.json.get("imageUrl")
    signed_url = request.json.get("signedUrl")
    seed = request.json.get("seed", 0)
    guidance_scale = request.json.get("guidance_scale", 3.0)
    num_steps = request.json.get("num_steps", 64)
    model: Model = context.get("model")

    if image_url is None:
        return Response(json={"error": "imageUrl not found"}, status=401)

    print("Generating 3D model for: ", image_url)

    # Download the image
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
    except requests.RequestException as e:
        print(f"Failed to download image: {e}")
        return Response(json={"error": "Failed to download image"}, status=500)
    except Exception as e:
        print(f"Failed to process image: {e}")
        return Response(json={"error": "Failed to process image"}, status=500)

    # Generate 3D model
    filename = model.run_image(image, seed, guidance_scale, num_steps)

    print("3D asset generated for:", image_url)

    # [Optional] Upload the 3D model somewhere and get a signed URL, similar to the previous handler...
    with open(filename, "rb") as f:
        response = requests.put(signed_url, data=f)

    if response.status_code != 200:
        print("Failed to upload to the signed URL")
        return Response(json={"error": "Failed to upload 3D model"}, status=500)

    return Response(
        json={"status": "done", "filename": filename},
        status=200,
    )


if __name__ == "__main__":
    app.serve()
