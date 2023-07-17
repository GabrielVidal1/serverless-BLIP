from potassium import Potassium, Request, Response

from lavis.models import load_model_and_preprocess
from PIL import Image
import torch
import base64
from io import BytesIO

app = Potassium("my_app")


# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip_caption", model_type="base_coco", is_eval=True, device=device
    )

    context = {"model": model, "vis_processors": vis_processors}

    return context


# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query = request.json.get("image_base64")

    # Validate the base64 query and return a 400 response if they are invalid
    if query is None:
        return Response(
            json={"details": "No image_base64 posted in query."}, status=400
        )

    # Convert base64 image string to image
    try:
        bytes_dec = base64.b64decode(query)
        image = Image.open(BytesIO(bytes_dec)).convert("RGB")
    except:
        return Response(
            json={"details": "Can't convert image_base64 to image."}, status=400
        )

    model = context.get("model")
    vis_processors = context.get("vis_processors")

    # preprocess the image
    # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
    image = vis_processors["eval"](image).unsqueeze(0).to(device)
    # generate caption
    outputs = model.generate({"image": image})[0]

    return Response(json={"outputs": outputs}, status=200)


if __name__ == "__main__":
    app.serve()
