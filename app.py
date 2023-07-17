from potassium import Potassium, Request, Response

from transformers import pipeline
import torch

app = Potassium("my_app")


# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    device = 0 if torch.cuda.is_available() else -1

    model = None  # TODO Load your model here, passing the device as an argument

    context = {"model": model}

    return context


# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    arg = request.json.get("arg")  # TODO Run your model here, passing the arguments

    # TODO Validate the arguments and return a 400 response if they are invalid

    model = context.get("model")

    outputs = model(arg)  # TODO Run your model here, passing the arguments

    return Response(json={"outputs": outputs}, status=200)


if __name__ == "__main__":
    app.serve()
