## From Banana docs

### potassium.Potassium

```python
from potassium import Potassium

app = Potassium("server")
```

This instantiates your HTTP app, similar to popular frameworks like [Flask](https://flask.palletsprojects.com/en/2.2.x/_)

### @app.init

The `@app.init` decorated function runs once on server startup, and is used to load any reuseable, heavy objects such as:

- Your AI model, loaded to GPU
- Tokenizers
- Precalculated embeddings

The return value is a dictionary which saves to the app's `context`, and is used later in the handler functions.

There may only be one `@app.init` function.

### @app.handler()

The `@app.handler` decorated function runs for every http call, and is used to run inference or training workloads against your model(s).

You may configure as many `@app.handler` functions as you'd like, with unique API routes.

The context dict passed in is a mutable reference, so you can modify it in-place to persist objects between warm handlers.

### app.serve()

`app.serve` runs the server, and is a blocking operation.

# BLIP

## Installation

1. (Optional) Creating conda environment

```bash
conda create -n lavis python=3.8
conda activate lavis
```

2. install from [PyPI](https://pypi.org/project/salesforce-lavis/)

```bash
pip install salesforce-lavis
```

3. Or, for development, you may build from source

```bash
git clone https://github.com/salesforce/LAVIS.git
cd LAVIS
pip install -e .
```

## Getting Started

### Model Zoo

Model zoo summarizes supported models in LAVIS, to view:

```python
from lavis.models import model_zoo
print(model_zoo)
# ==================================================
# Architectures                  Types
# ==================================================
# albef_classification           ve
# albef_feature_extractor        base
# albef_nlvr                     nlvr
# albef_pretrain                 base
# albef_retrieval                coco, flickr
# albef_vqa                      vqav2
# alpro_qa                       msrvtt, msvd
# alpro_retrieval                msrvtt, didemo
# blip_caption                   base_coco, large_coco
# blip_classification            base
# blip_feature_extractor         base
# blip_nlvr                      nlvr
# blip_pretrain                  base
# blip_retrieval                 coco, flickr
# blip_vqa                       vqav2, okvqa, aokvqa
# clip_feature_extractor         ViT-B-32, ViT-B-16, ViT-L-14, ViT-L-14-336, RN50
# clip                           ViT-B-32, ViT-B-16, ViT-L-14, ViT-L-14-336, RN50
# gpt_dialogue                   base
```

Let’s see how to use models in LAVIS to perform inference on example data. We first load a sample image from local.

```python
import torch
from PIL import Image
# setup device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load sample image
raw_image = Image.open("docs/_static/merlion.png").convert("RGB")
```

This example image shows [Merlion park](https://en.wikipedia.org/wiki/Merlion) ([source](https://theculturetrip.com/asia/singapore/articles/what-exactly-is-singapores-merlion-anyway/)), a landmark in Singapore.

### Image Captioning

In this example, we use the BLIP model to generate a caption for the image. To make inference even easier, we also associate each
pre-trained model with its preprocessors (transforms), accessed via `load_model_and_preprocess()`.

```python
import torch
from lavis.models import load_model_and_preprocess
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
# this also loads the associated image processors
model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)
# preprocess the image
# vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
# generate caption
model.generate({"image": image})
# ['a large fountain spewing water into the air']
```
