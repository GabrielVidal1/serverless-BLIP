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
