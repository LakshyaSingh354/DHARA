import modal

app = modal.App("example-get-started")


@app.function()
@modal.web_endpoint()
def square(x: int):
    return {"square": x**2}
