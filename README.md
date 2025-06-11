### To run a project:

- install dependencies
    - without uv:
    ``` bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```
    - with uv:
    ```bash
    uv sync
    ```

- activate virtual environment (if you haven't already)

``` bash
source .venv/bin/activate 
```

- run the program

```bash
make run
```

