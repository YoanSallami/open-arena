# License Apache 2.0: (c) 2026 Athena-Reply

class Dataset:
    """Base class for streaming datasets compatible with synalinks trainers.

    Synalinks' `Trainer.fit/evaluate/predict(x=...)` accepts a Python
    generator that yields `(inputs,)` or `(inputs, targets)` tuples — one
    tuple per batch. See
    `synalinks/src/trainers/data_adapters/generator_data_adapter.py` and
    the dispatch in `synalinks/src/trainers/data_adapters/__init__.py`.

    Subclasses implement `__iter__` as a generator method (using `yield`)
    that produces those tuples. Calling the dataset returns a fresh
    `types.GeneratorType` suitable for synalinks:

    ```python
    program.evaluate(x=my_dataset())
    ```

    Args:
        input_data_model (DataModel): The `synalinks.DataModel` describing
            batch inputs. Defaults to `synalinks.ChatMessages` when `None`.
        input_template (str): The jinja2 template string used to render raw
            rows into the `input_data_model`'s fields. Required.
        output_data_model (DataModel): The `synalinks.DataModel` describing
            batch targets. Defaults to `synalinks.ChatMessage` when `None`.
        output_template (str): The jinja2 template string used to render raw
            rows into the `output_data_model`'s fields. Required.
        batch_size (int): Number of examples per yielded batch. `None` lets
            the subclass decide (or yield the whole dataset as one batch).
        limit (int): Optional. Maximum number of examples to iterate over.
            `None` (default) means no cap. Useful for capping huge or
            streaming sources for quick experiments / smoke tests.
        **kwargs: Provider-specific fields forwarded by subclasses (e.g. HF
            dataset name, split, revision, API key, file path, ...).
    """

    def __init__(
        self,
        input_data_model=None,
        input_template=None,
        output_data_model=None,
        output_template=None,
        batch_size=None,
        limit: int = None,
        **kwargs,
    ):
        if input_template is None:
            raise ValueError("`input_template` is required (Jinja2 template).")
        if output_template is None:
            raise ValueError("`output_template` is required (Jinja2 template).")
        self.input_data_model = input_data_model
        self.input_template = input_template
        self.output_data_model = output_data_model
        self.output_template = output_template
        self.batch_size = batch_size
        self.limit = limit

    def __iter__(self):
        """Yield one `(inputs,)` or `(inputs, targets)` tuple per batch."""
        raise NotImplementedError

    def __call__(self):
        """Return a fresh generator over the dataset's batches."""
        return iter(self)

    def __len__(self):
        """Number of batches, if known. Override when the size is finite."""
        raise NotImplementedError
