# License Apache 2.0: (c) 2026 Athena-Reply

"""The editable synalinks program graph.

`build_program()` is the single entry point the trial runner in
`evaluate.py` calls. Edit it in place to swap the model graph, prompt
structure, or compile-time reward — the rest of the sweep machinery
treats whatever it returns as a black-box `synalinks.Program`.
"""

import synalinks


async def build_program(model_id: str, dataset, generator_kwargs: dict, reward):
    """Build (and compile) the synalinks program for one trial.

    Any graph topology works here.
    """
    inputs = synalinks.Input(
        schema=dataset.input_schema,
        data_model=dataset.input_data_model if dataset.input_schema is None else None,
    )

    gen_kwargs = dict(generator_kwargs)
    if dataset.output_schema is not None and "schema" not in gen_kwargs:
        gen_kwargs["schema"] = dataset.output_schema

    outputs = await synalinks.Generator(
        language_model=model_id,
        **gen_kwargs,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name=f"eval_{model_id.replace('/', '_').replace(':', '_')}",
    )
    program.compile(reward=reward)
    return program
