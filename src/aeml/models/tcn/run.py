from .tcn_dropout import TCNDropout


def run_model(
    x_train,
    y_train,
    input_chunk_length,
    output_chunk_length,
    num_layers=8,
    num_filters=64,
    kernel_size=3,
    dropout=0.7431,
    weight_norm=True,
    batch_size=64,
    n_epochs=200,
    lr=0.001,
):

    model_cov = TCNDropout(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        num_layers=num_layers,
        num_filters=num_filters,
        kernel_size=kernel_size,
        dropout=dropout,
        weight_norm=weight_norm,
        batch_size=batch_size,
        n_epochs=n_epochs,
        log_tensorboard=False,
        optimizer_kwargs={"lr": lr},
    )

    model_cov.fit(series=y_train, past_covariates=x_train, verbose=False)

    return model_cov
