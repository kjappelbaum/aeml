import matplotlib.pyplot as plt


def make_forecast_plot(y_connected, means, stds, target_names, target=1, outname=None):

    fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    target_str = str(target)
    y_connected_df = y_connected.pd_dataframe()
    try:
        x_axis = means[target_str].index - y_connected_df[target_names[target]].index[0]
    except KeyError:
        x_axis = means[target_names[target]].index - y_connected_df[target_names[target]].index[0]
    x = [val.total_seconds() / (60 * 60 * 24) for val in x_axis]
    x_conncected = (
        y_connected_df[target_names[target]].index - y_connected_df[target_names[target]].index[0]
    )
    x_conncected = [val.total_seconds() / (60 * 60 * 24) for val in x_conncected]

    try:
        ax.plot(x, means[target_str], c="b", alpha=0.9, lw=0.2)
        ax.fill_between(
            x,
            means[target_str] - 3 * stds[target_str],
            means[target_str] + 3 * stds[target_str],
            alpha=0.4,
            color="b",
            label="forecast",
        )
    except KeyError:
        ax.plot(x, means[target_names[target]], c="b", alpha=0.9, lw=0.2)
        ax.fill_between(
            x,
            means[target_names[target]] - 3 * stds[target_names[target]],
            means[target_names[target]] + 3 * stds[target_names[target]],
            alpha=0.4,
            color="b",
            label="forecast",
        )

    ax.plot(
        x_conncected,
        y_connected.pd_dataframe()[target_names[target]],
        c="k",
        label="actual",
        lw=0.2,
    )

    fig.legend(loc="upper left")
    # ax.vlines(x_conncected[2704], -0.25, 1.0, color="gray")
    # ax.vlines(x_conncected[4056], -0.15, 1.0, color="gray")
    # # plt.xticks([])

    # ax.ylim(-0.15, 0.8)
    ax.set_xlabel("time / days")
    ax.set_ylabel("normalized emissions")
    fig.tight_layout()

    if outname is not None:
        fig.savefig(outname, bbox_inches="tight")
