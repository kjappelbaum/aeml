# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt


def make_forecast_plot(
    y_connected,
    means_l,
    means_m,
    means_u,
    target=1,
    targets=["2-Amino-2-methylpropanol C4H11NO", "Piperazine C4H10N2"],
    outname=None,
    vlines=True,
):
    target_str = str(target)
    y_connected_df = y_connected.pd_dataframe()
    try:
        x_axis = means_m[target_str].index - y_connected_df[targets[target]].index[0]
    except KeyError:
        try:
            x_axis = means_m[targets[target]].index - y_connected_df[targets[target]].index[0]
        except Exception:
            x_axis = means_m["0"].index - y_connected_df[targets[target]].index[0]
    x = [val.total_seconds() / (60 * 60 * 24) for val in x_axis]
    x_conncected = y_connected_df[targets[target]].index - y_connected_df[targets[target]].index[0]
    x_conncected = [val.total_seconds() / (60 * 60 * 24) for val in x_conncected]

    plt.figure(figsize=(7, 3))

    try:

        plt.fill_between(
            x,
            means_l[target_str],
            means_u[target_str],
            alpha=0.2,
            color="b",
            label="forecast",
        )
        plt.plot(x, means_m[target_str], c="b", lw=0.2)
    except KeyError:
        try:
            plt.fill_between(
                x,
                means_l[targets[target]],
                means_u[targets[target]],
                alpha=0.2,
                color="b",
                label="forecast",
            )
            plt.plot(x, means_m[targets[target]], c="b", lw=0.2)
        except KeyError:
            plt.fill_between(
                x,
                means_l["0"],
                means_u["0"],
                alpha=0.2,
                color="b",
                label="forecast",
            )
            plt.plot(x, means_m["0"], c="b", lw=0.2)

    plt.plot(
        x_conncected,
        y_connected.pd_dataframe()[targets[target]],
        c="k",
        label="actual",
        lw=0.2,
    )

    plt.legend(loc="upper left")
    if vlines:
        plt.vlines(x_conncected[2704], -0.25, 1.0, color="gray")
        plt.vlines(x_conncected[4056], -0.15, 1.0, color="gray")
    # plt.xticks([])

    plt.ylim(-0.15, 0.8)
    plt.xlabel("time / days")
    plt.ylabel("normalized emissions")
    plt.tight_layout()

    if outname is not None:
        plt.savefig(outname, bbox_inches="tight")
