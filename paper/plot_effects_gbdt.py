import matplotlib.pyplot as plt
from glob import glob
import pickle
from pathlib import Path
import numpy as np
import click
import matplotlib as mpl
from scipy.ndimage import gaussian_filter
import os
import traceback
from loguru import logger 

plt.style.reload_library()
plt.style.use('science')
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
from scipy.constants import golden

TARGETS_clean = ["2-Amino-2-methylpropanol C4H11NO", "Piperazine C4H10N2"]

def load_pickle(filename):
    with open(filename, "rb") as handle:
        res = pickle.load(handle)
    return res


def get_filename_parts(name):
    name = name.replace("delta_t_2", "delta-t-2").replace("delta_t", "delta-t")
    parts = Path(name).stem.split("_")

    return {"target": parts[-2], "data": parts[-1], "rows": parts[1], "cols": parts[2]}


def get_grids(d):
    outer_keys = sorted(d.keys())
    inner_keys = sorted(d[outer_keys[0]].keys())

    return outer_keys, inner_keys

def make_image(res, objective="1"):
    outer, inner = get_grids(res)

    image_m = np.zeros((len(outer), len(inner)))
    image_l = np.zeros((len(outer), len(inner)))
    image_t = np.zeros((len(outer), len(inner)))

    for i, point_x in enumerate(outer):
        for j, point_y in enumerate(inner):
            image_m[i][j] = np.sum(
                res[point_x][point_y][1][objective].values()
                - res[0][0][1][objective].values()
            )

            image_l[i][j] = np.sum(
                res[point_x][point_y][0][objective].values()
                - res[0][0][1][objective].values()
            )

            image_t[i][j] = np.sum(
                res[point_x][point_y][2][objective].values()
                - res[0][0][1][objective].values()
            )


    return image_m, outer, inner


def get_conditions_from_name(name):
    stem = Path(name).stem
    stem = (
        stem.replace("_long_baseline", "")
        .replace("_baseline", "")
        .replace("_full_long_no_dt", "")
        .replace("_full_2", "")
        .replace("full_3", "")
        .replace("_full_4", "")
        .replace("_full", "")
        .replace("_co2", "")
        .replace("_amine", "")
        .replace("_amp", "")
        .replace("_pz", "")
        .replace("_nh3", "").replace("_True", "").replace('_False', "")
    )
    parts = stem.split("_")
    _ = parts.pop(0)
    conditions = "_".join(parts)
    return conditions


def get_all_conditions(all_files):
    all_conditions = set()

    for filename in all_files:
        condition = get_conditions_from_name(filename)
        all_conditions.add(condition)

    all_conditions = tuple(all_conditions)

    return all_conditions


def get_color_norm(numbers):
    all_values = numbers.flatten()
    maximum = max(all_values)
    minimum = min(all_values)

    try:
        assert minimum < 0
    except AssertionError:
        minimum = -0.1

    try:
        assert maximum > 0
    except AssertionError:
        maximum = 0.1
    assert maximum > minimum

    return mpl.colors.TwoSlopeNorm(0, vmin=minimum, vmax=maximum)


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def plot_amp_pz_image(
    condition,
    all_files,
    outdir: str = None,
    blur: float = 4,
    one_color_scale: bool = False,
    single_output: bool = False,
    forecast: bool = False,
    targets = TARGETS_clean
):

    pip_image, pip_inner, pip_outer = None, None, None
    amp_image, pip_inner, pip_outer = None, None, None


    for file in all_files:
        if condition in file:

            amp_name = targets[0]
            pz_name = targets[1]
            print(file, amp_name, pz_name)
            if single_output:
                if "amp" in file:
                    print(f"amp file {file}")
                    amp_image, amp_inner, amp_outer = make_image(load_pickle(file), objective=amp_name)
                if "pz" in file:
                    print(f"pz file {file}")
                    pip_image, pip_inner, pip_outer = make_image(load_pickle(file), objective=pz_name)
            else:
                if "amp" in file:
                    amp_image, amp_inner, amp_outer = make_image(load_pickle(file), objective=amp_name)
                    pip_image, pip_inner, pip_outer = make_image(load_pickle(file), objective=pz_name)

                continue

    # try:

    #     assert pip_image is not None
    #     assert amp_image is not None
    #     assert pip_image[10][10] == 0
    #     assert amp_image[10][10] == 0
    # except AssertionError:
    #     print(condition)

    fig, ax = plt.subplots(1, 2, sharex="all", sharey="all", figsize=cm2inch(10, 10/golden))

    print(pip_image.shape)
    if blur is not None:
        pip_image = gaussian_filter(pip_image, blur)
        amp_image = gaussian_filter(amp_image, blur)

        pip_image = pip_image - pip_image[10][10]
        amp_image = amp_image - amp_image[10][10]

    if one_color_scale:
        images = [pip_image, amp_image]  # co2_image, nh3_image,

        norm = get_color_norm(np.array([im.flatten() for im in images]))

        norm_amp = norm
        norm_pip = norm

    else:

        norm_amp = get_color_norm(amp_image)
        norm_pip = get_color_norm(pip_image)

    # left, right, bottom, top
    print(
        f"left: {amp_inner[0]}, right:  {amp_inner[-1]}, bottom: {amp_outer[-1]}, top: {amp_outer[0]}"
    )
    im_amp = ax[0].imshow(
        amp_image,
        cmap="coolwarm",
        extent=[amp_inner[0], amp_inner[-1], amp_outer[-1], amp_outer[0]],
        norm=norm_amp,
        aspect="auto",
    )

    im_pip = ax[1].imshow(
        pip_image,
        cmap="coolwarm",
        extent=[pip_inner[0], pip_inner[-1], pip_outer[-1], pip_outer[0]],
        norm=norm_pip,
        aspect="auto",
    )

    ax[0].set_title("AMP")
    ax[1].set_title("Pz")

    ax[0].set_ylim(-20, 20)
    ax[0].set_xlim(-20, 20)

    if one_color_scale:
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([1, 0.2, 0.02, 0.6])
        fig.colorbar(im_pip, cax=cbar_ax, shrink=0.4, label="cumulative normalized effect")

    else:
        fig.colorbar(im_amp, ax=ax[0], shrink=0.5, location="right")
        fig.colorbar(im_pip, ax=ax[1], shrink=0.5, location="right")

    raw_conditions = condition
    condition = condition.replace("delta_t_2", "ΔT(TI-22, TI-19)")
    condition = condition.replace("delta_t", "ΔT(solvent, gas)")
    condition = condition.replace("*", "/")
    condition_parts = condition.split("_")
    y_condition = condition_parts.pop(0)
    x_condition = condition_parts.pop(0)

    ax[0].set_ylabel(y_condition)
    ax[1].set_ylabel(y_condition)

    ax[0].set_xlabel(x_condition)
    ax[1].set_xlabel(x_condition)

    fig.tight_layout()

    if outdir is not None:
        fig.savefig(
            os.path.join(outdir, f"{raw_conditions}_{str(one_color_scale)}_{str(blur)}_{str(forecast)}.pdf"),
            bbox_inches="tight",
        )
        plt.close(fig)
    else:
        fig.show()


@click.command("cli")
@click.argument("indir", type=click.Path(exists=True))
@click.argument("outdir", type=click.Path(exists=False))
@click.option('--forecast', is_flag=True)
def compute_single_output_maps(indir, outdir, forecast):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if forecast:
        all_files = glob(os.path.join(indir, "*_True"))
        targets = ['2-Amino-2-methylpropanol C4H11NO', 'Piperazine C4H10N2'] 
    else: 
        all_files = glob(os.path.join(indir, "*_False"))
        targets = ['0', '0'] 

    logger.info(f'Found {len(all_files)} files')
    all_conditions = get_all_conditions(all_files)

    for condition in all_conditions:
        try:
            plot_amp_pz_image(condition, all_files, outdir=outdir, single_output=True, forecast=forecast, targets=targets)
        except Exception as e:
            print(traceback.format_exc())
            print(e)
            pass


if __name__ == "__main__":
    compute_single_output_maps()
