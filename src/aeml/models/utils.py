# -*- coding: utf-8 -*-
def enable_dropout(model):
    """Function to enable the dropout layers during test-time"""
    for mod in model.children():
        for m2 in mod.children():
            for m3 in m2.children():
                if m3.__class__.__name__.startswith("Dropout"):
                    m3.train()


def choose_index(series, fraction: float):
    timestamps = series.time_index
    fraction_index = int(len(timestamps) * fraction)

    return timestamps[fraction_index]


def choose_index(series, fraction):
    timestamps = series.time_index
    fraction_index = int(len(timestamps) * fraction)

    return timestamps[fraction_index]


def split_data(x, y, targets, fraction_train, fraction_test=0.5):
    ts = choose_index(x, fraction_train)
    x_before, x_after = x.split_before(ts)
    y_before, y_after = y.split_before(ts)

    ts_2 = choose_index(x_after, fraction_test)

    x_valid, x_test = x_after.split_before(ts_2)
    y_valid, y_test = y_after.split_before(ts_2)

    return (
        (x_before, y_before[targets]),
        (x_valid, y_valid[targets]),
        (x_test, y_test[targets]),
        ts,
        ts_2,
    )


def get_data(x, y, num_outputs, targets_clean):
    targets = targets_clean if num_outputs == 1 else [targets_clean[0]]
    train, valid, test, ts, ts1 = split_data(x, y, targets, 0.5)

    return (train, valid, test)
