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
        ts, ts_2
    )

def get_data(num_outputs, x, y, targets):
    targets = targets if num_outputs == 1 else [targets[0]]
    train, valid, test, ts, ts_2 = split_data(x, y, targets, 0.5)
    return (train, valid, test, ts, ts_2)
