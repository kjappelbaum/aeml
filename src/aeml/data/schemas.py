
import pandera as pa

schema = pa.DataFrameSchema({
    "TI-19": pa.Column(float),
    "TI-3": pa.Column(float),
    "FI-19": pa.Column(float),
    "FI-11": pa.Column(float),
    "TI-1213": pa.Column(float),
    "TI-35": pa.Column(float),
})

