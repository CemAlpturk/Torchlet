from hypothesis.strategies import floats, integers


small_ints = integers(min_value=1, max_value=3)
small_floats = floats(min_value=-100, max_value=100, allow_nan=False)
med_ints = integers(min_value=1, max_value=20)


def assert_close(a: float, b: float) -> None:

    assert abs(a - b) < 1e-2, f"Failure x={a} y={b}"
