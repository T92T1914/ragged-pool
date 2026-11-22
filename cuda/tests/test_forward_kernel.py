"""Adaptive forward kernel checks against independent float64 math and the existing kernel.

FORWARD_KERNEL.md gives the design. These tests check values only. Timing
belongs to the study script, and these tests make no speed claim.
"""
import os
import subprocess
import sys

import pytest
import torch

from ragged_pool_cuda import ragged_pool


DTYPES = (torch.float32, torch.float64)
EXISTING, ADAPTIVE = 0, 1
SWITCHES = (0, 32, 4096)  # every nonempty group on the block path, the first study's switch, the warp path


@pytest.fixture(scope="module", autouse=True)
def declared_environment():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    assert str(torch.__version__) == "2.9.1+cu128", "Use the declared isolated CUDA environment"
    assert torch.version.cuda == "12.8"
    assert torch.cuda.is_available(), "An unavailable CUDA prerequisite is not a skipped pass"
    torch.cuda.set_device(0)


def tolerance(dtype, endpoints=False):
    if dtype == torch.float32:
        return {"rtol": 2e-5, "atol": 2e-6}
    return {"rtol": 2e-10, "atol": 2e-10 if endpoints else 2e-12}


def study(values, logits, offsets, kernel=ADAPTIVE, switch_length=32, repeats=0):
    return torch.ops.ragged_pool_cuda._forward_study.default(
        values, logits, offsets, kernel, switch_length, repeats)


def batch(lengths, width, dtype, seed, scale=6.0):
    generator = torch.Generator().manual_seed(seed)
    count = sum(lengths)
    values = (torch.rand(count, width, generator=generator, dtype=torch.float64) * 2 - 1) * scale
    logits = (torch.rand(count, generator=generator, dtype=torch.float64) * 2 - 1) * scale
    offsets = torch.tensor([0, *lengths], dtype=torch.int64).cumsum(0)
    return values.to(dtype).cuda(), logits.to(dtype).cuda(), offsets.cuda()


def oracle(values, logits, offsets):
    """Float64 softmax and weighted sums on the CPU, one group at a time."""
    values64 = values.detach().cpu().to(torch.float64)
    logits64 = logits.detach().cpu().to(torch.float64)
    boundaries = offsets.cpu().tolist()
    rows = []
    for start, stop in zip(boundaries, boundaries[1:]):
        if start == stop:
            rows.append(torch.zeros(values64.shape[1], dtype=torch.float64))
        else:
            weights = torch.softmax(logits64[start:stop], dim=0)
            rows.append((weights[:, None] * values64[start:stop]).sum(dim=0))
    if not rows:
        return torch.zeros(0, values64.shape[1], dtype=torch.float64)
    return torch.stack(rows)


def assert_matches_oracle(actual, values, logits, offsets, endpoints=False):
    assert actual.dtype == values.dtype and actual.device == values.device
    expected = oracle(values, logits, offsets).to(values.dtype)
    torch.testing.assert_close(actual.cpu(), expected, **tolerance(values.dtype, endpoints))


def same_bytes(first, second):
    """Compare the stored bits, so -0.0 and 0.0 count as different."""
    if first.dtype != second.dtype or first.shape != second.shape:
        return False
    integer = torch.int32 if first.dtype == torch.float32 else torch.int64
    return torch.equal(first.contiguous().view(integer), second.contiguous().view(integer))


# The eight CPU study shapes and the six shapes at the bounds from FORWARD_KERNEL.md.
SHAPES = {
    "empty batch": ([], 8, torch.float32),
    "empty groups": ([0, 0, 0, 0], 8, torch.float64),
    "singletons": ([1] * 128, 8, torch.float32),
    "many short": ([4] * 128, 16, torch.float32),
    "balanced": ([32] * 32, 32, torch.float32),
    "skewed": ([0] + [1] * 14 + [1000], 32, torch.float32),
    "one long": ([4096], 16, torch.float64),
    "wide": ([64] * 8, 128, torch.float64),
    "full batch float32": ([32] * 128, 128, torch.float32),
    "full batch float64": ([32] * 128, 128, torch.float64),
    "one full group": ([4096], 128, torch.float64),
    "narrowest full batch": ([32] * 128, 1, torch.float32),
    "skewed full batch": ([1] * 127 + [3969], 128, torch.float32),
    "widest singletons": ([1] * 128, 128, torch.float32),
}


@pytest.mark.parametrize("switch_length", SWITCHES)
@pytest.mark.parametrize("name", list(SHAPES))
def test_study_and_bound_shapes_match_float64_math(name, switch_length):
    lengths, width, dtype = SHAPES[name]
    values, logits, offsets = batch(lengths, width, dtype, seed=72001)
    actual, timing = study(values, logits, offsets, ADAPTIVE, switch_length)
    assert actual.shape == (len(lengths), width)
    assert timing.numel() == 0
    assert_matches_oracle(actual, values, logits, offsets)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("width", (1, 2, 3, 7, 16, 31, 32, 33, 63, 64, 65, 96, 127, 128))
def test_every_column_layout_on_both_paths(dtype, width):
    # Lengths on both sides of the switch, so one launch uses both paths.
    lengths = [1, 2, 31, 32, 33, 64, 257, 0, 5]
    values, logits, offsets = batch(lengths, width, dtype, seed=72100 + width)
    for switch_length in SWITCHES:
        actual, _ = study(values, logits, offsets, ADAPTIVE, switch_length)
        assert_matches_oracle(actual, values, logits, offsets)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("length", (1, 2, 31, 32, 33, 255, 256, 257, 4095, 4096))
def test_lengths_around_the_switch_and_the_block_width(dtype, length):
    values, logits, offsets = batch([length], 33, dtype, seed=72200 + length)
    for switch_length in (0, length - 1, length, 4096):
        actual, _ = study(values, logits, offsets, ADAPTIVE, switch_length)
        assert_matches_oracle(actual, values, logits, offsets)


@pytest.mark.parametrize("dtype", DTYPES)
def test_documented_endpoints_keep_the_maximum_subtraction(dtype):
    # Logits at +-1000 overflow exp() without the maximum subtraction.
    lengths = [3, 40, 1]
    values, logits, offsets = batch(lengths, 5, dtype, seed=72300, scale=1000.0)
    logits[0], logits[1], logits[2] = 1000.0, -1000.0, 1000.0
    logits[3:43] = torch.linspace(-1000, 1000, 40, dtype=dtype, device="cuda")
    for switch_length in SWITCHES:
        actual, _ = study(values, logits, offsets, ADAPTIVE, switch_length)
        assert torch.isfinite(actual).all()
        assert_matches_oracle(actual, values, logits, offsets, endpoints=True)


def generated_cases(count, seed):
    generator = torch.Generator().manual_seed(seed)
    for index in range(count):
        groups = int(torch.randint(0, 129, (1,), generator=generator))
        width = int(torch.randint(1, 129, (1,), generator=generator))
        budget = int(torch.randint(0, 4097, (1,), generator=generator))
        lengths = []
        for _ in range(groups):
            # Mostly short groups, some long, some empty, never past N=4096.
            kind = int(torch.randint(0, 4, (1,), generator=generator))
            ceiling = (0, 8, 64, 4096)[kind]
            length = int(torch.randint(0, ceiling + 1, (1,), generator=generator))
            length = min(length, budget - sum(lengths))
            lengths.append(length)
        dtype = DTYPES[index % 2]
        yield index, lengths, width, dtype


@pytest.mark.parametrize("index, lengths, width, dtype", list(generated_cases(48, 72400)))
def test_generated_batches_match_float64_math_and_the_existing_kernel(index, lengths, width, dtype):
    values, logits, offsets = batch(lengths, width, dtype, seed=72500 + index)
    existing, _ = study(values, logits, offsets, EXISTING)
    assert_matches_oracle(existing, values, logits, offsets)
    for switch_length in SWITCHES:
        adaptive, _ = study(values, logits, offsets, ADAPTIVE, switch_length)
        assert_matches_oracle(adaptive, values, logits, offsets)
        torch.testing.assert_close(adaptive, existing, **tolerance(dtype))


@pytest.mark.parametrize("name", ("balanced", "skewed full batch", "one full group", "many short"))
def test_repeated_launches_give_identical_bytes(name):
    lengths, width, dtype = SHAPES[name]
    values, logits, offsets = batch(lengths, width, dtype, seed=72001)
    for switch_length in SWITCHES:
        first, _ = study(values, logits, offsets, ADAPTIVE, switch_length)
        for _ in range(3):
            again, _ = study(values, logits, offsets, ADAPTIVE, switch_length)
            assert same_bytes(first, again)


def order_witness():
    """One group where only the order of the additions decides the result.

    Equal logits give weights of exactly 1/4, so the products are 250, t,
    -250 and t with t = 2**-47, a quarter of the spacing between doubles
    near 250. The existing kernel adds left to right, ((250 + t) - 250) + t,
    and 250 + t rounds to 250, so it returns t. The adaptive kernel adds the
    pairs first, (250 + t) + (-250 + t), and returns 0.
    """
    tiny = 2.0 ** -45
    values = torch.tensor([[1000.0], [tiny], [-1000.0], [tiny]], dtype=torch.float64, device="cuda")
    logits = torch.zeros(4, dtype=torch.float64, device="cuda")
    offsets = torch.tensor([0, 4], dtype=torch.int64, device="cuda")
    return values, logits, offsets


def test_the_two_kernels_add_in_their_documented_orders():
    values, logits, offsets = order_witness()
    assert study(values, logits, offsets, EXISTING)[0].item() == 2.0 ** -47
    for switch_length in SWITCHES:
        assert study(values, logits, offsets, ADAPTIVE, switch_length)[0].item() == 0.0


def row_lanes(width):
    lanes = 1
    while lanes < 32 and lanes < width:
        lanes *= 2
    return lanes


def switch_for_width(width):
    """The switch pool uses with adaptive, written out the way pool.cu computes it."""
    return min(32, 8 * (32 // row_lanes(width)))


def test_public_pool_matches_the_kernel_this_process_selected():
    values, logits, offsets = batch([3, 0, 40, 1], 9, torch.float64, seed=72600)
    chosen = ADAPTIVE if os.environ.get("RAGGED_POOL_CUDA_FORWARD") == "adaptive" else EXISTING
    expected, _ = study(values, logits, offsets, chosen, switch_for_width(9))
    assert same_bytes(ragged_pool(values, logits, offsets), expected)


def path_witness(width, length):
    """One float64 group whose result shows which path of the adaptive kernel ran.

    Equal logits give every item the same weight. Column 0 holds 1000, -1000
    and t = 2**-50, and every other entry is 0. A lone warp runs
    32 / row_lanes(width) item slots side by side. When there are at least two,
    1000 and -1000 sit in slots 0 and 1, and t sits in slot 0 after 1000, where
    it rounds away, so the warp path returns 0. The block path gives that item
    to warp 1, the warp 0 pair cancels first and the result is t times the
    weight. With one slot the warp walks the items in order, so t follows 1000
    directly and -1000 comes at item 8, which the block path gives to warp 0
    beside item 0.
    """
    slots = 32 // row_lanes(width)
    tiny = 2.0 ** -50
    values = torch.zeros(length, width, dtype=torch.float64)
    values[0, 0] = 1000.0
    if slots >= 2:
        values[1, 0], values[slots, 0] = -1000.0, tiny
    else:
        values[1, 0], values[8, 0] = tiny, -1000.0
    logits = torch.zeros(length, dtype=torch.float64)
    offsets = torch.tensor([0, length], dtype=torch.int64)
    return values.cuda(), logits.cuda(), offsets.cuda()


def witness_length(width):
    """The shortest group in which path_witness tells the two paths apart."""
    slots = 32 // row_lanes(width)
    return slots + 1 if slots >= 2 else 9


@pytest.mark.parametrize("width", range(1, 129))
def test_public_pool_takes_the_path_the_width_rule_names(width):
    # One item past the switch the block path must run. At the switch the warp
    # path must run, wherever the witness can tell the paths apart there. For
    # D = 1 and D > 16 it cannot, because both paths add those items in the
    # same order, so the choice cannot change the result.
    switch = switch_for_width(width)
    adaptive = os.environ.get("RAGGED_POOL_CUDA_FORWARD") == "adaptive"
    checked = 0
    for length in (switch + 1, switch):
        if length < witness_length(width):
            continue
        values, logits, offsets = path_witness(width, length)
        warp, _ = study(values, logits, offsets, ADAPTIVE, 4096)
        block, _ = study(values, logits, offsets, ADAPTIVE, 0)
        assert warp[0, 0].item() == 0.0 and block[0, 0].item() != 0.0
        if adaptive:
            expected = block if length > switch else warp
        else:
            expected, _ = study(values, logits, offsets, EXISTING)
        assert same_bytes(ragged_pool(values, logits, offsets), expected)
        checked += 1
    assert checked == (2 if 2 <= width <= 16 else 1)


@pytest.mark.parametrize("kernel", (EXISTING, ADAPTIVE))
@pytest.mark.parametrize("name", ("empty batch", "skewed", "one full group"))
def test_timed_launches_return_the_same_output_and_finite_times(kernel, name):
    lengths, width, dtype = SHAPES[name]
    values, logits, offsets = batch(lengths, width, dtype, seed=72001)
    untimed, _ = study(values, logits, offsets, kernel)
    timed, timing = study(values, logits, offsets, kernel, 32, repeats=3)
    assert same_bytes(timed, untimed)
    assert timing.device.type == "cpu" and timing.dtype == torch.float64
    assert timing.shape == (4,)
    assert torch.isfinite(timing).all() and (timing >= 0).all()
    # The last entry spans the whole block, so it is at least each pair.
    assert timing[-1] >= timing[:-1].max()


@pytest.mark.parametrize("kernel, switch_length, repeats, message", (
    (2, 32, 0, "kernel must"), (-1, 32, 0, "kernel must"),
    (ADAPTIVE, -1, 0, "switch_length must"), (ADAPTIVE, 4097, 0, "switch_length must"),
    (ADAPTIVE, 32, -1, "repeats must"), (ADAPTIVE, 32, 201, "repeats must"),
))
def test_study_arguments_are_refused_before_admission(kernel, switch_length, repeats, message):
    # A NaN would fail admission, so the argument message shows the argument
    # check ran first.
    values, logits, offsets = batch([2, 3], 4, torch.float32, seed=72700)
    poisoned = values.clone()
    poisoned[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match=message):
        study(poisoned, logits, offsets, kernel, switch_length, repeats)
    actual, _ = study(values, logits, offsets)
    assert_matches_oracle(actual, values, logits, offsets)


@pytest.mark.parametrize("kernel", (EXISTING, ADAPTIVE))
def test_study_refuses_logical_negative_views_like_pool(kernel):
    values, logits, offsets = batch([2, 3], 4, torch.float64, seed=72750)
    negative = torch._neg_view(values)
    assert negative.is_neg()
    with pytest.raises(RuntimeError, match="negative"):
        study(negative, logits, offsets, kernel)
    with pytest.raises(RuntimeError, match="negative"):
        torch.ops.ragged_pool_cuda.pool.default(negative, logits, offsets)


@pytest.mark.parametrize("kernel", (EXISTING, ADAPTIVE))
def test_study_keeps_the_public_admission(kernel):
    values, logits, offsets = batch([2, 3], 4, torch.float64, seed=72800)
    bad_values = values.clone()
    bad_values[1, 2] = float("nan")
    with pytest.raises(RuntimeError, match="finite"):
        study(bad_values, logits, offsets, kernel)
    bad_offsets = torch.tensor([0, 3, 2, 5], dtype=torch.int64, device="cuda")
    with pytest.raises(RuntimeError, match="offsets"):
        study(values, logits, bad_offsets, kernel)
    with pytest.raises(RuntimeError, match="N must be <=4096"):
        study(torch.zeros(4097, 1, dtype=torch.float64, device="cuda"),
              torch.zeros(4097, dtype=torch.float64, device="cuda"),
              torch.tensor([0, 4097], dtype=torch.int64, device="cuda"), kernel)


CHILD = """
import sys, torch, ragged_pool_cuda
tiny = 2.0 ** -45
values = torch.tensor([[1000.0], [tiny], [-1000.0], [tiny]], dtype=torch.float64).cuda()
logits = torch.zeros(4, dtype=torch.float64).cuda()
offsets = torch.tensor([0, 4], dtype=torch.int64).cuda()
study = torch.ops.ragged_pool_cuda._forward_study.default
try:
    public = ragged_pool_cuda.ragged_pool(values, logits, offsets)
except RuntimeError as error:
    print("refused", "RAGGED_POOL_CUDA_FORWARD" in str(error))
    sys.exit(0)
existing = torch.equal(public, study(values, logits, offsets, 0, 32, 0)[0])
adaptive = torch.equal(public, study(values, logits, offsets, 1, 32, 0)[0])
print("existing" if existing and not adaptive else "adaptive" if adaptive and not existing else "both")
"""


def child_choice(setting):
    environment = dict(os.environ)
    environment.pop("RAGGED_POOL_CUDA_FORWARD", None)
    if setting is not None:
        environment["RAGGED_POOL_CUDA_FORWARD"] = setting
    completed = subprocess.run([sys.executable, "-I", "-c", CHILD], env=environment,
                               capture_output=True, text=True, timeout=300, check=False)
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip().splitlines()[-1]


def test_environment_switch_selects_the_public_kernel_for_the_whole_process():
    # The child uses the order witness, so the public result shows which
    # kernel ran. "both" would mean the witness stopped telling them apart.
    assert child_choice(None) == "existing"
    assert child_choice("existing") == "existing"
    assert child_choice("adaptive") == "adaptive"
    assert child_choice("fastest") == "refused True"
