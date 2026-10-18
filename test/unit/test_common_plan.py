"""``plan.KernelAbi`` on CPU: a kernel and its launcher take the ABI's names in
order (the launcher then ``stream``), and ``pack`` puts the arguments in that
order, refusing a missing or an extra one."""

import re

import pytest

from sgl_amd_mono.common.plan import KernelAbi

ABI = KernelAbi(("x", "out", "n"))


class _Kernel:
    """A ``@flyc.kernel`` / ``@flyc.jit`` stand-in: the Python function behind it."""

    def __init__(self, fn, attr="func"):
        setattr(self, attr, fn)


def _kernel(x, out, n): ...


def _launcher(x, out, n, stream): ...


def test_check_accepts_the_abi():
    ABI.check(_Kernel(_kernel), _Kernel(_launcher, "_func"))


@pytest.mark.parametrize(
    "kernel, launcher",
    [
        (lambda out, x, n: None, _launcher),  # another order
        (_kernel, _kernel),  # no stream
        (_kernel, lambda x, out, n, m, stream: None),  # one more
    ],
)
def test_check_refuses_another_signature(kernel, launcher):
    with pytest.raises(TypeError):
        ABI.check(_Kernel(kernel), _Kernel(launcher))


def test_check_refuses_a_plain_object():
    with pytest.raises(TypeError, match="no Python function"):
        ABI.check(object(), _Kernel(_launcher))


def test_pack_orders_the_values():
    assert ABI.pack({"n": 3, "x": 1, "out": 2}) == [1, 2, 3]
    assert ABI.zeros() == [0, 0, 0]


@pytest.mark.parametrize(
    "values, words",
    [
        ({"x": 1, "out": 2}, "missing ['n']"),
        ({"x": 1, "out": 2, "n": 3, "m": 4}, "unexpected ['m']"),
        ({"x": 1, "out": 2, "m": 4}, "missing ['n'], unexpected ['m']"),
    ],
)
def test_pack_refuses_a_mismatch(values, words):
    with pytest.raises(TypeError, match=re.escape(words)):
        ABI.pack(values)
