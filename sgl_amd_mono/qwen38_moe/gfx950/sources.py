# SPDX-License-Identifier: Apache-2.0
"""The kernel sources in every build's cache key: an edit to them rebuilds."""

import functools

from sgl_amd_mono.common.plan import source_digest


@functools.cache
def digest() -> str:
    return source_digest("common", "qwen38_moe/gfx950")
