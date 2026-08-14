"""Regression tests for target detection and filename safety.

The IMDS test exists because of a real failure: GitHub's Arm runners answer the
EC2 link-local metadata address with an HTTP 400 HTML page. An earlier version
took that body at face value as the instance type and interpolated the HTML into
an output filename, crashing the run. curl exits 0 there, so exception handling
alone never saw it -- the content itself has to be validated.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gravitune.sweep import _INSTANCE_TYPE_RE, detect_target  # noqa: E402
from gravitune.__main__ import _safe_tag  # noqa: E402


def test_real_instance_types_accepted():
    for good in ("c8g.4xlarge", "c7g.metal", "t4g.nano", "m7g.16xlarge"):
        assert _INSTANCE_TYPE_RE.match(good), good


def test_garbage_rejected():
    bad = [
        '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN">',
        "<HTML><HEAD><TITLE>Bad Request</TITLE>",
        "HTTP Error 400. The request has an invalid header name.",
        "", "   ", "no-dot-here", "way.toolongtobeaninstancetypesuffix",
        "a b.c d", "../../etc/passwd",
    ]
    for b in bad:
        assert not _INSTANCE_TYPE_RE.match(b.strip()), b


def test_safe_tag_neutralises_path_traversal_and_html():
    assert "/" not in _safe_tag("../../etc/passwd")
    assert "<" not in _safe_tag("<HTML><TITLE>Bad Request</TITLE>")
    assert _safe_tag("Neoverse-V2") == "neoverse-v2"
    assert _safe_tag("c8g.4xlarge") == "c8g-4xlarge"
    assert _safe_tag("") == "unknown"
    assert len(_safe_tag("x" * 500)) <= 40


def test_detect_never_returns_garbage_instance_type():
    """Whatever machine this runs on, instance_type is a real type or empty."""
    t = detect_target()
    assert t.instance_type == "" or _INSTANCE_TYPE_RE.match(t.instance_type)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
