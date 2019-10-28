"""Santity check tests.

Verifies that the tests are actually running and that Python is not misbehaving.
"""


def test_package_exists() -> None:
    """Test the existence of our main package."""
    import decuen
    assert decuen is not None
