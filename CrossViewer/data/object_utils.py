"""Small object-name helpers used by the retained data pipeline."""

import re


def extract_object_category(object_name):
    """Strip a trailing ``_N`` suffix from an object name when present."""
    match = re.match(r"^(.+?)(?:_\d+)?$", object_name)
    if match:
        return match.group(1)
    return object_name
