"""
Some general-purpose IDE file manipulation funcions.
"""

from ebmlite import loadSchema

__all__ = ['validate']


# ============================================================================
#
# ============================================================================

def validate(stream, from_pos=False, lookahead=25, percent=.5):
    """
    Determine if a stream contains IDE data.

    :param stream: A file-like stream (something that supports the methods
        `tell()` and `seek()`).
    :param from_pos: If `True`, validation of the stream will start at its
        current position. If `False` (default), the validation will start
        from its beginning.
    :param lookahead: The number of EBML elements to check.
    :param percent: The minumum percentage of EBML elements identified as
        being part of the IDE schema for validation. A small number of
        unknown elements may not indicate an invalid file; it may simply
        have been created using a different version of the schema.
    :return: `True` if validation passed, `False` if it failed.
    """
    # TODO: Make validation more thorough, test for corrupt files?

    orig_pos = stream.tell()
    if not from_pos:
        stream.seek(0)

    try:
        schema = loadSchema('mide_ide.xml')
        doc = schema.load(stream, headers=True)

        # Basic test: is it EBML data in the expected schema?
        known = 0
        for idx, el in enumerate(doc):
            if idx >= lookahead:
                break
            if el.id in schema.elements:
                known += 1
        if known < lookahead * percent:
            return False

        return True

    finally:
        stream.seek(orig_pos)

