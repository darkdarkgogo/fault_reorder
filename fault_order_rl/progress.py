"""Small, dependency-free progress logger for long-running solver stages."""

import sys


def _field(value):
    text = str(value).replace("\r", "\\r").replace("\n", "\\n")
    if any(character.isspace() for character in text):
        return repr(text)
    return text


def progress(scope, event, **fields):
    parts = ["[{}]".format(scope), event]
    parts.extend("{}={}".format(key, _field(value)) for key, value in fields.items())
    print(" ".join(parts), file=sys.stderr, flush=True)


def sample_progress(index, first, every):
    return index <= first or index % every == 0
