from typing import Type
# Separate file to avoid autoreload from resetting the dictionary
RESULT_CLASSES: dict[str, Type['Result']] = {}
