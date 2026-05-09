"""
Test configuration — sets up sys.path and patches heavy infrastructure
imports so pure-function unit tests run without Neo4j / Ollama.
"""
import sys
import os
from unittest.mock import MagicMock, patch

# Make `src` importable from the api root
API_ROOT = os.path.join(os.path.dirname(__file__), "..")
if API_ROOT not in sys.path:
    sys.path.insert(0, API_ROOT)

# Patch heavy infrastructure modules before any src import
_MOCKED_MODULES = [
    "langchain_core",
    "langchain_core.runnables",
    "langchain_ollama",
    "ollama",
    "neo4j",
    "torch",
    "transformers",
    "sentence_transformers",
    "pythainlp",
]
for mod in _MOCKED_MODULES:
    sys.modules.setdefault(mod, MagicMock())
