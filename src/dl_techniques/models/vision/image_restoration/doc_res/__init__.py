"""PLACEHOLDER package marker for ``doc_res`` -- NOT the curated surface.

This file exists only so ``components.py`` is importable while the package is
still being built. Plan step 7 replaces it with the real curated
``__init__.py`` exporting ``__all__ = ["DocRes", "create_doc_res"]``. Do not
add exports here in the meantime; import from the leaf module
(``...doc_res.components``) until step 7 lands.
"""
