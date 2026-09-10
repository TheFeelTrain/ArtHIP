"""Support code for the ArtHIP correctness suite.

The modules here are deliberately free of test collection: pytest discovers
only the ``test_*.py`` files at the suite root.

- ``paths``     repository/build locations and the shipped model files.
- ``build``     builds the engine driver, the execution provider and the plugin.
- ``modelgen``  tiny ONNX fixtures, registered by name.
- ``reference`` NumPy reference implementations and comparison helpers.
- ``engine``    ctypes wrapper around ``engine_harness.cc``.
"""
