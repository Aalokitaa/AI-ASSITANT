# Preemptively import sentence_transformers on Windows to avoid DLL collision / Access Violation (0xC0000005) crashes with pyarrow/torch/fitz load orders.
try:
    import sentence_transformers
except Exception:
    pass

# Marks the app directory as a Python package.

