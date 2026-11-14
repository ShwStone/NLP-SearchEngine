"""Features and binding shared by the experimental fixed-RRF CE40 gate."""

GATE_DEPTH = 40
GATE_FEATURE_NAMES = ("n_terms", "vocab_coverage", "relative_gap", "candidate_hits")
GATE_RETRIEVAL_KEYS = ("index_identity", "vector_identity", "search_config")


def gate_binding(retrieval, model):
    # The receipt keeps actual device placement. The model binding follows the
    # frozen weights, dtype, tokenizer and batching across CPU/GPU placement.
    return {"retrieval": {key: retrieval[key] for key in GATE_RETRIEVAL_KEYS},
            "model": {key: value for key, value in model.items() if key != "device"}}
