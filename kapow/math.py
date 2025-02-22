import math

def distance(vec1, vec2):
    """
    Calculate the Euclidean distance between two embedding vectors.

    Args:
        vec1 (list[float]): First embedding vector.
        vec2 (list[float]): Second embedding vector.

    Returns:
        float: The Euclidean distance between vec1 and vec2.

    Raises:
        ValueError: If the two vectors are not the same length.
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must be of the same length")

    squared_diffs = [(a - b) ** 2 for a, b in zip(vec1, vec2)]
    return math.sqrt(sum(squared_diffs))
