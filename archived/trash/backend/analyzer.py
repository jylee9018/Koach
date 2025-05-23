# analyzer.py

def compare_phonemes(expected: list[str], actual: list[str]) -> list[dict]:
    from difflib import SequenceMatcher

    matcher = SequenceMatcher(None, expected, actual)
    result = []

    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            for i in range(i2 - i1):
                result.append({"symbol": expected[i1 + i], "correct": True})
        elif op == "replace":
            for i in range(i2 - i1):
                result.append({
                    "symbol": expected[i1 + i],
                    "correct": False,
                    "detected_as": actual[j1 + i] if j1 + i < j2 else None
                })
        elif op == "delete":
            for i in range(i1, i2):
                result.append({
                    "symbol": expected[i],
                    "correct": False,
                    "detected_as": None
                })
        elif op == "insert":
            for j in range(j1, j2):
                result.append({
                    "symbol": None,
                    "correct": False,
                    "detected_as": actual[j]
                })
    return result