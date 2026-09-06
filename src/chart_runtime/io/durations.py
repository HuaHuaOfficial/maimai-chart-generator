"""Simai duration lexical conversion; vector support is compiled by Harness."""
def _ratio_seconds(signature: str, bpm: float) -> float:
    a, b = signature.split(":", 1)
    return 240.0 * float(b) / (max(float(bpm), 1e-6) * float(a))


def hold_seconds(signature: str, bpm: float) -> float:
    if not signature or signature.startswith("<"):
        return 0.0
    if "#" not in signature:
        return _ratio_seconds(signature, bpm)
    left, right = signature.split("#", 1)
    if not left:
        return float(right)
    return _ratio_seconds(right, float(left))


def slide_seconds(signature: str, bpm: float) -> tuple[float, float]:
    if "##" in signature:
        wait_text, move_text = signature.split("##", 1)
        wait = float(wait_text)
        if "#" in move_text:
            temp, ratio = move_text.split("#", 1)
            move = _ratio_seconds(ratio, float(temp))
        elif ":" in move_text:
            move = _ratio_seconds(move_text, bpm)
        else:
            move = float(move_text)
        return wait, move
    if "#" in signature:
        temp, move_text = signature.split("#", 1)
        temp_bpm = float(temp)
        wait = 60.0 / max(temp_bpm, 1e-6)
        move = _ratio_seconds(move_text, temp_bpm) if ":" in move_text else float(move_text)
        return wait, move
    return 60.0 / max(float(bpm), 1e-6), _ratio_seconds(signature, bpm)
