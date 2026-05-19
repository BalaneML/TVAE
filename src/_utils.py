
def _format_score(score):
    score = max(-99.99, min(99.99, score))
    sign = '+' if score >= 0 else '-'
    return f'{sign}{abs(score):05.2f}'

