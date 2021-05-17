from collections import Counter
count = Counter(
    [
        "M", "M", "F", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "F",
        "F", "F", "M", "F", "M", "M", "F", "F", "F", "M", "M", "M", "M", "M",
        "F", "F", "F", "M", "M", "M", "M", "M", "M", "M", "F", "F", "M", "M",
        "F", "M", "M", "F", "M", "F", "M", "M", "M", "M", "M", "F", "M", "F",
        "M", "F", "F", "M", "M"
    ]
)
print(count["M"]/ 61)
print(count["F"]/ 61)
print([
        "M", "M", "F", "M", "M", "M", "M", "M", "M", "M", "M", "M", "M", "F",
        "F", "F", "M", "F", "M", "M", "F", "F", "F", "M", "M", "M", "M", "M",
        "F", "F", "F", "M", "M", "M", "M", "M", "M", "M", "F", "F", "M", "M",
        "F", "M", "M", "F", "M", "F", "M", "M", "M", "M", "M", "F", "M", "F",
        "M", "F", "F", "M", "M"
    ][44:])