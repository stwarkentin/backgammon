import random

def roll():
    """Rolls two dice and returns the results as a list. If doubles are rolled, the results are added to the list twice."""
    results = [random.randint(1,6),random.randint(1,6)]
    if results[0] == results[1]:
        results = results + results
    return results
