import random
import math

flavors = ["Vanilla", "Chocolate", "Strawberry", "Butterscotch"]

Q = {
    "Vanilla": 3,
    "Chocolate": 10,
    "Strawberry": 5,
    "Butterscotch": 2
}
def calculate_entropy(probabilities):
    H = 0
    for p in probabilities.values():
        if p > 0:  # avoid log(0)
            H += -p * math.log(p)
    return H


def choose_action_with_entropy(Q, temperature):
   
    exp_values = {f: math.exp(Q[f] / temperature) for f in flavors}
    total = sum(exp_values.values())
    probabilities = {f: exp_values[f] / total for f in flavors}


    return random.choices(list(probabilities.keys()),
                          list(probabilities.values()))[0], probabilities



high_temp = 5.0  
low_temp = 0.5   

print("\n=== Low Entropy (Temperature = 0.5) ===")
a1, p1 = choose_action_with_entropy(Q, low_temp)
H1 = calculate_entropy(p1)
print("Chosen flavor :", a1)
print("Probabilities :", p1)
print("entropy:",H1)

print("\n=== High Entropy (Temperature = 5.0) ===")
a2, p2 = choose_action_with_entropy(Q, high_temp)
H2 = calculate_entropy(p2)
print("Chosen flavor :", a2)
print("Probabilities :", p2)
print("entropy:",H2)

