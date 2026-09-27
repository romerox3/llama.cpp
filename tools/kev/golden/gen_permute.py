"""Golden permutation orders of kev/serve.py systemone_permute: random.Random(seed), first order unshuffled, then rng.shuffle."""
import json, random, sys
rows = []
for seed in [0, 1, 7, 42, 12345, 2**31 - 1, 2**40 + 3]:
    for k in [1, 2, 3, 5, 10, 64]:
        for n_perm in [1, 6, 64]:
            keys = [f"k{i}" for i in range(k)]
            rng = random.Random(seed); orders = []
            for i in range(n_perm):
                order = list(keys)
                if i > 0: rng.shuffle(order)
                orders.append(order)
            rows.append({"seed": seed, "keys": keys, "n_perm": n_perm, "orders": orders})
with open(sys.argv[1], "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print(len(rows))
