from ray import tune


def _try_num(x: str):
    try:
        i = int(x)
        return i
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return x  # keep as string


def parse_dist(spec: str):
    kind, args = spec.split(":")
    parts = [p.strip() for p in args.split(",") if p.strip() != ""]
    k = kind.lower()
    if k == "randint":
        return tune.randint(int(parts[0]), int(parts[1]))
    if k == "qrandint":
        return tune.qrandint(int(parts[0]), int(parts[1]), int(parts[2]))
    if k == "uniform":
        return tune.uniform(float(parts[0]), float(parts[1]))
    if k == "loguniform":
        return tune.loguniform(float(parts[0]), float(parts[1]))
    if k == "choice":
        return tune.choice([_try_num(p) for p in parts])
    if k == "grid":
        return tune.grid_search([_try_num(p) for p in parts])
    raise ValueError(f"Unsupported distribution spec: {spec}")
