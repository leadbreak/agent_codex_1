RECIPES = {
    "transformer": {},
    "mamba": {},
    "hymba": {},
    "hymba2": {},
    "hymba3": {},
    "hymba_plus": {},
}


def get_recipe(name: str):
    if name not in RECIPES:
        raise KeyError(f"Unknown recipe '{name}'")
    return RECIPES[name]
