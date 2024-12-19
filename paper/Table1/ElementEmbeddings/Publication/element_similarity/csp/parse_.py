import re

def parse_species(species: str) -> tuple[str, int]:
    """
    Parses a species string into its atomic symbol and oxidation state.

    :param species: the species string
    :return: a tuple of the atomic symbol and oxidation state

    """

    ele = re.match(r"[A-Za-z]+", species).group(0)

    charge_match = re.search(r"\d+", species)
    ox_state = int(charge_match.group(0)) if charge_match else 0

    if "-" in species:
        ox_state *= -1

    # Handle cases of X+ or X- (instead of X1+ or X1-)
    if "+" in species and ox_state == 0:
        ox_state = 1

    if ox_state == 0 and "-" in species:
        ox_state = -1
    return ele, ox_state
