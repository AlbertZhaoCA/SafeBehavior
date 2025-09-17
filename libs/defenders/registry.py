defender_registry = {}

def register_defender(name):
    """
    Decorator to register a defender class in the registry.
    Args:
        name (str): The name of the defender.
    """
    def decorator(cls):
        defender_registry[name] = cls
        return cls
    return decorator

def load_defender(defender_name, model=None, **kwargs):
    """
    Dynamically load a defender from the registry.
    Args:
        defender_name (str): The name of the defender to load.
        model (str): Optional model name to pass to the defender.
    Returns:
        An instance of the defender class.
    """
    if defender_name not in defender_registry:
        raise ValueError(f"Unknown defender type: {defender_name}")
    defender_class = defender_registry[defender_name]
    return defender_class(model, **kwargs) if model else defender_class()