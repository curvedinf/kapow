import inspect
from kapow.metafunctions import nn_metafunction

def get_deep_type(obj):
    """
    Recursively determine the type of the object, handling nested lists, dicts, and sets.
    """
    if isinstance(obj, list):
        return [get_deep_type(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: get_deep_type(value) for key, value in obj.items()}
    elif isinstance(obj, set):
        return {get_deep_type(item) for item in obj}
    else:
        return type(obj)

def mf(optimal_function):
    if not optimal_function:
        raise ValueError("An optimizer function must be passed to the mf decorator.")
    
    def decorator(func):
        # Pre-calculate types during the decoration step
        try:
            dummy_args = (None,) * len(inspect.signature(func).parameters)
            dummy_kwargs = {k: None for k in inspect.signature(func).parameters.keys()}
            # Capture types by running the functions
            func_output_types = get_deep_type(func(*dummy_args, **dummy_kwargs))
            optimizer_output_types = get_deep_type(optimal_function(*dummy_args, **dummy_kwargs))
            print(f"Function Output types: {func_output_types}")
            print(f"Optimizer Output types: {optimizer_output_types}")
        except Exception as e:
            print(f"Error running function stubs for introspection: {e}")
            func_output_types = optimizer_output_types = None

        def wrapper(*args, **kwargs):
            # Pass the pre-calculated types to the nn_metafunction
            output = nn_metafunction(*args, **kwargs, func_types=func_output_types, optimizer_types=optimizer_output_types)
            output_types = get_deep_type(output)
            
            # Optionally, output the types (for debugging purposes)
            print(f"Output types: {output_types}")
            
            return output
        
        return wrapper
    
    return decorator