import inspect

def metafunction_definition(*args, **kwargs):
    # Stubbed function to replace decorated functions
    pass

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
        def wrapper(*args, **kwargs):
            output = metafunction_definition(*args, **kwargs)
            output_types = get_deep_type(output)
            
            # Optionally, output the types (for debugging purposes)
            print(f"Output types: {output_types}")
            
            return output
        
        # Run both functions with dummy parameters to infer input/output types
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

        return wrapper
    
    return decorator