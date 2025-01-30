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
    
def get_default_types(func):
    sig = inspect.signature(func)
    return {
        param.name: type(param.default) for param in sig.parameters.values() if param.default is not param.empty
    }

def wrap_in_tuple(item):
    """Ensure that the output is always encapsulated in a tuple at the root level."""
    if not isinstance(item, tuple):
        return (item,)
    return item

def metafunction(optimizer_function, mf_def=nn_metafunction):
    if not optimizer_function:
        raise ValueError("An optimizer function must be passed to the mf decorator.")
    
    def decorator(func):
        # Pre-calculate types during the decoration step
        function_output_signature = get_default_types(func)
        optimizer_output_signature = get_default_types(optimizer_function)
        print(f"Function default argument types: {function_output_signature}")
        print(f"Optimizer default argument types: {optimizer_output_signature}")
        try:
            dummy_args = (None,) * len(inspect.signature(func).parameters)
            dummy_kwargs = {k: None for k in inspect.signature(func).parameters.keys()}
            # Capture types by running the functions and encapsulate outputs in a tuple
            function_signature = wrap_in_tuple(get_deep_type(func(*dummy_args, **dummy_kwargs)))
            optimizer_signature = wrap_in_tuple(get_deep_type(optimizer_function(*dummy_args, **dummy_kwargs)))
            print(f"Function output types: {function_signature}")
            print(f"Optimizer output types: {optimizer_signature}")
        except Exception as e:
            print(f"Error running function for introspection: {e}")
            function_signature = optimizer_signature = None
            
        def wrapper(*args, **kwargs):
            output = mf_def(*args, **kwargs, function_signature=function_signature,
                                     optimizer_signature=optimizer_signature,
                                     function_output_signature=function_output_signature,
                                     optimizer_output_signature=optimizer_output_signature)
            output_types = wrap_in_tuple(get_deep_type(output))
            
            # Optionally, output the types (for debugging purposes)
            print(f"Output types: {output_types}")
            
            return output
        
        return wrapper
    
    return decorator