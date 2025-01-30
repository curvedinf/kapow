def nn_metafunction(
        *args,
        function_signature=None,
        optimizer_signature=None,
        function_output_signature=None,
        optimizer_output_signature=None,
        **kwargs,
):
    
    # Debugging output to see what introspection revealed
    print(f"function_signature: {function_signature}")
    print(f"optimizer_signature: {optimizer_signature}")
    print(f"function_output_signature: {function_output_signature}")
    print(f"optimizer_output_signature: {optimizer_output_signature}")

    # Produce a tuple of default values based on the output signature types
    temp_output = tuple(typ() for typ in function_output_signature.values())
    print(f"Temp output: {temp_output}")
    
    return temp_output