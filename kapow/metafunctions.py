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

    # For each function output signature collection item, produce a value of the type in an identically
    # shaped collection

    temp_output = function_output_signature()
    print(f"Temp output: {temp_output}")

    return temp_output