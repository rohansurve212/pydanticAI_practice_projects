def input_with_default(prompt, default):
    # Display the prompt with the default value
    user_input = input(f"{prompt} [{default}]: ")
    # Return the user input if provided, otherwise return the default value
    return user_input or default