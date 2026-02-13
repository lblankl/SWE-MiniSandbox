def custom_install_cmd(instance_id):
    """
    Custom install commands for specific instance IDs.

    Attributes:
        instance_id (str): The ID of the instance.
    Returns:
        list[str] or None: List of custom install command strings if applicable, else None.
    """
    if instance_id == "instance_001":
        return ["echo 'custom for 001'", "pip install -e ."]
    if instance_id.startswith("exp_"):
        return ["pip install experimental-pkg"]
    # if None, we fall back to default install commands
    return None
def custom_test_cmd(instance_id):
    """
    Custom test commands for specific instance IDs.
    
    Attributes:
        instance_id (str): The ID of the instance.

    Returns:
        str or None: Custom test command string if applicable, else None.
    """ 
    if instance_id == "instance_002":
        return "echo 'setup for 002'"
    # if None, we fall back to default setup commands
    return None

