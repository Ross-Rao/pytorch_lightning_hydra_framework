def get_multi_attr(modules: list, attr: dict):
    """Get multiple attributes from multiple modules.

    Args:
        modules : List of modules.
        attr (dict): Dictionary of attributes to get from the modules.

    Returns:
        list: List of results from the attributes.

    Example:
        >>> from utils import custom_transforms as custom_transforms
        >>> from torchvision import transforms
        >>> transform = {"ResampleNifti": None, "PermuteDimensions": (2, 0, 1), "PadChannels": 220,
        >>>              "Resize": {"size": 256}, "RandomCrop": {"size": 224}, }
        >>> transform_lt = get_multi_attr([transforms, custom_transforms], transform)
        >>> transform = transforms.Compose(transform_lt)
    """
    results = []
    for func_name, params in attr.items():
        funcs = [getattr(module, func_name, None) for module in modules]
        valid_funcs = [func for func in funcs if func is not None]
        if not valid_funcs:
            raise AttributeError(f"Attribute '{func_name}' not found in any of the modules: {', '.join(modules)}")
        elif len(valid_funcs) > 1:
            raise AttributeError(f"Attribute '{func_name}' found in multiple modules: {', '.join(modules)}")
        else:
            func = valid_funcs[0]
            # if failed, check the value of params
            if params is None:
                result = func()
            elif isinstance(params, dict):
                result = func(**params)
            else:
                result = func(params)
            results.append(result)

    return results
