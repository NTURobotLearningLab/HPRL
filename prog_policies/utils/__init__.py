def get_env_name(task_name: str) -> str:
    from ..karel_tasks import TASK_NAME_LIST as KAREL_TASK_NAME_LIST
    from ..minigrid_tasks import TASK_NAME_LIST as MINIGRID_TASK_NAME_LIST
    """
    Get the environment name based on the task name.

    Args:
        task_name (str): The name of the task.

    Returns:
        str: The environment name.
    """
    if task_name in KAREL_TASK_NAME_LIST:
        return "karel"
    elif task_name in MINIGRID_TASK_NAME_LIST:
        return "minigrid"
    else:
        raise ValueError(f"Unknown task name: {task_name}")
