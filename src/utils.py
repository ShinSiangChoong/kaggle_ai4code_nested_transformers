import os
from tqdm import tqdm
from typing import Iterable, List


def nice_pbar(iterable: Iterable, total: int, desc: str) -> tqdm:
    """
    Create that nice progress bar.

    Args:
        iterable (Iterable): Iterable to iterate over.
        total (int): Number of elements in iterable.
        desc (str): Brief description of the process.
    Returns:
        tqdm (tqdm.std.tqdm): The tqdm object with the _nice_ settings
    """
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        ascii=True,
        bar_format='{l_bar}{bar:10}{r_bar}',
        # disable=not wandb.config.enable_tqdm
    )


def make_folder(folder: str) -> bool:
    """Makes the folder if not already present
    Args:
        folder (str): Name of folder to create
    Returns:
        created (bool): Whether or not a folder was created
    """
    try:
        os.mkdir(folder)
        return True
    except FileExistsError:
        pass
    return False


def lr_to_4sf(lr: List[float]) -> str:
    """Get string of lr list that is rounded to 4sf to not clutter pbar

    Warning:
        Doesn't work for floats > 10000
    """
    def _f(x) -> str:
        a = str(x).partition('e')
        return a[0][:5] + 'e' + a[-1]
    return '[' + ', '.join(map(_f, lr)) + ']'
