import pandas as pd
from pathlib import Path

def nb_to_df(path: Path) -> pd.DataFrame:
    """Convert the jsons into dataframes
    
    Args:
        path (Path): path to json
    Returns:
        df (DataFrame): dataframe representation with following columns:
            cell_id (index): e.g. 54c7cab3, fe66203e
            cell_type (str): code / md
            source (str): source code / md
            id (str): id of json / data point
    """
    return (
        pd.read_json(
            path,
            dtype={'cell_type': 'category', 'source': 'str'})
            .assign(id=path.stem)
            .rename_axis('cell_id')
    )
