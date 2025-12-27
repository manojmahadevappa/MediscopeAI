from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    data_dir: Path = Path('data')
    raw_dir: Path = Path('data/raw')
    processed_dir: Path = Path('data/processed')
    experiments_dir: Path = Path('experiments')
    device: str = 'cuda'


cfg = Config()
