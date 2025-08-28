import os 
from dataclasses import dataclass
from pathlib import Path 
from typing import Dict, Any, List

@dataclass 
class Config: 
    """
    Configurations for the fantasy data collector 
    """
    DATA_DIR: str = 'data'
    DEFAULT_SEASON: int = 2025
    REQUEST_TIMEOUT: int = 30
    RATE_LIMIT_DELAY: float = 2.0
    MAX_RETRIES: int = 3
    ESPN_POSITION_LIMITS: Dict[str, int] = None
    ESPN_SLOT_NUMS: Dict[str, int] = None
    YAHOO_POSITIONS: List[str] = None
    YAHOO_DEFAULT_COUNT: int = None
    YAHOO_POSITION_LIMITS: Dict[str, int] = None

    def __post_init__(self):
        if self.YAHOO_POSITIONS is None:
            self.YAHOO_POSITIONS = ["QB", "RB", "WR", "TE", "K", "DEF"]
        if self.YAHOO_DEFAULT_COUNT is None:
            self.YAHOO_POSITION_LIMITS = {
                'QB': 100,
                'WR': 300,
                'RB': 200,
                'TE': 200,
                'DEF': 32, 
                'K': 32
            }
        if self.ESPN_POSITION_LIMITS is None:
            self.ESPN_POSITION_LIMITS = {
                'QB': 100,
                'RB' : 200,
                'WR' : 300,
                'TE': 200,
                'DST': 32,
                'K': 32
            }
        if self.ESPN_SLOT_NUMS is None:
            self.ESPN_SLOT_NUMS = {
                'QB': 0,
                'RB': 2,
                'WR': 4,
                'TE': 6,
                'DST': 16,
                'K': 17
            }
        
        Path(self.DATA_DIR).mkdir(parents = True, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> 'Config':
        return cls(
            DATA_DIR = os.getenv('FANTASY_DATA_DIR', 'data'),
            DEFAULT_SEASON = int(os.getenv('FANTASY_SEASON', '2025')),
            YAHOO_COLLECT_ALL_POSITITIONS = os.getenv('YAHOO_COLLECT_ALL_POSITIONS', 'true').lower() == 'true'
            
        )

