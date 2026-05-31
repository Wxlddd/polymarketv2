from typing import Dict, Type
from src.core.base_strategy import BaseStrategy

class StrategyFactory:
    """
    Central strategy factory enabling dynamic loading of option pricing strategies.
    Decouples core orchestrator and backtesting runner from concrete implementations.
    """
    
    @classmethod
    def get_strategy(cls, name: str, config) -> BaseStrategy:
        name_lower = name.lower()
        
        if name_lower == "merton":
            from src.strategies.merton_strategy import MertonStrategy
            return MertonStrategy(config)
        elif name_lower == "legacy_merton":
            from src.strategies.legacy_merton_strategy import MertonStrategy as LegacyMertonStrategy
            return LegacyMertonStrategy(config)
        else:
            raise ValueError(
                f"Unknown strategy name: '{name}'. "
                f"Available strategies: ['merton', 'legacy_merton']"
            )
