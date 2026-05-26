from abc import ABC, abstractmethod
from src.core.market_context import MarketContext

class BaseStrategy(ABC):
    """
    Abstract Base Class for option pricing and probability estimation strategies.
    Defines a standard interface for dynamic plug-and-play strategy loading.
    """
    
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def get_probability(self, context: MarketContext) -> float:
        """
        Calculates the theoretical probability that the spot price will expire 
        above the strike price (i.e. YES option exercise probability).
        
        Args:
            context: The current MarketContext containing spot, strike, tau, vol, OFI, and books.
            
        Returns:
            Calculated probability (p) in the range [0.0, 1.0].
        """
        pass

    def calculate_edge(self, model_prob: float, market_price: float, is_buy: bool) -> float:
        """
        Calculates the theoretical edge in probability/price space.
        
        For buying YES (or NO): edge = model_prob - market_price
        For selling YES (or NO): edge = market_price - model_prob
        """
        if is_buy:
            return model_prob - market_price
        else:
            return market_price - model_prob
