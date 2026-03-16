"""
Opponent modelling across a 1000-hand match.

Tracks per-street statistics so the betting strategy can adapt:
  - fold frequency   (how often they give up)
  - raise frequency  (how often they escalate)
  - VPIP             (voluntarily put money in pot pre-flop)
  - aggression factor (raises / calls)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _StreetStats:
    """Accumulator for a single street."""
    actions: int = 0
    folds: int = 0
    raises: int = 0
    calls: int = 0
    checks: int = 0
    total_raise_amount: int = 0
    
    # Enhanced tracking: bet sizing patterns
    small_raises: int = 0  # 30-50% of pot
    medium_raises: int = 0  # 50-80% of pot
    large_raises: int = 0  # 80%+ of pot
    total_pot_size: int = 0  # For calculating raise sizes relative to pot
    
    # Recent trend tracking (exponential decay)
    recent_fold_rate: float = 0.0
    recent_aggression: float = 0.0
    recent_actions: int = 0

    @property
    def fold_rate(self) -> float:
        return self.folds / self.actions if self.actions else 0.0

    @property
    def raise_rate(self) -> float:
        return self.raises / self.actions if self.actions else 0.0

    @property
    def avg_raise_size(self) -> float:
        return self.total_raise_amount / self.raises if self.raises else 0.0

    @property
    def aggression_factor(self) -> float:
        """Raises / calls.  Higher = more aggressive."""
        return self.raises / self.calls if self.calls else float(self.raises)
    
    @property
    def small_raise_rate(self) -> float:
        """Fraction of raises that are small (30-50% pot)."""
        return self.small_raises / self.raises if self.raises else 0.0
    
    @property
    def large_raise_rate(self) -> float:
        """Fraction of raises that are large (80%+ pot)."""
        return self.large_raises / self.raises if self.raises else 0.0
    
    def update_recent_trends(self, fold_rate: float, aggression: float, decay: float = 0.7):
        """Update recent trends with exponential decay."""
        if self.recent_actions == 0:
            self.recent_fold_rate = fold_rate
            self.recent_aggression = aggression
        else:
            self.recent_fold_rate = decay * self.recent_fold_rate + (1 - decay) * fold_rate
            self.recent_aggression = decay * self.recent_aggression + (1 - decay) * aggression
        self.recent_actions += 1


class OpponentModel:
    """
    Lightweight per-match tracker.  Call `record_action` after every
    opponent move; query `fold_rate`, `raise_rate`, etc. when deciding.
    """

    def __init__(self) -> None:
        # Per-street stats (0=pre-flop … 3=river) + an aggregate bucket
        self.streets: list[_StreetStats] = [_StreetStats() for _ in range(4)]
        self.overall = _StreetStats()

        # VPIP: count of hands where opponent voluntarily put chips in pre-flop
        self._hands_seen: int = 0
        self._vpip_count: int = 0
        self._current_hand_vpip: bool = False
        
        # Context-aware stats: track by pot size buckets
        self._small_pot_stats = _StreetStats()  # pots < 10 chips
        self._medium_pot_stats = _StreetStats()  # pots 10-50 chips
        self._large_pot_stats = _StreetStats()  # pots > 50 chips

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def new_hand(self) -> None:
        """Call at the start of every new hand."""
        self._hands_seen += 1
        self._current_hand_vpip = False

    def record_action(
        self,
        street: int,
        action_name: str,
        raise_amount: int = 0,
        pot_size: int = 0,
    ) -> None:
        """
        Record an opponent action.

        Parameters
        ----------
        street : int  (0-3)
        action_name : str  – one of "FOLD", "RAISE", "CHECK", "CALL", "DISCARD"
        raise_amount : int – only meaningful for RAISE
        pot_size : int – pot size at time of action (for context-aware tracking)
        """
        if action_name == "DISCARD":
            return  # mandatory, not strategic

        bucket = self.streets[min(street, 3)]
        bucket.actions += 1
        self.overall.actions += 1
        
        # Context-aware tracking by pot size
        if pot_size > 0:
            if pot_size < 10:
                pot_bucket = self._small_pot_stats
            elif pot_size < 50:
                pot_bucket = self._medium_pot_stats
            else:
                pot_bucket = self._large_pot_stats
            
            pot_bucket.actions += 1
            pot_bucket.total_pot_size += pot_size

        if action_name == "FOLD":
            bucket.folds += 1
            self.overall.folds += 1
            if pot_size > 0:
                pot_bucket.folds += 1
        elif action_name == "RAISE":
            bucket.raises += 1
            bucket.total_raise_amount += raise_amount
            bucket.total_pot_size += pot_size
            self.overall.raises += 1
            self.overall.total_raise_amount += raise_amount
            self.overall.total_pot_size += pot_size
            
            # Track bet sizing patterns (relative to pot)
            if pot_size > 0:
                raise_fraction = raise_amount / pot_size
                if raise_fraction < 0.50:
                    bucket.small_raises += 1
                elif raise_fraction < 0.80:
                    bucket.medium_raises += 1
                else:
                    bucket.large_raises += 1
                
                pot_bucket.raises += 1
                pot_bucket.total_raise_amount += raise_amount
            
            if street == 0:
                self._current_hand_vpip = True
        elif action_name == "CALL":
            bucket.calls += 1
            self.overall.calls += 1
            if pot_size > 0:
                pot_bucket.calls += 1
            if street == 0:
                self._current_hand_vpip = True
        elif action_name == "CHECK":
            bucket.checks += 1
            self.overall.checks += 1
            if pot_size > 0:
                pot_bucket.checks += 1
        
        # Update recent trends after recording action
        if bucket.actions > 0:
            bucket.update_recent_trends(bucket.fold_rate, bucket.aggression_factor)

    def end_hand(self) -> None:
        """Call at the end of every hand to finalize VPIP."""
        if self._current_hand_vpip:
            self._vpip_count += 1

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def vpip(self) -> float:
        """Fraction of hands where opponent voluntarily invested pre-flop."""
        return self._vpip_count / self._hands_seen if self._hands_seen else 0.5

    def fold_rate(self, street: int | None = None) -> float:
        bucket = self.streets[street] if street is not None else self.overall
        return bucket.fold_rate

    def raise_rate(self, street: int | None = None) -> float:
        bucket = self.streets[street] if street is not None else self.overall
        return bucket.raise_rate

    def aggression(self, street: int | None = None) -> float:
        bucket = self.streets[street] if street is not None else self.overall
        return bucket.aggression_factor

    def avg_raise_size(self, street: int | None = None) -> float:
        bucket = self.streets[street] if street is not None else self.overall
        return bucket.avg_raise_size

    @property
    def hands_seen(self) -> int:
        return self._hands_seen
    
    def recent_fold_rate(self, street: int | None = None) -> float:
        """Get recent fold rate (weighted toward recent hands)."""
        bucket = self.streets[street] if street is not None else self.overall
        if bucket.recent_actions > 0:
            return bucket.recent_fold_rate
        return bucket.fold_rate
    
    def recent_aggression(self, street: int | None = None) -> float:
        """Get recent aggression (weighted toward recent hands)."""
        bucket = self.streets[street] if street is not None else self.overall
        if bucket.recent_actions > 0:
            return bucket.recent_aggression
        return bucket.aggression_factor
    
    def is_tight(self) -> bool:
        """Classify opponent as tight (low VPIP, high fold rate)."""
        if self._hands_seen < 20:
            return False  # Not enough data
        return self.vpip < 0.40 and self.overall.fold_rate > 0.50
    
    def is_loose(self) -> bool:
        """Classify opponent as loose (high VPIP, low fold rate)."""
        if self._hands_seen < 20:
            return False  # Not enough data
        return self.vpip > 0.60 and self.overall.fold_rate < 0.40
    
    def bet_sizing_tendency(self, street: int | None = None) -> str:
        """
        Classify opponent's bet sizing tendency.
        Returns: 'small', 'medium', 'large', or 'mixed'
        """
        bucket = self.streets[street] if street is not None else self.overall
        if bucket.raises < 5:
            return 'mixed'  # Not enough data
        
        small_rate = bucket.small_raise_rate
        large_rate = bucket.large_raise_rate
        
        if small_rate > 0.6:
            return 'small'
        elif large_rate > 0.5:
            return 'large'
        else:
            return 'medium'
