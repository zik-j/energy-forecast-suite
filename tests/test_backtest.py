import pandas as pd
import numpy as np
from backtest import equity_pnl_from_signal

def test_equity_basic():
    np.random.seed(0)
    p = pd.Series(100 * (1 + 0.001*np.random.randn(300))).cumsum()
    sig = pd.Series(np.sign(np.random.randn(300)))
    pnl, metrics = equity_pnl_from_signal(p, sig, cost_bps=5)
    assert "equity" in pnl.columns
    assert isinstance(metrics["sharpe"], float)
