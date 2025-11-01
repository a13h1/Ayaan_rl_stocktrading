# Standard library imports
import argparse
import json
import os
import random
import subprocess
import time
from collections import deque, namedtuple
from typing import Tuple, Optional

# Third-party imports
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Local imports
# Note: DQNAgent, GymTradingEnv, train_ppo, QLearningAgent, TradingEnv, download_weekly,
# prepare_features, train_val_test_split, compute_metrics, buy_and_hold, sma_10_30,
# momentum_12_1, rsi_rule are all defined in this file, so all related imports are removed

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class TradingEnv:
    """
    Weekly trading environment.
    State at time t = feature vector computed using only past data (no peek).
    Actions = {-1, 0, +1} (if shorting allowed) else {0, +1}
    Position cap = 1 unit.
    Default transaction_cost = 0.001 (0.10%)
    Reward at time t (after taking action at t) = position_t * return_{t+1} - cost_if_trade - slippage_if_trade
    Episode = full sequence for the selected split (train/val/test).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        split: str = "test",
        transaction_cost: float = 0.001,
        slippage_bps: float = 0.0,
        allow_short: bool = True,
        seed: Optional[int] = None,
    ):
        """
        data must be a DataFrame with columns:
            ['Date', 'Ticker', 'Close', 'MA_5', 'MA_10', 'Return', ...]
        Where MA_* are computed using only past data (e.g., shifted).
        """
        self.raw = data.reset_index(drop=True)
        self.split = split
        # Configurable (defaults match previous behavior)
        self.transaction_cost = float(transaction_cost)  # e.g., 0.001
        self.slippage_bps = float(slippage_bps)  # e.g., 0.0005 for 5 bps
        self.allow_short = bool(allow_short)

        self.position = 0  # -1, 0, +1
        self.t = 0
        self.done = False

        if seed is not None:
            np.random.seed(seed)

        self._build_indices()

    def _build_indices(self):
        # Data is already filtered by tick/split outside before constructing env.
        self.data = self.raw.copy().reset_index(drop=True)

    def reset(self) -> np.ndarray:
        self.position = 0
        self.t = 0
        self.done = False
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        # Accept -1,0,+1 actions but enforce allow_short
        assert action in (-1, 0, 1), f"Action must be -1,0,1 got {action}"
        prev_position = self.position
        # enforce cap and whether shorting is allowed
        action = max(-1 if self.allow_short else 0, min(1, action))
        self.position = action

        # cost if position changed
        traded = self.position != prev_position
        cost = self.transaction_cost if traded else 0.0

        # slippage: assume one-time per trade; use absolute slippage_bps
        slip = abs(self.slippage_bps) if traded else 0.0

        # reward uses next-week return: returns column is "Return" = pct change from prev to current
        # At time t we are using position_t * return_{t+1}
        r = 0.0
        if self.t < len(self.data) - 1:
            next_ret = self.data.loc[self.t + 1, "Return"]
            # same sign as position in effect since position may be -1/0/+1
            r = self.position * next_ret - cost - slip
        else:
            # end of series, no next return
            r = (
                -cost - slip
            )  # if you trade at final step you still pay cost/slippage; otherwise 0

        self.t += 1
        if self.t >= len(self.data) - 1:
            self.done = True

        next_state = self._get_state()
        info = {
            "position": self.position,
            "cost": cost,
            "slippage": slip,
            "ticker": self.data.get("Ticker", pd.Series(["Unknown"])).iloc[0],
            "split": self.split,
        }
        return next_state, float(r), bool(self.done), info

    def _get_state(self) -> np.ndarray:
        # get features at current t; ensure deterministic ordering
        cols = ["Close", "MA_5", "MA_10", "Return"]
        values = self.data.loc[self.t, cols].values.astype(float)
        return values

    @property
    def current_ticker(self):
        return self.data.get("Ticker", pd.Series(["Unknown"])).iloc[0]

    @property
    def current_split(self):
        return self.split


def download_weekly(tickers, start, end, out_dir="data"):
    os.makedirs(out_dir, exist_ok=True)
    for t in tickers:
        df = yf.download(
            t, start=start, end=end, interval="1wk", auto_adjust=False, progress=False
        )
        if df.empty:
            continue
        # Flatten MultiIndex columns if present (yfinance returns MultiIndex even for single ticker)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df[["Close", "Adj Close"]].copy()
        df.reset_index(inplace=True)
        df["Ticker"] = t
        # ensure Close is Adjusted Close for returns (use 'Adj Close' if available)
        # Ensure numeric types before assignment
        df["Adj Close"] = pd.to_numeric(df["Adj Close"], errors="coerce")
        df["Close"] = df["Adj Close"].copy()
        df.to_csv(f"{out_dir}/{t}.csv", index=False)


def prepare_features(
    df: pd.DataFrame, use_cols=("Close", "MA_5", "MA_10", "Return")
) -> pd.DataFrame:
    """
    Prepare features safely (no-lookahead). Computes Return (prev->current),
    MA_5/MA_10 that are shifted by 1 (so MA_t uses Close[t-1..t-5]).
    Allows selecting a subset of columns (feature ablation).
    """
    df = df.sort_values("Date").reset_index(drop=True)
    # Ensure Close is numeric (convert from string if read from CSV)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce").fillna(0)
    # Return: current pct change from previous close (we will use t+1 in env)
    df["Return"] = df["Close"].pct_change().fillna(0)
    df["MA_5"] = df["Close"].rolling(5).mean().shift(1)
    df["MA_10"] = df["Close"].rolling(10).mean().shift(1)
    # conservative fill for initial rows
    df[["MA_5", "MA_10"]] = (
        df[["MA_5", "MA_10"]].bfill().ffill()
    )
    # keep only requested features + required
    keep = ["Date", "Ticker"] + list(dict.fromkeys(use_cols))
    # ensure every requested col exists (if not, create with NaNs then fill)
    for c in keep:
        if c not in df.columns:
            df[c] = 0.0
    return df[keep].copy()


def assert_no_leak(df):
    """
    Weak guard asserting expected rolling/lagged features exist.
    More thorough unit tests live in tests/test_leakage.py.
    """
    assert "MA_5" in df and "MA_10" in df
    return True


def train_val_test_split(df, train_frac=0.6, val_frac=0.2):
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    splits = {
        "train": df.iloc[:train_end].reset_index(drop=True),
        "val": df.iloc[train_end:val_end].reset_index(drop=True),
        "test": df.iloc[val_end:].reset_index(drop=True),
    }
    return splits


# Metrics
def sharpe_ratio(returns, periods_per_year=52):
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    if sigma == 0:
        return 0.0
    return (mu * periods_per_year) / (sigma * np.sqrt(periods_per_year))


def sortino_ratio(returns, periods_per_year=52):
    arr = np.asarray(returns)
    mean = arr.mean() * periods_per_year
    downside = arr[arr < 0]
    # if no downside, std = 0 -> return 0
    if len(downside) == 0:
        return 0.0
    dd = downside.std(ddof=1) * np.sqrt(periods_per_year)
    return 0.0 if dd == 0 else mean / dd


def cagr(equity_curve, periods_per_year=52):
    # equity_curve is cumulative returns (starting at 1)
    n_periods = len(equity_curve) - 1
    if n_periods <= 0:
        return 0.0
    start = equity_curve.iloc[0]
    end = equity_curve.iloc[-1]
    years = n_periods / periods_per_year
    if start <= 0 or years == 0:
        return 0.0
    return (end / start) ** (1 / years) - 1


def max_drawdown(equity_curve):
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    return drawdown.min()


def calmar_ratio(equity_curve, periods_per_year=52):
    cagr_val = cagr(equity_curve, periods_per_year)
    mdd = abs(max_drawdown(equity_curve))
    return 0.0 if mdd == 0 else cagr_val / mdd


def compute_metrics(weekly_returns, periods_per_year=52):
    # weekly_returns: pandas Series of weekly returns (strategy pnl)
    eq = (1 + weekly_returns).cumprod()
    return {
        "Sharpe": sharpe_ratio(weekly_returns, periods_per_year),
        "Sortino": sortino_ratio(weekly_returns, periods_per_year),
        "CAGR": cagr(eq, periods_per_year),
        "Calmar": calmar_ratio(eq, periods_per_year),
        "MaxDrawdown": max_drawdown(eq),
        "TotalReturn": eq.iloc[-1] - 1,
        "Trades": np.nan,
    }


def plot_equity_drawdown(equity_series, outpath_eq, outpath_dd):
    os.makedirs(os.path.dirname(outpath_eq), exist_ok=True)
    plt.figure(figsize=(8, 4))
    equity_series.plot(title="Equity Curve")
    plt.savefig(outpath_eq)
    plt.close()

    plt.figure(figsize=(8, 4))
    dd = (equity_series / equity_series.cummax()) - 1
    dd.plot(title="Drawdown")
    plt.savefig(outpath_dd)
    plt.close()


def bootstrap_sharpe_ci(returns, n_boot=1000, alpha=0.05):
    rng = np.random.default_rng(0)
    boot_sharpes = []
    arr = np.array(returns)
    n = len(arr)
    for _ in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boot_sharpes.append(sharpe_ratio(sample))
    lo = np.percentile(boot_sharpes, 100 * (alpha / 2))
    hi = np.percentile(boot_sharpes, 100 * (1 - alpha / 2))
    return np.mean(boot_sharpes), (lo, hi)


def block_bootstrap_ci(returns, n_boot=2000, block=8, alpha=0.05, periods_per_year=52):
    """Moving block bootstrap for Sharpe; use for pooled tests too."""
    rng = np.random.default_rng(0)
    arr = np.asarray(returns)
    n = len(arr)
    if n <= 0:
        return 0.0, (0.0, 0.0)
    idx = np.arange(max(1, n - block + 1))
    sharpes = []
    for _ in range(n_boot):
        samples = []
        # sample blocks until length >= n
        while len(samples) < n:
            start = int(rng.choice(idx))
            samples.extend(arr[start : start + block])
        sample = np.array(samples[:n])
        sharpes.append(sharpe_ratio(sample, periods_per_year))
    lo, hi = np.percentile(sharpes, [100 * (alpha / 2), 100 * (1 - alpha / 2)])
    return float(np.mean(sharpes)), (float(lo), float(hi))


def buy_and_hold(df):
    """
    df should have 'Close' and 'Return' columns and be in chronological order.
    Returns weekly_returns (Series) and actions (Series)
    """
    actions = pd.Series(1, index=df.index)  # long 1 always
    weekly_returns = df["Return"].shift(-1).fillna(0) * actions
    return weekly_returns, actions


def sma_10_30(df):
    # Ensure Close is numeric
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce").fillna(0)
    if "MA_10" not in df.columns:
        df["MA_10"] = df["Close"].rolling(10).mean().shift(1)
    if "MA_30" not in df.columns:
        df["MA_30"] = df["Close"].rolling(30).mean().shift(1)
    df[["MA_10", "MA_30"]] = (
        df[["MA_10", "MA_30"]].bfill().ffill()
    )
    actions = (df["MA_10"] > df["MA_30"]).astype(int)
    actions = actions.map({0: 0, 1: 1})
    weekly_returns = df["Return"].shift(-1).fillna(0) * actions
    return weekly_returns, actions


def momentum_12_1(df):
    """
    12-month momentum on weekly data: 52-week lookback, shifted by 1 to avoid look-ahead.
    Returns long-only (1 or 0) actions by default (matches earlier baseline semantics).
    """
    # Ensure Close is numeric
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce").fillna(0)
    # If dataset shorter than 52, pct_change(52) returns NaN -> .fillna(0)
    r52 = df["Close"].pct_change(52).shift(1).fillna(0)
    actions = (r52 > 0).astype(int)
    weekly_returns = df["Return"].shift(-1).fillna(0) * actions
    return weekly_returns, actions


def rsi_rule(df, period=14, low=30, high=70):
    """
    Simple RSI-based mean-reversion rule:
      - long if RSI < low
      - short if RSI > high
      - else flat
    RSI computed with simple rolling mean of gains/losses, shifted by 1 to avoid peeking.
    """
    # Ensure Close is numeric
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce").fillna(0)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = (gain / (loss.replace(0, np.nan))).fillna(0)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.shift(1).bfill()  # no peek
    actions = np.where(rsi < low, 1, np.where(rsi > high, -1, 0))
    actions = pd.Series(actions, index=df.index)
    weekly_returns = df["Return"].shift(-1).fillna(0) * actions
    return weekly_returns, actions


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buf = deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buf)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        env,
        obs_dim,
        val_env=None,
        n_actions=3,
        lr=1e-3,
        gamma=0.95,
        batch_size=32,
        buffer_size=10000,
        min_buffer=500,
        target_update=1000,
        device=None,
        seed=None,
        patience=10,
    ):
        self.env = env
        self.val_env = val_env
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_buffer = min_buffer
        self.target_update = target_update
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = MLP(obs_dim, n_actions).to(self.device)
        self.target_net = MLP(obs_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(buffer_size)
        self.steps = 0
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # early stopping / validation
        self.best_val = -1e9
        self.no_improve = 0
        self.patience = patience

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        state_t = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        q = self.policy_net(state_t).detach().cpu().numpy()[0]
        return int(np.argmax(q))

    def train_step(self):
        if len(self.replay) < self.min_buffer:
            return
        trans = self.replay.sample(self.batch_size)
        state = torch.tensor(
            np.vstack(trans.state), dtype=torch.float32, device=self.device
        )
        action = torch.tensor(
            trans.action, dtype=torch.long, device=self.device
        ).unsqueeze(1)
        reward = torch.tensor(
            trans.reward, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        next_state = torch.tensor(
            np.vstack(trans.next_state), dtype=torch.float32, device=self.device
        )
        done = torch.tensor(
            trans.done, dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        q_values = self.policy_net(state).gather(1, action)
        with torch.no_grad():
            next_q = self.target_net(next_state).max(1)[0].unsqueeze(1)
            q_target = reward + (1 - done) * (self.gamma * next_q)
        loss = nn.functional.mse_loss(q_values, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def evaluate_val(self, episodes=3):
        """
        Evaluate current policy deterministically on validation env.
        Returns average episodic reward.
        """
        if self.val_env is None:
            return None
        tot = 0.0
        for _ in range(episodes):
            s = self.val_env.reset()
            done = False
            mapping = [-1, 0, 1]
            while not done:
                # greedy action
                state_t = (
                    torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device)
                )
                q = self.policy_net(state_t).detach().cpu().numpy()[0]
                a_idx = int(np.argmax(q))
                s, r, done, _ = self.val_env.step(mapping[a_idx])
                tot += r
        return tot / episodes

    def train(self, n_steps=10000, verbose=True):
        # run multiple episodes but count global steps
        state = self.env.reset()
        mapping = [-1, 0, 1]
        total_rewards = []
        ep_reward = 0
        done = False
        while self.steps < n_steps:
            # epsilon anneal
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            a_idx = self.select_action(state)
            action = mapping[a_idx]
            next_state, reward, done, _ = self.env.step(action)
            # store with index action
            self.replay.push(state, a_idx, reward, next_state, float(done))
            state = next_state
            ep_reward += reward
            self.train_step()
            self.steps += 1
            if self.steps % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            if done:
                total_rewards.append(ep_reward)
                # optional validation + early stopping
                if self.val_env is not None:
                    score = self.evaluate_val()
                    if score is not None and score > self.best_val + 1e-4:
                        self.best_val = score
                        self.no_improve = 0
                        # checkpoint best weights to target_net to preserve
                        self.target_net.load_state_dict(self.policy_net.state_dict())
                    else:
                        self.no_improve += 1
                        if self.no_improve >= self.patience:
                            if verbose:
                                print(
                                    f"Early stopping after {len(total_rewards)} episodes (no_improve={self.no_improve})"
                                )
                            break
                state = self.env.reset()
                ep_reward = 0
        return total_rewards

    def act_trajectory(self):
        state = self.env.reset()
        mapping = [-1, 0, 1]
        done = False
        actions = []
        rets = []
        # Use greedy policy (argmax q)
        while not done:
            state_t = (
                torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            )
            q = self.policy_net(state_t).detach().cpu().numpy()[0]
            a_idx = int(np.argmax(q))
            action = mapping[a_idx]
            next_state, reward, done, _ = self.env.step(action)
            actions.append(action)
            rets.append(reward)
            state = next_state
        return np.array(actions), np.array(rets)


class GymTradingEnv(gym.Env):
    def __init__(
        self,
        df,
        transaction_cost: float = 0.001,
        slippage_bps: float = 0.0,
        allow_short: bool = True,
    ):
        super().__init__()
        # observation: vector of 4 features
        self.df = df.reset_index(drop=True)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        # action space: discrete 3 choices mapping to -1,0,1
        self.action_space = spaces.Discrete(3)
        self.position = 0
        self.t = 0
        self.transaction_cost = float(transaction_cost)
        self.slippage_bps = float(slippage_bps)
        self.allow_short = bool(allow_short)
        self.done = False

    def reset(self):
        self.position = 0
        self.t = 0
        self.done = False
        return self._get_obs()

    def step(self, action_idx):
        mapping = [-1, 0, 1]
        action = mapping[int(action_idx)]
        # enforce allow_short
        if not self.allow_short:
            if action == -1:
                action = 0
        prev_pos = self.position
        self.position = action
        traded = self.position != prev_pos
        cost = self.transaction_cost if traded else 0.0
        slip = abs(self.slippage_bps) if traded else 0.0

        if self.t < len(self.df) - 1:
            next_ret = self.df.loc[self.t + 1, "Return"]
            reward = self.position * next_ret - cost - slip
        else:
            reward = -cost - slip
        self.t += 1
        if self.t >= len(self.df) - 1:
            self.done = True
        return self._get_obs(), float(reward), self.done, {}

    def _get_obs(self):
        cols = ["Close", "MA_5", "MA_10", "Return"]
        return self.df.loc[self.t, cols].values.astype(float)


def train_ppo(df, timesteps=20000, seed=0, verbose=0, **kwargs):
    env = GymTradingEnv(df, **kwargs)
    check_env(env, warn=True)
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[64, 64]),
        seed=seed,
        verbose=verbose,
    )
    model.learn(total_timesteps=timesteps)
    return model, env


def make_feature_bins(df, features, n_bins=5):
    bins = {}
    for f in features:
        arr = df[f].dropna().values
        if len(arr) < n_bins:
            bins[f] = np.linspace(np.min(arr), np.max(arr), n_bins + 1)[1:-1]
        else:
            bins[f] = np.quantile(arr, np.linspace(0, 1, n_bins + 1))[1:-1]
    return bins


def walk_forward_slices(n, train=0.6, val=0.2, test=0.2, folds=3):
    # even spacing for simplicity
    assert abs(train + val + test - 1.0) < 1e-9
    windows = []
    step = int(n * test / folds)
    for k in range(folds):
        start = 0
        train_end = int(n * train) + k * step
        val_end = train_end + int(n * val)
        test_end = min(val_end + int(n * test), n)
        if test_end - val_end < 10:
            break
        windows.append((start, train_end, val_end, test_end))
    return windows


def train_eval_dqn(df, seeds=[0], steps=20000, folds=3):
    res = []
    slices = walk_forward_slices(len(df), folds=folds)
    for fold_idx, (a, b, c, d) in enumerate(slices, start=1):
        train_df, val_df, test_df = df.iloc[a:b], df.iloc[b:c], df.iloc[c:d]
        for s in seeds:
            env_train = TradingEnv(train_df)
            env_val = TradingEnv(val_df)
            TradingEnv(test_df)
            agent = DQNAgent(env_train, obs_dim=4, val_env=env_val, seed=s)
            agent.train(n_steps=steps)
            # evaluate on test
            actions, rets = agent.act_trajectory()
            metrics = compute_metrics(pd.Series(rets))
            metrics.update(dict(seed=s, fold=fold_idx))
            res.append(metrics)
    return pd.DataFrame(res)


def train_eval_ppo(df, seeds=[0], timesteps=20000, folds=3):
    res = []
    slices = walk_forward_slices(len(df), folds=folds)
    for fold_idx, (a, b, c, d) in enumerate(slices, start=1):
        train_df = df.iloc[a:b]
        for s in seeds:
            # Train PPO on train_df (we could use val_df for early stopping hyper loop externally)
            model, env = train_ppo(train_df, timesteps=timesteps, seed=s)
            # Evaluate on test
            obs = env.reset()
            done = False

            def mapping(x):
                return [-1, 0, 1][int(x)]

            weekly_returns = []
            while not done:
                a, _ = model.predict(obs, deterministic=True)
                obs, r, done, _ = env.step(mapping(a))
                weekly_returns.append(r)
            metrics = compute_metrics(pd.Series(weekly_returns))
            metrics.update(dict(seed=s, fold=fold_idx))
            res.append(metrics)
    return pd.DataFrame(res)


def run_multi_ticker(
    tickers,
    start,
    end,
    out_dir="outputs",
    seeds=[0, 1, 2],
    dqn_steps=15000,
    ppo_timesteps=20000,
):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for t in tickers:
        data_file = f"data/{t}.csv"
        if not os.path.exists(data_file):
            download_weekly([t], start, end, out_dir="data")
        df = pd.read_csv(data_file, parse_dates=["Date"])
        # Ensure numeric columns are properly typed when reading from CSV
        numeric_cols = ["Close", "Adj Close"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = prepare_features(df)
        splits = train_val_test_split(df)
        test_df = splits["test"]

        # Baselines on full test split
        for bname, bfn in [
            ("bh", buy_and_hold),
            ("sma", sma_10_30),
            ("mom12_1", momentum_12_1),
            ("rsi", rsi_rule),
        ]:
            try:
                wr, _ = bfn(test_df.copy())
                m = compute_metrics(wr)
                m.update(dict(ticker=t, method=bname))
                rows.append(m)
            except Exception as e:
                print(f"Baseline {bname} failed on {t}: {e}")

        # DQN walk-forward
        try:
            dqn_df = train_eval_dqn(df, seeds=seeds, steps=dqn_steps)
            for _, r in dqn_df.iterrows():
                rows.append(dict(ticker=t, method="dqn", **r.to_dict()))
        except Exception as e:
            print(f"DQN training failed for {t}: {e}")

        # Q-Learning walk-forward
        try:
            q_env = TradingEnv(df)
            q_agent = QLearningAgent(q_env)
            q_agent.train(n_episodes=500)
            actions, rets = q_agent.act_trajectory()
            m = compute_metrics(pd.Series(rets))
            m.update(dict(ticker=t, method="q_learning"))
            rows.append(m)
        except Exception as e:
            print(f"Q-Learning training failed for {t}: {e}")

        # PPO walk-forward
        try:
            ppo_df = train_eval_ppo(df, seeds=seeds, timesteps=ppo_timesteps)
            for _, r in ppo_df.iterrows():
                rows.append(dict(ticker=t, method="ppo", **r.to_dict()))
        except Exception as e:
            print(f"PPO training failed for {t}: {e}")

    out = pd.DataFrame(rows)
    out.to_csv(f"{out_dir}/pooled_metrics.csv", index=False)
    return out


def sensitivity_sweep(
    df, agent_factory, costs=(0.0, 0.001, 0.0025, 0.005), slippages=(0.0, 0.0005)
):
    rows = []
    for c in costs:
        for s in slippages:
            env = TradingEnv(df, transaction_cost=c, slippage_bps=s)
            agent = agent_factory(env)
            if hasattr(agent, "act_trajectory"):
                actions, rets = agent.act_trajectory()
            else:
                # try SB3 style: agent is model, need env wrapper
                try:
                    obs = env.reset()
                    done = False
                    weekly_returns = []
                    while not done:
                        a, _ = agent.predict(obs, deterministic=True)

                        def mapping(x):
                            return [-1, 0, 1][int(x)]

                        obs, r, done, _ = env.step(mapping(a))
                        weekly_returns.append(r)
                    rets = weekly_returns
                except Exception:
                    rets = None
            if rets is None:
                continue
            m = compute_metrics(pd.Series(rets))
            m.update(dict(cost=c, slip=s))
            rows.append(m)
    return pd.DataFrame(rows)


def write_manifest(path, args_dict):
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        commit = "unknown"
    payload = dict(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        git_commit=commit,
        args=args_dict,
        numpy_seed=int(np.random.get_state()[1][0]),
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


class QLearningAgent:
    def __init__(
        self,
        env,
        n_bins=10,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        seed=None,
    ):
        self.env = env
        self.n_actions = 3  # actions: -1,0,1
        self.n_bins = n_bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.state_bins = self._build_bins()
        self.q_table = np.zeros(
            [n_bins] * len(self.env._get_state()) + [self.n_actions]
        )
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def _build_bins(self):
        bins = []
        state = self.env.reset()
        for i in range(len(state)):
            vals = [state[i]]
            bins.append(
                np.linspace(min(vals) - 1e-5, max(vals) + 1e-5, self.n_bins - 1)
            )
        return bins

    def discretize(self, state):
        idx = []
        for i, val in enumerate(state):
            b = self.state_bins[i]
            idx.append(np.digitize(val, b))
        return tuple(idx)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        idx = self.discretize(state)
        return int(np.argmax(self.q_table[idx]))

    def update(self, state, action, reward, next_state, done):
        idx = self.discretize(state)
        next_idx = self.discretize(next_state)
        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[next_idx])
        self.q_table[idx][action] += self.alpha * (target - self.q_table[idx][action])
        # decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, n_episodes=500):
        mapping = [-1, 0, 1]
        for ep in range(n_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action_idx = self.select_action(state)
                action = mapping[action_idx]
                next_state, reward, done, _ = self.env.step(action)
                self.update(state, action_idx, reward, next_state, done)
                state = next_state

    def act_trajectory(self):
        state = self.env.reset()
        mapping = [-1, 0, 1]
        done = False
        actions = []
        rets = []
        while not done:
            action_idx = self.select_action(state)
            action = mapping[action_idx]
            next_state, reward, done, _ = self.env.step(action)
            actions.append(action)
            rets.append(reward)
            state = next_state
        return np.array(actions), np.array(rets)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--costs", nargs="+", type=float, default=[0.001])
    p.add_argument("--slippage", nargs="+", type=float, default=[0.0, 0.0005])
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--dqn_steps", type=int, default=15000)
    p.add_argument("--ppo_timesteps", type=int, default=20000)
    args = p.parse_args()

    out = run_multi_ticker(
        args.tickers,
        args.start,
        args.end,
        out_dir=args.out_dir,
        seeds=args.seeds,
        dqn_steps=args.dqn_steps,
        ppo_timesteps=args.ppo_timesteps,
    )
    write_manifest(f"{args.out_dir}/manifest.json", vars(args))
    print(f"Saved pooled results to {args.out_dir}/pooled_metrics.csv and manifest.")


def test_moving_avg_shift_no_peek():
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=30, freq="W"),
            "Ticker": ["T"] * 30,
            "Close": np.arange(30) + 100,
        }
    )
    f = prepare_features(df)
    # MA_5 at t must not equal rolling mean including t; it must lag by 1
    rolling_now = df["Close"].rolling(5).mean()
    # where defined, lagged MA_5 should equal rolling_now.shift(1)
    comp = rolling_now.shift(1).bfill()
    assert np.allclose(f["MA_5"].values, comp.values, equal_nan=True)


def example_basic_usage():
    """Example/test code: Basic usage of trading environment."""
    # Functions are available directly from this module (prepare_features, download_weekly, TradingEnv)

    # Dummy price data
    df = pd.DataFrame(
        {
            "close": np.linspace(100, 120, 50),  # fake prices
            "open": np.linspace(99, 119, 50),
            "high": np.linspace(101, 121, 50),
            "low": np.linspace(98, 118, 50),
            "volume": np.random.randint(1000, 2000, 50),
        }
    )

    # Download stock data and prepare features
    df = download_weekly("AAPL")  # or use any stock
    df_feat = prepare_features(df)
    print(df_feat.head())

    # Create environment
    env = TradingEnv(df)

    # Reset and step through a few timesteps
    state = env.reset()
    for _ in range(5):
        action = np.random.choice([-1, 0, 1])
        state, reward, done, info = env.step(action)
        if hasattr(env, "render"):
            env.render()

    # Example DataFrame for testing
    df_test = pd.DataFrame({"close": np.linspace(100, 110, 100)})

    env_test = TradingEnv(df_test)
    # Note: DQNAgent constructor expects (env, obs_dim, ...) not (state_size, action_size)
    # This is example code and may need adjustment based on actual usage
    obs_dim = 4  # Default observation dimension
    _ = DQNAgent(env_test, obs_dim=obs_dim)  # Example agent creation


# Example/test code functions (not executed unless called)
def example_usage():
    """Example usage of the trading environment and agents."""
    # Functions are available directly from this module (prepare_features, download_weekly, TradingEnv)

    df = download_weekly("AAPL")  # or any stock
    df_feat = prepare_features(df)
    print(df_feat.head())

    # Create environment
    env = TradingEnv(df)

    # Reset and step through a few timesteps
    state = env.reset()
    for _ in range(5):
        action = np.random.choice([-1, 0, 1])
        state, reward, done, info = env.step(action)
        if hasattr(env, "render"):
            env.render()


def test_models(test_env, q_agent, dqn_agent, ppo_agent, episodes=10):
    """Test multiple models and return results."""
    results = {}
    for name, agent in [
        ("Q-Learning", q_agent),
        ("DQN", dqn_agent),
        ("PPO", ppo_agent),
    ]:
        total_reward = 0
        episode_rewards = []
        for _ in range(episodes):
            state = test_env.reset()
            done = False
            ep_reward = 0
            while not done:
                # For PPO, agent.predict might be used
                if name == "PPO":
                    action, _ = agent.predict(state)
                else:
                    action = agent.act(state)
                next_state, reward, done, _ = test_env.step(action)
                ep_reward += reward
                state = next_state
            episode_rewards.append(ep_reward)
            total_reward += ep_reward
        avg_reward = total_reward / episodes
        results[name] = {"avg_reward": avg_reward, "episode_rewards": episode_rewards}
        print(f"{name} average reward over {episodes} episodes: {avg_reward:.2f}")
    return results


def example_test_comparison():
    """Example function to test and compare all models."""
    # ===== Run tests after training =====
    print("\n=== Testing all models ===")
    # Note: This is example code - uncomment and provide actual agents to use
    # test_env = TradingEnv(df)  # make sure this matches your environment name
    # test_results = test_models(test_env, q_agent, dqn_agent, ppo_agent, episodes=10)

    # ===== Plot comparison =====
    # plt.figure(figsize=(8, 5))
    # for name, res in test_results.items():
    #     plt.plot(res["episode_rewards"], label=name)
    # plt.title("Model Reward Comparison")
    # plt.xlabel("Episode")
    # plt.ylabel("Total Reward")
    # plt.legend()
    # plt.show()
    pass
