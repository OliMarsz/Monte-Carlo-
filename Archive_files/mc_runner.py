import importlib
import numpy as np
import pandas as pd
from contextlib import contextmanager, redirect_stdout
import io
from data_providers import path_to_df

"""
Main runner. See note below for usage instruction and assumptions
"""

def _resolve_dotted(name):
    
    mod_name, attr = name.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    return mod, attr


@contextmanager
def patch_dotted_functions(dotted_names, fake_df):
    originals = []
    try:
        for dotted in dotted_names:
            mod, attr = _resolve_dotted(dotted)
            orig = getattr(mod, attr)
            originals.append((mod, attr, orig))

            def make_fake():
                def fake(*args, **kwargs):
                    return fake_df.copy()
                return fake

            setattr(mod, attr, make_fake())
        yield
    finally:
        for mod, attr, orig in originals:
            setattr(mod, attr, orig)


@contextmanager
def quiet_prints(enabled=True):
    if not enabled:
        yield
        return
    buf = io.StringIO()
    with redirect_stdout(buf):
        yield


class MCRunner:
    """
    Runs an existing bot/backtest on Monte Carlo paths 
    by patching API calls to return fake data. Will be turned into a nicer
    interface for running experiments on bots in the future,
    but for now it's a bit hacky and specific. Required Panda dataframe output from APIs is assumed,
    and the bot's entrypoint function must be specified.
    You can specify other APIs to patch by passing a list of dotted names to the constructor.
    """

    def __init__(self, *, bot_cls, bot_kwargs, entrypoint, apis=("yfinance.download",),
                 df_start="2020-01-01", df_freq="B", quiet=True):
        self.bot_cls = bot_cls
        self.bot_kwargs = dict(bot_kwargs)
        self.entrypoint = entrypoint
        self.apis = list(apis)
        self.df_start = df_start
        self.df_freq = df_freq
        self.quiet = quiet

    def run_one(self, path, *, entry_kwargs=None):
        entry_kwargs = entry_kwargs or {}
        df = path_to_df(path, start=self.df_start, freq=self.df_freq)

        bot = self.bot_cls(**self.bot_kwargs)
        fn = getattr(bot, self.entrypoint)

        with patch_dotted_functions(self.apis, df), quiet_prints(self.quiet):
            return fn(**entry_kwargs)

    def run_paths(self, paths, *, entry_kwargs=None):
        entry_kwargs = entry_kwargs or {}
        results = []
        for i in range(paths.shape[0]):
            results.append(self.run_one(paths[i], entry_kwargs=entry_kwargs))
        return results

    def run_paths_extract(self, paths, *, entry_kwargs=None,
                      returns_key,
                      position_key=None,
                      initial_capital=10000):
        
        ###Runs bot on each path and extracts returns + optional position using key named
       
        entry_kwargs = entry_kwargs or {}
        outs = self.run_paths(paths, entry_kwargs=entry_kwargs)

        n_sims, n_pts = paths.shape
        n_steps = n_pts - 1

        equity = np.zeros((n_sims, n_steps + 1))
        equity[:, 0] = initial_capital
        pos = None if position_key is None else np.zeros((n_sims, n_steps))

        for i, out in enumerate(outs):
            if returns_key not in out:
                raise KeyError(f"{returns_key} not found in bot output")

            sr = out[returns_key]
            rets = np.asarray(sr.fillna(0.0), dtype=float)[:n_steps]
            equity[i, 1:] = initial_capital * np.cumprod(1.0 + rets)

            if position_key is not None:
                if position_key not in out:
                    raise KeyError(f"{position_key} not found in bot output")
                p = np.asarray(out[position_key].fillna(0.0), dtype=float)[:n_steps]
                pos[i] = p

        return {"outs": outs, "equity": equity, "position": pos}