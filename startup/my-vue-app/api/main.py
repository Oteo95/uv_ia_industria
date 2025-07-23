from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import os
import httpx
import asyncio

app = FastAPI()

# Strategy Registry
AVAILABLE_STRATEGIES = [
    {
        "name": "Moving Average Crossover",
        "endpoint": "http://localhost:8000/ma-crossover-strategy",
        "description": "Simple moving average crossover strategy",
        "parameters": {
            "short_window": {"type": "int", "default": 20, "min": 1, "max": 50},
            "long_window": {"type": "int", "default": 30, "min": 2, "max": 100}
        }
    },
    {
        "name": "Moving Average Crossover 2",
        "endpoint": "http://localhost:8000/ma-crossover-strategy",
        "description": "Simple moving average crossover strategy",
        "parameters": {
            "short_window": {"type": "int", "default": 20, "min": 1, "max": 50},
            "long_window": {"type": "int", "default": 30, "min": 2, "max": 100}
        }
    },
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

class Price(BaseModel):
    date: str
    ticker: str
    open: float
    high: float
    low: float
    close: float
    volume: float

class BacktestRequest(BaseModel):
    prices: List[Price]

class MultiStrategyRequest(BaseModel):
    prices: List[Price]
    strategies: List[str] = []  # List of strategy endpoints to run
    initial_cash: Optional[float] = 10000
    parameters: Optional[Dict[str, Any]] = {}

class Trade:
    def __init__(self, date, ticker, action, price, shares, cash_after):
        self.date = date
        self.ticker = ticker
        self.action = action
        self.price = price
        self.shares = shares
        self.cash_after = cash_after

    def as_dict(self):
        return {
            "date": self.date,
            "ticker": self.ticker,
            "action": self.action,
            "price": self.price,
            "shares": self.shares,
            "cash_after": self.cash_after
        }

class PortfolioManager:
    def __init__(self, initial_cash=10000, max_positions=10, max_per_stock=0.5):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}    # stock: shares
        self.trades = []
        self.max_positions = max_positions    # max number of stocks to hold at once
        self.max_per_stock = max_per_stock    # max % of portfolio per stock

    def buy(self, date, stock, price, desired_money):
        investable = min(desired_money, self.max_per_stock * self.current_value({stock: price}), self.cash)
        shares = int(investable // price)
        if shares < 1:
            self.trades.append(Trade(date, stock, "BUY (FAILED)", price, 0, self.cash))
            return
        total_cost = shares * price
        if self.cash >= total_cost:
            self.cash -= total_cost
            self.positions[stock] = self.positions.get(stock, 0) + shares
            self.trades.append(Trade(date, stock, "BUY", price, shares, self.cash))
        else:
            self.trades.append(Trade(date, stock, "BUY (FAILED)", price, 0, self.cash))

    def sell(self, date, stock, price):
        shares = self.positions.get(stock, 0)
        if shares > 0:
            total_return = shares * price
            self.cash += total_return
            self.trades.append(Trade(date, stock, "SELL", price, shares, self.cash))
            self.positions[stock] = 0
        else:
            self.trades.append(Trade(date, stock, "SELL (FAILED)", price, 0, self.cash))

    def current_value(self, prices):
        # prices: dict stock->price
        return self.cash + sum(self.positions.get(stock, 0) * price for stock, price in prices.items())

    def get_trades(self):
        return [t.as_dict() for t in self.trades]

def generate_signals(prices_df):
    # Simple moving average crossover strategy for each stock
    signals = []
    for stock in prices_df['ticker'].unique():
        df = prices_df[prices_df['ticker'] == stock].copy()
        df = df.sort_values('date')
        df['ma_short'] = df['close'].rolling(window=20).mean()
        df['ma_long'] = df['close'].rolling(window=30).mean()
        prev_signal = None
        for i, row in df.iterrows():
            # Buy when short MA crosses above long MA, sell when crosses below
            if pd.notnull(row['ma_short']) and pd.notnull(row['ma_long']):
                if row['ma_short'] > row['ma_long'] and prev_signal != 'BUY':
                    signals.append({
                        "date": row['date'],
                        "ticker": stock,
                        "action": "BUY",
                        "quantity": 10000    # Example: always try to invest $10000
                    })
                    prev_signal = 'BUY'
                elif row['ma_short'] < row['ma_long'] and prev_signal != 'SELL':
                    signals.append({
                        "date": row['date'],
                        "ticker": stock,
                        "action": "SELL",
                        "quantity": 0
                    })
                    prev_signal = 'SELL'
    return pd.DataFrame(signals)

def run_benchmark(prices_df, initial_cash=10000):
    """
    Benchmark strategy: Equally balanced buy-and-hold
    Buy equal amounts of each stock at the beginning and never sell
    """
    stocks = prices_df['ticker'].unique()
    price_pivot = prices_df.pivot(index='date', columns='ticker', values='close').ffill()
    all_dates = price_pivot.index.sort_values()
    
    # Calculate equal allocation per stock
    cash_per_stock = initial_cash / len(stocks)
    
    # Get first day prices for initial purchase
    first_day = all_dates[0]
    first_day_prices = price_pivot.loc[first_day].to_dict()
    
    # Calculate shares to buy for each stock
    benchmark_positions = {}
    benchmark_trades = []
    remaining_cash = initial_cash
    
    for stock in stocks:
        price = first_day_prices[stock]
        if pd.notnull(price) and price > 0:
            shares = int(cash_per_stock // price)
            if shares > 0:
                cost = shares * price
                benchmark_positions[stock] = shares
                remaining_cash -= cost
                benchmark_trades.append({
                    "date": first_day.strftime('%Y-%m-%d'),
                    "ticker": stock,
                    "action": "BUY (BENCHMARK)",
                    "price": price,
                    "shares": shares,
                    "cash_after": remaining_cash
                })
    
    # Calculate portfolio values over time
    benchmark_values = []
    benchmark_dates = []
    
    for dt in all_dates:
        today_prices = price_pivot.loc[dt].to_dict()
        portfolio_value = remaining_cash
        
        for stock, shares in benchmark_positions.items():
            price = today_prices.get(stock, 0)
            if pd.notnull(price):
                portfolio_value += shares * price
        
        benchmark_values.append(portfolio_value)
        benchmark_dates.append(dt.strftime('%Y-%m-%d'))
    
    # Calculate benchmark metrics
    total_return = 100 * (benchmark_values[-1] - initial_cash) / initial_cash
    returns = np.diff(benchmark_values) / benchmark_values[:-1]
    annualized_return = ((benchmark_values[-1] / initial_cash) ** (252/len(benchmark_values)) - 1) * 100 if len(benchmark_values) > 1 else 0
    max_drawdown = 100 * ((np.maximum.accumulate(benchmark_values) - benchmark_values) / np.maximum.accumulate(benchmark_values)).max()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    
    return {
        "dates": benchmark_dates,
        "portfolio_values": benchmark_values,
        "trade_log": benchmark_trades,
        "summary": {
            "total_return": round(total_return, 2),
            "annualized_return": round(annualized_return, 2),
            "max_drawdown": round(-max_drawdown, 2),
            "sharpe_ratio": round(sharpe_ratio, 2)
        }
    }

@app.get("/prices")
async def get_prices():
    """Read prices.csv from the public folder and return the data"""
    try:
        # Get the path to the public folder relative to the API
        csv_path = os.path.join(".", "data", "prices.csv")
        
        # Check if file exists
        if not os.path.exists(csv_path):
            return {"error": "prices.csv file not found"}
        
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Convert to list of dictionaries
        prices_data = df.to_dict('records')
        
        return {
            "success": True,
            "data": prices_data,
            "csv_content": df.to_csv(index=False)
        }
    except Exception as e:
        return {"error": f"Failed to read CSV file: {str(e)}"}

@app.get("/strategies")
async def get_available_strategies():
    """Get list of available strategies"""
    return {"strategies": AVAILABLE_STRATEGIES}

@app.post("/ma-crossover-strategy")
async def ma_crossover_strategy(data: BacktestRequest):
    """Moving Average Crossover Strategy - Built-in implementation"""
    prices_df = pd.DataFrame([p.dict() for p in data.prices])
    prices_df['date'] = pd.to_datetime(prices_df['date'])
    prices_df = prices_df.sort_values(['date', 'ticker'])

    # Run the trading strategy (same as current backtest)
    signals_df = generate_signals(prices_df)
    stocks = prices_df['ticker'].unique()
    price_pivot = prices_df.pivot(index='date', columns='ticker', values='close').ffill()
    all_dates = price_pivot.index.sort_values()
    signals_map = {(pd.to_datetime(row['date']), row['ticker']): (row['action'], row['quantity']) for i, row in signals_df.iterrows()}

    pm = PortfolioManager(initial_cash=10000, max_positions=10, max_per_stock=0.5)
    portfolio_values = []
    cash_values = []
    dates = []

    for dt in all_dates:
        today_prices = price_pivot.loc[dt].to_dict()
        for stock in stocks:
            price = today_prices.get(stock)
            signal = signals_map.get((dt, stock))
            if signal:
                action, quantity = signal
                if action == "BUY":
                    pm.buy(dt.strftime('%Y-%m-%d'), stock, price, quantity)
                elif action == "SELL":
                    pm.sell(dt.strftime('%Y-%m-%d'), stock, price)
        value = pm.current_value(today_prices)
        portfolio_values.append(value)
        cash_values.append(pm.cash)
        dates.append(dt.strftime('%Y-%m-%d'))

    initial_cash = pm.initial_cash
    total_return = 100 * (portfolio_values[-1] - initial_cash) / initial_cash
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    annualized_return = ((portfolio_values[-1] / initial_cash) ** (252/len(portfolio_values)) - 1) * 100 if len(portfolio_values) > 1 else 0
    max_drawdown = 100 * ((np.maximum.accumulate(portfolio_values) - portfolio_values) / np.maximum.accumulate(portfolio_values)).max()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

    return {
        "strategy_name": "Moving Average Crossover",
        "dates": dates,
        "portfolio_values": portfolio_values,
        "cash_values": cash_values,
        "trade_log": pm.get_trades(),
        "summary": {
            "total_return": round(total_return, 2),
            "annualized_return": round(annualized_return, 2),
            "max_drawdown": round(-max_drawdown, 2),
            "sharpe_ratio": round(sharpe_ratio, 2)
        },
        "parameters_used": {
            "short_window": 20,
            "long_window": 30
        }
    }

async def call_strategy_endpoint(endpoint: str, data: dict):
    """Call a strategy endpoint and return results"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(endpoint, json=data, timeout=30.0)
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": f"Strategy endpoint returned {response.status_code}",
                    "strategy_name": f"Strategy at {endpoint}",
                    "success": False
                }
    except Exception as e:
        return {
            "error": f"Failed to call strategy endpoint: {str(e)}",
            "strategy_name": f"Strategy at {endpoint}",
            "success": False
        }

@app.post("/multi-strategy-backtest")
async def multi_strategy_backtest(data: MultiStrategyRequest):
    """Run multiple strategies and compare results"""
    
    # Prepare data for strategy endpoints
    strategy_data = {
        "prices": [p.model_dump() for p in data.prices],
        "initial_cash": data.initial_cash,
        "parameters": data.parameters
    }
    
    # If no strategies specified, use all available strategies
    if not data.strategies:
        strategy_endpoints = [s["endpoint"] for s in AVAILABLE_STRATEGIES]
    else:
        strategy_endpoints = data.strategies
    
    # Run all strategies concurrently
    tasks = [call_strategy_endpoint(endpoint, strategy_data) for endpoint in strategy_endpoints]
    strategy_results = await asyncio.gather(*tasks)
    
    # Run benchmark
    prices_df = pd.DataFrame(strategy_data["prices"])
    prices_df['date'] = pd.to_datetime(prices_df['date'])
    prices_df = prices_df.sort_values(['date', 'ticker'])
    benchmark_results = run_benchmark(prices_df, data.initial_cash)
    
    # Calculate comparisons
    comparisons = []
    for result in strategy_results:
        if "error" not in result and "summary" in result:
            strategy_return = result["summary"]["total_return"]
            benchmark_return = benchmark_results["summary"]["total_return"]
            outperformance = strategy_return - benchmark_return
            
            comparisons.append({
                "strategy_name": result.get("strategy_name", "Unknown"),
                "outperformance": round(outperformance, 2),
                "beats_benchmark": strategy_return > benchmark_return
            })
    
    return {
        "strategies": strategy_results,
        "benchmark": benchmark_results,
        "comparisons": comparisons,
        "summary": {
            "total_strategies": len(strategy_results),
            "successful_strategies": len([r for r in strategy_results if "error" not in r]),
            "strategies_beating_benchmark": len([c for c in comparisons if c["beats_benchmark"]])
        }
    }

@app.post("/backtest")
async def backtest(data: BacktestRequest):

    prices_df = pd.DataFrame([p.model_dump() for p in data.prices])
    prices_df['date'] = pd.to_datetime(prices_df['date'])
    prices_df = prices_df.sort_values(['date', 'ticker'])

    # Run the trading strategy
    signals_df = generate_signals(prices_df)

    stocks = prices_df['ticker'].unique()
    price_pivot = prices_df.pivot(index='date', columns='ticker', values='close').ffill()
    all_dates = price_pivot.index.sort_values()
    signals_map = {(pd.to_datetime(row['date']), row['ticker']): (row['action'], row['quantity']) for i, row in signals_df.iterrows()}
    print(signals_map)
    pm = PortfolioManager(initial_cash=10000, max_positions=10, max_per_stock=0.5)
    portfolio_values = []
    cash_values = []
    dates = []

    for dt in all_dates:
        today_prices = price_pivot.loc[dt].to_dict()
        for stock in stocks:
            price = today_prices.get(stock)
            signal = signals_map.get((dt, stock))
            if signal:
                action, quantity = signal
                if action == "BUY":
                    pm.buy(dt.strftime('%Y-%m-%d'), stock, price, quantity)
                elif action == "SELL":
                    pm.sell(dt.strftime('%Y-%m-%d'), stock, price)
        value = pm.current_value(today_prices)
        portfolio_values.append(value)
        cash_values.append(pm.cash)
        dates.append(dt.strftime('%Y-%m-%d'))

    initial_cash = pm.initial_cash
    total_return = 100 * (portfolio_values[-1] - initial_cash) / initial_cash
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    annualized_return = ((portfolio_values[-1] / initial_cash) ** (252/len(portfolio_values)) - 1) * 100 if len(portfolio_values) > 1 else 0
    max_drawdown = 100 * ((np.maximum.accumulate(portfolio_values) - portfolio_values) / np.maximum.accumulate(portfolio_values)).max()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

    # Run the benchmark strategy
    benchmark_results = run_benchmark(prices_df, initial_cash)

    # Calculate outperformance
    strategy_final_return = total_return
    benchmark_final_return = benchmark_results["summary"]["total_return"]
    outperformance = strategy_final_return - benchmark_final_return

    return {
        "strategy": {
            "dates": dates,
            "portfolio_values": portfolio_values,
            "cash_values": cash_values,
            "trade_log": pm.get_trades(),
            "summary": {
                "total_return": round(total_return, 2),
                "annualized_return": round(annualized_return, 2),
                "max_drawdown": round(-max_drawdown, 2),
                "sharpe_ratio": round(sharpe_ratio, 2)
            }
        },
        "benchmark": benchmark_results,
        "comparison": {
            "outperformance": round(outperformance, 2),
            "strategy_beats_benchmark": strategy_final_return > benchmark_final_return
        },
        # Keep backward compatibility
        "dates": dates,
        "portfolio_values": portfolio_values,
        "cash_values": cash_values,
        "trade_log": pm.get_trades(),
        "summary": {
            "total_return": round(total_return, 2),
            "annualized_return": round(annualized_return, 2),
            "max_drawdown": round(-max_drawdown, 2),
            "sharpe_ratio": round(sharpe_ratio, 2)
        }
    }
