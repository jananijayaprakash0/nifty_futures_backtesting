import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Loading Data
df = pd.read_excel("nifty_data.xlsx")

df['Date'] = pd.to_datetime(df['Date'])

# Calculating Indicators
df['Return'] = df['Close'].pct_change()
df['MA5'] = df['Close'].rolling(5).mean()
df['MA20'] = df['Close'].rolling(20).mean()
df['Volatility'] = df['Close'].rolling(10).std()

# Generate trading signals
df['Signal'] = 0
df.loc[df['MA5'] > df['MA20'], 'Signal'] = 1
df.loc[df['MA5'] < df['MA20'], 'Signal'] = -1
df['Position'] = df['Signal'].shift(1)

# Strategy performance
df['Strategy_Return'] = df['Position'] * df['Return']
df['Cumulative_Market'] = (1 + df['Return']).cumprod()
df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod()

# Performance metrics
total_return = df['Cumulative_Strategy'].iloc[-1] - 1
avg_daily_return = df['Strategy_Return'].mean()
volatility = df['Strategy_Return'].std()
sharpe_ratio = avg_daily_return / volatility * np.sqrt(252)
win_ratio = len(df[df['Strategy_Return'] > 0]) / len(df[df['Strategy_Return'].notna()])

print("----- PERFORMANCE SUMMARY -----")
print(f"Total Return: {total_return:.2%}")
print(f"Average Daily Return: {avg_daily_return:.4%}")
print(f"Volatility: {volatility:.4%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Win Ratio: {win_ratio:.2%}")

# Plot
plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Cumulative_Market'], label='Market Return', linewidth=2)
plt.plot(df['Date'], df['Cumulative_Strategy'], label='Strategy Return', linewidth=2)
plt.title('NIFTY Futures – Strategy vs Market Performance')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.show()

#  top rows and summary
print(df[['Date','Close','MA5','MA20','Volatility']].head(10))
print("\nNull counts:\n", df[['Return','MA5','MA20','Volatility','Position','Strategy_Return']].isna().sum())
df = df.dropna(subset=['MA5','MA20','Volatility']).reset_index(drop=True)

#  functions
def annualized_return(series):
    # series is cumulative returns series (1 + r).cumprod()
    total_periods = len(series)
    total_return = series.iloc[-1] - 1
    years = total_periods / 252
    return (1 + total_return) ** (1/years) - 1

def max_drawdown(cum_returns):
    peak = cum_returns.cummax()
    dd = (cum_returns / peak) - 1
    return dd.min(), (dd.idxmin(), dd.min())

cum_strat = df['Cumulative_Strategy']
daily_strat = df['Strategy_Return'].dropna()

cagr = annualized_return(cum_strat)
ann_vol = daily_strat.std() * np.sqrt(252)
sharpe = (daily_strat.mean() / daily_strat.std()) * np.sqrt(252) if daily_strat.std() != 0 else np.nan
dd, dd_info = max_drawdown(cum_strat)

print(f"CAGR: {cagr:.2%}")
print(f"Annualized volatility: {ann_vol:.2%}")
print(f"Sharpe (ann): {sharpe:.2f}")
print(f"Max Drawdown: {dd:.2%}, at index {dd_info[0]}")


# Price + MAs + signals (markers)
plt.figure(figsize=(14,5))
plt.plot(df['Date'], df['Close'], label='Close')
plt.plot(df['Date'], df['MA5'], label='MA5', alpha=0.8)
plt.plot(df['Date'], df['MA20'], label='MA20', alpha=0.8)
# markers for buy signals
buys = df[df['Signal']==1]
sells = df[df['Signal']==-1]
plt.scatter(buys['Date'], buys['Close'], marker='^', color='g', s=40, label='Buy')
plt.scatter(sells['Date'], sells['Close'], marker='v', color='r', s=40, label='Sell')
plt.legend()
plt.title('Price and Moving Averages with Signals')
plt.savefig('price_ma_signals.png', dpi=200, bbox_inches='tight')
plt.show()

# Cumulative returns
plt.figure(figsize=(12,5))
plt.plot(df['Date'], df['Cumulative_Market'], label='Market')
plt.plot(df['Date'], df['Cumulative_Strategy'], label='Strategy')
plt.legend()
plt.title('Cumulative Returns')
plt.savefig('cumulative_returns.png', dpi=200, bbox_inches='tight')
plt.show()

# Drawdown plot
peak = df['Cumulative_Strategy'].cummax()
drawdown = df['Cumulative_Strategy']/peak - 1
plt.figure(figsize=(12,4))
plt.fill_between(df['Date'], drawdown, 0, color='red')
plt.title('Strategy Drawdown')
plt.savefig('drawdown.png', dpi=200, bbox_inches='tight')
plt.show()

# create trades from position changes
df['PositionChange'] = df['Position'].diff()
entries = df[df['PositionChange'] == 1].copy()
exits = df[df['PositionChange'] == -1].copy()

# align entries and exits - simple approach assumes long-only strategy
trades = []
for i in range(len(entries)):
    entry_idx = entries.index[i]
    # find next exit after entry
    exit_idx = exits[exits.index > entry_idx].index
    if len(exit_idx) == 0:
        break
    exit_idx = exit_idx[0]
    entry_date = df.loc[entry_idx, 'Date']
    exit_date = df.loc[exit_idx, 'Date']
    entry_price = df.loc[entry_idx, 'Close']
    exit_price = df.loc[exit_idx, 'Close']
    pnl = (exit_price / entry_price) - 1
    duration = (exit_date - entry_date).days
    trades.append({
        'entry_date': entry_date, 'exit_date': exit_date,
        'entry_price': entry_price, 'exit_price': exit_price,
        'pnl': pnl, 'duration_days': duration
    })

trades_df = pd.DataFrame(trades)
print(trades_df.head(10))
trades_df.to_csv('trades_log.csv', index=False)

# regime thresholds by volatility percentiles
low_th = df['Volatility'].quantile(0.33)
high_th = df['Volatility'].quantile(0.67)

def regime(vol):
    if vol <= low_th:
        return 'Low'
    if vol <= high_th:
        return 'Medium'
    return 'High'

df['Vol_Regime'] = df['Volatility'].apply(regime)

# group performance by regime
group = df.groupby('Vol_Regime').agg(
    days=('Date','count'),
    strat_return=('Strategy_Return', lambda x: (1+x).prod()-1),
    avg_daily_return=('Strategy_Return','mean'),
    vol=('Strategy_Return','std')
).reset_index()
print(group)

# cumulative plots per regime
for r in ['Low','Medium','High']:
    subset = df[df['Vol_Regime']==r]
    if subset.empty:
        continue
    cr = (1 + subset['Strategy_Return'].fillna(0)).cumprod()
    plt.plot(subset['Date'], cr, label=r)
plt.legend(); plt.title('Cumulative Strategy Returns by Volatility Regime'); plt.show()

df.to_csv('nifty_strategy_full_output.csv', index=False)
group.to_csv('performance_by_regime.csv', index=False)
