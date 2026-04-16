import requests
import zipfile
import io
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg


class FamaFrenchDownloader:
    """下载并处理Fama - French五因子数据（每日）。"""
    FF5_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"

    @staticmethod
    def download_ff5():
        """从网络获取并提取F - F五因子数据。"""
        response = requests.get(FamaFrenchDownloader.FF5_URL)
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            file_name = z.namelist()[0]
            with z.open(file_name) as f:
                return f.read().decode("utf - 8").splitlines()

    @staticmethod
    def parse_ff5_data():
        """提取并清理F - F五因子数据。"""
        lines = FamaFrenchDownloader.download_ff5()
        # 仅保留数据行（以8位日期开头）
        data_lines = [line for line in lines if re.match(r'^\s*\d{8}', line)]
        # 读取到DataFrame
        df = pd.read_csv(io.StringIO("\n".join(data_lines)), sep=r"\s*,\s*", header=None, engine='python')
        # 分配列名
        df.columns = ["Date", "MKT_RF", "SMB", "HML", "RMW", "CMA", "RF"]
        # 转换日期列
        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d", errors="coerce")
        # 转换数值列
        df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
        return df


class MomentumDownloader:
    """下载并处理Fama - French动量因子数据（每日）。"""
    MOM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"

    @staticmethod
    def download_momentum():
        """从网络获取并提取动量因子数据。"""
        response = requests.get(MomentumDownloader.MOM_URL)
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            file_name = z.namelist()[0]
            with z.open(file_name) as f:
                return f.read().decode("utf - 8").splitlines()

    @staticmethod
    def parse_momentum_data():
        """提取并清理动量因子数据（每日）。"""
        lines = MomentumDownloader.download_momentum()
        # 仅保留数据行（以8位日期YYYYMMDD开头）
        data_lines = [line for line in lines if re.match(r'^\s*\d{8}', line)]
        # 读取到DataFrame
        df = pd.read_csv(io.StringIO("\n".join(data_lines)), sep=r"\s*,\s*", header=None, engine='python')
        # 分配列名
        df.columns = ["Date", "MOM"]
        # 转换日期列
        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d", errors="coerce")
        # 将MoM转换为数值
        df["MOM"] = pd.to_numeric(df["MOM"], errors="coerce")
        # 删除无效行
        df.dropna(subset=["Date", "MOM"], inplace=True)
        return df


class StockDataFetcher:
    """从东方财富下载并处理股票数据。"""

    @staticmethod
    def get_stock_data(tickers, weights, start_date="1990-01-01"):
        """
        获取多个股票代码的数据并计算加权每日回报

        参数:
        tickers (list): 股票代码列表（例如，["AAPL", "MSFT"]）
        weights (list): 权重列表（例如，[0.3, 0.7]）。必须与tickers长度相同
        start_date (str): 历史数据的开始日期。

        返回:
        DataFrame: 包含日期、股票价格和投资组合每日回报。
        """
        if len(tickers) != len(weights):
            raise ValueError("股票代码和权重必须具有相同的长度")

        all_stock_data = {}
        for ticker in tickers:
            # 区分上证和深证股票代码
            if ticker.startswith('6'):
                secid = f"1.{ticker}"
            else:
                secid = f"0.{ticker}"
            url = f'https://27.push2his.eastmoney.com/api/qt/stock/kline/get?secid={secid}&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61&klt=101&fqt=0&end=20500101&lmt=10000'
            try:
                response = requests.get(url)
                data = response.json()
                if 'data' in data and 'klines' in data['data']:
                    r = data['data']['klines']
                    l = [i.split(',') for i in r]
                    df = pd.DataFrame(l)
                    df = df[[0, 2]]
                    df.columns = ['Date', ticker]
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                    # 将股票价格列转换为数值类型
                    df[ticker] = pd.to_numeric(df[ticker], errors='coerce')
                    all_stock_data[ticker] = df
                else:
                    print(f"获取 {ticker} 数据时出错: 未找到有效数据")
            except Exception as e:
                print(f"获取 {ticker} 数据时出错: {e}")

        if not all_stock_data:
            raise ValueError("未获取到任何有效股票数据，请检查股票代码或网络连接。")

        combined_df = pd.concat(all_stock_data.values(), axis=1)
        combined_df.reset_index(inplace=True)

        # 计算每日回报。
        returns_df = combined_df[tickers].pct_change() * 100
        # 计算加权投资组合回报。
        portfolio_return = returns_df.dot(weights)

        # 合并数据。
        final_df = combined_df.copy()
        final_df["Portfolio Return"] = portfolio_return
        final_df.reset_index(inplace=True)

        return final_df


# 下载因子数据
ff5_df = FamaFrenchDownloader.parse_ff5_data()
mom_df = MomentumDownloader.parse_momentum_data()

# 下载股票数据
tickers = ["002625","600000"]  # 示例股票代码，根据实际情况修改
weights = [0.4,0.6]
stock_df = StockDataFetcher.get_stock_data(tickers, weights)

# 合并数据
merged_factors_df = pd.merge(ff5_df, mom_df, on="Date", how="outer")
final_df = pd.merge(stock_df, merged_factors_df, on="Date", how="inner")
final_df.dropna(inplace=True)

# 可视化因子数据和回报
plt.style.use('dark_background')
# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
final_df['Date'] = pd.to_datetime(final_df['Date'])
try:
    stock_ticker_cols = tickers
except NameError:
    stock_ticker_cols = []
exclude_cols = ['Date', 'Adj Close'] + stock_ticker_cols
exclude_cols += [col for col in final_df.columns if 'Daily Return' in col]
cols_to_plot = [col for col in final_df.columns if col not in exclude_cols]
num_plots = len(cols_to_plot)
fig, axes = plt.subplots(num_plots, 1, figsize=(20, 4 * num_plots), sharex=True)
if num_plots == 1:
    axes = [axes]
date_formatter = mdates.DateFormatter("%Y-%m-%d")
for ax in axes:
    ax.xaxis.set_major_formatter(date_formatter)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.tick_params(labelbottom=True)
for ax, col in zip(axes, cols_to_plot):
    pct_series = final_df[col]
    # 处理可能的溢出问题
    pct_series = pct_series.clip(lower=-1e6, upper=1e6)
    log_return = np.log1p(pct_series / 100)
    cum_log_return = log_return.cumsum()
    level_series = 100 * np.exp(cum_log_return - cum_log_return.iloc[0])
    # 剪切以避免对数刻度的零值
    level_series = level_series.clip(lower=1e-5)
    # 在主y轴上绘制百分比系列（红色）
    ax.plot(final_df['Date'], pct_series, color='red', label=f'{col} %', alpha=0.7)
    ax.set_ylabel(f'{col} %', color='red')
    ax.tick_params(axis='y', labelcolor='red')
    # 为标准化的水平系列创建第二个y轴（蓝色）
    ax2 = ax.twinx()
    ax2.plot(final_df['Date'], level_series, color='blue', label=f'{col} Level')
    ax2.set_ylabel(f'{col} Level', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    # 如果水平值很大，则使用对数刻度
    if level_series.max() > 1e6:
        ax2.set_yscale('log')
plt.xlabel('Date')
plt.tight_layout()
plt.show()

# 对投资组合回报运行普通最小二乘回归
final_df["Excess Return"] = final_df["Portfolio Return"] - final_df["RF"]
X = final_df[["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"]]
X = sm.add_constant(X)  # 添加截距（阿尔法）
y = final_df["Excess Return"]
model = sm.OLS(y, X).fit()
print(model.summary())

# 跟踪因子敞口随时间的变化
window = 252
rolling_dates = []
rolling_alphas = []
rolling_residuals = []
rolling_betas = {"MKT_RF": [], "SMB": [], "HML": [], "RMW": [], "CMA": [], "MOM": []}
for i in range(window, len(final_df)):
    subset = final_df.iloc[i - window:i]  # 1年窗口
    y = subset["Portfolio Return"] - subset["RF"]
    X = subset[["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"]]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    rolling_dates.append(final_df.iloc[i]["Date"])
    rolling_alphas.append(model.params["const"])
    rolling_residuals.append(np.mean(np.abs(model.resid)))
    for factor in rolling_betas.keys():
        rolling_betas[factor].append(model.params[factor])
rolling_results = pd.DataFrame({
    "Date": rolling_dates,
    "Alpha": rolling_alphas,
    "Error": rolling_residuals
})
for factor in rolling_betas.keys():
    rolling_results[f"{factor} Beta"] = rolling_betas[factor]
rolling_results["Date"] = pd.to_datetime(rolling_results["Date"])
rolling_results.set_index("Date", inplace=True)
# 设置日期索引的频率
rolling_results = rolling_results.asfreq('D')
# 填充缺失值
rolling_results = rolling_results.fillna(method='ffill')

# 绘制滚动回归结果
final_df_date = final_df.copy()
final_df_date["Date"] = pd.to_datetime(final_df_date["Date"])
final_df_date.set_index("Date", inplace=True)
portfolio_pct_series = final_df_date["Portfolio Return"]
log_return = np.log1p(portfolio_pct_series / 100)
cum_log_return = log_return.cumsum()
portfolio_prices = 100 * np.exp(cum_log_return - cum_log_return.iloc[0])
portfolio_returns = final_df_date["Portfolio Return"]
portfolio_prices_rolling = portfolio_prices.reindex(rolling_results.index)
portfolio_returns_rolling = portfolio_returns.reindex(rolling_results.index)
fig, axes = plt.subplots(3, 1, figsize=(20, 12), sharex=True)
ax = axes[0]
for factor in rolling_betas.keys():
    ax.plot(rolling_results.index, rolling_results[f"{factor} Beta"], label=f"{factor} Beta")
ax.plot(rolling_results.index, rolling_results["Alpha"], color="darkgray", linewidth=0.8, label="Alpha")
ax.set_title("滚动因子贝塔值和阿尔法值（1年窗口）")
ax.axhline(0, color="darkgray", linewidth=0.8, linestyle="dotted")
ax.legend()
ax2 = axes[1]
ax3 = ax2.twinx()
ax2.plot(portfolio_prices_rolling.index, portfolio_prices_rolling, label="Portfolio Value")
ax3.plot(portfolio_returns_rolling.index, portfolio_returns_rolling, label="Portfolio Return")
undervalued = rolling_results["Alpha"] > 0
min_val = portfolio_prices_rolling.min()
max_val = portfolio_prices_rolling.max()
ax2.fill_between(rolling_results.index, min_val, max_val, where=undervalued, color="green", alpha=0.2, label="Undervalued")
ax2.fill_between(rolling_results.index, min_val, max_val, where=~undervalued, color="red", alpha=0.2, label="Overvalued")
ax2.set_title("投资组合价值与回报以及高估/低估区域")
ax2.legend(loc="upper left")
ax3.legend(loc="lower left")
ax4 = axes[2]
ax4.plot(rolling_results.index, rolling_results["Error"], color="darkred", linewidth=0.8)
ax4.set_title("滚动回归误差（平均绝对残差）")
ax4.axhline(rolling_results["Error"].mean(), color="darkgray", linestyle="dotted")
ax4.set_ylabel("平均绝对误差")
ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.tight_layout()
plt.show()

# 测试阿尔法
df_copy = rolling_results.copy()
df_copy = df_copy.dropna(subset=["Alpha"])
adf_result = adfuller(df_copy["Alpha"])
print(f"ADF Statistic: {adf_result[0]}")
print(f"p - value: {adf_result[1]}")
if adf_result[1] < 0.05:
    print("阿尔法值是均值回归（平稳的）")
else:
    print("阿尔法值遵循随机游走（不是均值回归）")
ar_model = AutoReg(df_copy["Alpha"], lags=1).fit()
rho = ar_model.params[1]
print(f"AR(1)系数 (rho): {rho}")
if rho < 1:
    print("阿尔法值是均值回归")
else:
    print("阿尔法值遵循随机游走")
half_life = np.log(0.5) / np.log(abs(rho))
print(f"阿尔法值的半衰期: {half_life} 个周期")
if half_life < 20:
    print("阿尔法值迅速回归（短期低效现象）")
else:
    print("阿尔法值需要更长的时间回归（持续的错误定价）")


def variance_ratio_test(series, lag=4):
    series = series.dropna()
    var_1 = np.var(series.diff(1).dropna(), ddof=1)
    var_k = np.var(series.diff(lag).dropna(), ddof=1)
    vr_stat = var_k / (var_1 * lag)
    return vr_stat


vr_stat = variance_ratio_test(df_copy["Alpha"], lag=4)
print(f"方差比: {vr_stat}")
if vr_stat < 1:
    print("阿尔法值是均值回归")
else:
    print("阿尔法值遵循随机游走")

# 跟踪阿尔法的持续性随时间的变化
df_copy = rolling_results.copy()
df_copy = df_copy.dropna(subset=["Alpha"])
rolling_window = 126
rolling_half_life = []
for i in range(len(df_copy) - rolling_window):
    # 确保使用 iloc 访问位置索引
    window_data = df_copy["Alpha"].iloc[i:i + rolling_window]
    try:
        ar_model = AutoReg(window_data, lags=1).fit()
        rho = ar_model.params[1]
        if abs(rho) < 1:
            half_life = np.log(0.5) / np.log(abs(rho))
        else:
            half_life = np.nan
    except:
        half_life = np.nan
    rolling_half_life.append(half_life)
df_copy = df_copy.iloc[rolling_window:].copy()
df_copy["Rolling_Half_Life"] = rolling_half_life
plt.figure(figsize=(12, 5))
plt.plot(df_copy.index, df_copy["Rolling_Half_Life"], color="red", label="Rolling Half - Life")
plt.axhline(y=145, color="darkgray", linestyle="dashed", label="Reference Half - Life")
plt.xlabel("Time")
plt.ylabel("Half - Life (Periods)")
plt.title("滚动投资组合阿尔法半衰期随时间的变化")
plt.legend()
plt.grid(True)
plt.show()
    