"""
 Fama-French 五因子数据
"""
import requests
import zipfile
import io
import re
import pandas as pd
# import yfinance as yf

class FamaFrenchDownloader:
    """下载并处理Fama-French五因子数据（每日）。"""

    @staticmethod
    def download_ff5():
        """从网络获取并提取F-F五因子数据。"""
        response = requests.get(FamaFrenchDownloader.FF5_URL)
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            file_name = z.namelist()[0]
            with z.open(file_name) as f:
                return f.read().decode("utf-8").splitlines()

    @staticmethod
    def parse_ff5_data():
        """提取并清理F-F五因子数据。"""
        lines = FamaFrenchDownloader.download_ff5()
        # 仅保留数据行（以8位日期开头）
        data_lines = [line for line in lines if re.match(r'^\s*\d{8}', line)]
        # 读取到DataFrame
        df = pd.read_csv(io.StringIO("\n".join(data_lines)), sep=r"\s*,\s*", header=None)
        # 分配列名
        df.columns = ["Date", "MKT_RF", "SMB", "HML", "RMW", "CMA", "RF"]
        # 转换日期列
        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d", errors="coerce")
        # 转换数值列
        df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
        return df

# 调用函数
ff5_df = FamaFrenchDownloader.parse_ff5_data()
ff5_df

"""
动量因子
"""
import requests
import zipfile
import io
import re
import pandas as pd

class MomentumDownloader:
    """下载并处理Fama-French动量因子数据（每日）。"""

    MOM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_"

    @staticmethod
    def download_momentum():
        """从网络获取并提取动量因子数据。"""
        response = requests.get(MomentumDownloader.MOM_URL)
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            file_name = z.namelist()[0]
            with z.open(file_name) as f:
                return f.read().decode("utf-8").splitlines()

    @staticmethod
    def parse_momentum_data():
        """提取并清理动量因子数据（每日）。"""
        lines = MomentumDownloader.download_momentum()

        # 仅保留数据行（以8位日期YYYYMMDD开头）
        data_lines = [line for line in lines if re.match(r'^\s*\d{8}', line)]
        # 读取到DataFrame
        df = pd.read_csv(io.StringIO("\n".join(data_lines)), sep=r"\s*,\s*", header=None)

        # 分配列名
        df.columns = ["Date", "MOM"]

        # 转换日期列
        df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d", errors="coerce")
        # 将MoM转换为数值
        df["MOM"] = pd.to_numeric(df["mom"], errors="coerce")

        # 删除无效行
        df.dropna(subset=["Date", "MOM"], inplace=True)

        return df

# 调用函数
mom_df = MomentumDownloader.parse_momentum_data()
mom_df

"""
下载股票价格数据
"""
import yfinance as yf
import pandas as pd

class StockDataFetcher:
    """从雅虎财经下载并处理股票数据。"""

    @staticmethod
    def get_stock_data(tickers, weights, start_date="1990-01-01"):
        # 获取多个股票代码的数据并计算加权每日回报

        参数:
        tickers (list): 股票代码列表(例如,["AAPL", "MSFT"])
        weights (list): 权重列表(例如,[0.3, 0.7]).必须与tickers长度相同
        start_date (str): 历史数据的开始日期.

        返回:
        DataFrame: 包含日期\股票价格和投资组合每日回报.

        if len(tickers) != len(weights):
            raise ValueError("股票代码和权重必须具有相同的长度")

        # 使用默认的auto_adjust = True获取数据。
        raw_df = yf.download(tickers, start=start_date, progress=False, group_by="column")

        # 处理多股票代码数据。
        if isinstance(raw_df.columns, pd.MultiIndex):
            # 如果可能，使用'Adj Close'，否则使用'Close'
            level_vals = raw_df.columns.get_level_values(1)
            price_col = "Adj Close"if"Adj Close"in level_vals else"Close"
            price_cols = [col for col in raw_df.columns if col[1] == price_col]
            raw_df = raw_df[price_cols]
            raw_df.columns = [col[0] for col in raw_df.columns]
        else:
            # 对于单个股票代码下载。
            price_col = "Adj Close"if"Adj Close"in raw_df.columns else"close"
            raw_df = raw_df[[price_col]]
            raw_df.columns = tickers

        # 确保DataFrame列遵循股票代码的顺序。
        raw_df = raw_df[tickers]

        # 计算每日回报。
        returns_df = raw_df.pct_change() * 100
        # 计算加权投资组合回报。
        portfolio_return = returns_df.dot(weights)

        # 合并数据。
        final_df = raw_df.copy()
        final_df["Portfolio Return"] = portfolio_return
        final_df.reset_index(inplace=True)

        return final_df

# 获取股票/投资组合数据
tickers = ["AAPL", "MSFT"]
weights = [0.3, 0.7]  # 这是股票在投资组合中的权重
stock_df = StockDataFetcher.get_stock_data(tickers, weights)
stock_df
"""
将股票回报与因子数据合并
"""
# 合并Fama-French五因子和动量因子数据
merged_factors_df = pd.merge(ff5_df, mom_df, on="Date", how="outer")

# 与股票回报合并（使用通用的股票DataFrame，例如stock_df）
final_df = pd.merge(stock_df, merged_factors_df, on="Date", how="inner")
final_df.dropna(inplace=True)
# 显示最终合并的数据集
final_df
"""
可视化因子数据和回报
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.style.use('dark_background')

# 将Date转换为datetime
final_df['Date'] = pd.to_datetime(final_df['Date'])

# 假设股票代码在变量^tickers中定义
try:
    stock_ticker_cols = tickers
except NameError:
    stock_ticker_cols = []

# 排除股票价格列和任何每日回报列
exclude_cols = ['Date', 'Adj Close'] + stock_ticker_cols
exclude_cols += [col for col in final_df.columns if'Daily Return'in col]
cols_to_plot = [col for col in final_df.columns if col notin exclude_cols]
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

"""
对投资组合回报运行普通最小二乘回归
"""
import statsmodels.api as sm

# 计算投资组合超额回报
final_df["Excess Return"] = final_df["Portfolio Return"] - final_df["RF"]
# 选择因子回报作为自变量
X = final_df[["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"]
X = sm.add_constant(X)  # 添加截距（阿尔法）

# 设置因变量（超额回报）
y = final_df["Excess Return"]

# 运行OLS回归
model = sm.OLS(y, X).fit()

# 打印回归摘要
print(model.summary())

"""
运行滚动回归
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 设置滚动窗口（252个交易日，大约1年）
window = 252

# 存储滚动估计
rolling_dates = []
rolling_alphas = []
rolling_residuals = []
rolling_betas = {"MKT_RF": [], "SMB": [], "HML": [], "RMW": [], "CMA": [], "MOM": []}

# 执行滚动回归
for i in range(window, len(final_df)):
    subset = final_df.iloc[i - window: i]  # 1年窗口
    # 计算投资组合超额回报
    y = subset["Portfolio Return"] - subset["RF"]
    X = subset[["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"]
    X = sm.add_constant(X)
    # 运行回归
    model = sm.OLS(y, X).fit()
    # 存储结果
    rolling_dates.append(final_df.iloc[i]["Date"])
    rolling_alphas.append(model.params["const"])
    rolling_residuals.append(np.mean(np.abs(model.resid)))
    for factor in rolling_betas.keys():
        rolling_betas[factor].append(model.params[factor])

# 将滚动结果转换为DataFrame
rolling_results = pd.DataFrame({
    "Date": rolling_dates,
    "Alpha": rolling_alphas,
    "Error": rolling_residuals
})
for factor in rolling_betas.keys():
    rolling_results[f"{factor} Beta"] = rolling_betas[factor]
rolling_results["Date"] = pd.to_datetime(rolling_results["Date"])
rolling_results.set_index("Date", inplace=True)
rolling_results

"""
运行滚动回归
"""
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 确保final_df的Date是datetime索引。
final_df_date = final_df.copy()
final_df_date["Date"] = pd.to_datetime(final_df_date["Date"])
final_df_date.set_index("Date", inplace=True)

# 从投资组合回报计算投资组合价值系列。
portfolio_pct_series = final_df_date["Portfolio Return"]
log_return = np.log1p(portfolio_pct_series / 100)
cum_log_return = log_return.cumsum()
portfolio_prices = 100 * np.exp(cum_log_return - cum_log_return.iloc[0])
portfolio_returns = final_df_date["Portfolio Return"]

# 将投资组合系列重新索引以匹配滚动回归结果。
portfolio_prices_rolling = portfolio_prices.reindex(rolling_results.index)
portfolio_returns_rolling = portfolio_returns.reindex(rolling_results.index)

# 创建3个子图，共享x轴。
fig, axes = plt.subplots(3, 1, figsize=(20,12), sharex=True)

# 绘制滚动因子贝塔值和阿尔法值。
ax = axes[0]

for factor in rolling_betas.keys():
    ax.plot(rolling_results.index, rolling_results[f"{factor} Beta"], label=f"{factor} Beta")
ax.plot(rolling_results.index, rolling_results["Alpha"], color="darkgray", linewidth=0.8, label="Alpha")
ax.set_title("滚动因子贝塔值和阿尔法值（1年窗口）")
ax.axhline(0, color="darkgray", linewidth=0.8, linestyle="dotted")
ax.legend()

# 在单独的轴上绘制投资组合价值和每日回报。
ax2 = axes[1]
ax3 = ax2.twinx()  # 次级y轴。
ax2.plot(portfolio_prices_rolling.index, portfolio_prices_rolling, label="Portfolio Value")
ax3.plot(portfolio_returns_rolling.index, portfolio_returns_rolling, label="Portfolio Return")

# 根据滚动回归中的阿尔法值填充区域。
undervalued = rolling_results["Alpha"] > 0
min_val = portfolio_prices_rolling.min()
max_val = portfolio_prices_rolling.max()
ax2.fill_between(rolling_results.index, min_val, max_val, where=undervalued, color="green", alpha=0.2, label="Undervalued")
ax2.fill_between(rolling_results.index, min_val, max_val, where=~undervalued, color="red", alpha=0.2, label="Overvalued")
ax2.set_title("投资组合价值与回报以及高估/低估区域")
ax2.legend(loc="upper left")
ax3.legend(loc="lower left")

# 绘制滚动回归误差（平均绝对残差）。
ax4 = axes[2]
ax4.plot(rolling_results.index, rolling_results["Error"], color="darkred", linewidth=0.8)
ax4.set_title("滚动回归误差（平均绝对残差）")
ax4.axhline(rolling_results["Error"].mean(), color="darkgray", linestyle="dotted")
ax4.set_ylabel("平均绝对误差")

# 设置x轴刻度每3个月一次，并格式化日期。
ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.tight_layout()
plt.show()

"""
测试阿尔法
5.1 均值回归
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg

# 创建滚动结果DataFrame的副本
df_copy = rolling_results.copy()

# 删除阿尔法值的NaN值以进行分析
df_copy = df_copy.dropna(subset=["Alpha"])

# 1. 增广迪基-福勒（ADF）测试
adf_result = adfuller(df_copy["Alpha"])
print(f"ADF Statistic: {adf_result[0]}")
print(f"p-value: {adf_result[1]}")
if adf_result[1] < 0.05:
    print("阿尔法值是均值回归（平稳的）")
else:
    print("阿尔法值遵循随机游走（不是均值回归）")

# 2. 自回归模型（AR-1）

ar_model = AutoReg(df_copy["Alpha"], lags=1).fit()
rho = ar_model.params[1]  # AR(1)系数

print(f"AR(1)系数 (rho): {rho}")
if rho < 1:
    print("阿尔法值是均值回归")
else:
    print("阿尔法值遵循随机游走")

# 3. 半衰期计算
half_life = np.log(0.5) / np.log(abs(rho))

print(f"阿尔法值的半衰期: {half_life} 个周期")
if half_life < 20:
    print("阿尔法值迅速回归（短期低效现象）")
else:
    print("阿尔法值需要更长的时间回归（持续的错误定价）")

# 4. 自定义方差比测试
def variance_ratio_test(series, lag=4):
    series = series.dropna()
    # 计算方差
    var_1 = np.var(series.diff(1).dropna(), ddof=1)
    var_k = np.var(series.diff(lag).dropna(), ddof=1)
    # 方差比
    vr_stat = var_k / (var_1 * lag)
    return vr_stat

vr_stat = variance_ratio_test(df_copy["Alpha"], lag=4)
print(f"方差比: {vr_stat}")
if vr_stat < 1:
    print("阿尔法值是均值回归")
else:
    print("阿尔法值遵循随机游走")

"""
5.2 跟踪阿尔法的持续性随时间的变化
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg

# 创建rolling_results DataFrame的副本
df_copy = rolling_results.copy()

# 删除阿尔法值的NaN值
df_copy = df_copy.dropna(subset=["Alpha"])

# 定义滚动窗口大小（例如，126个交易日）
rolling_window = 126

# 计算滚动AR(1)系数（rho）和半衰期
rolling_half_life = []

for i in range(len(df_copy) - rolling_window):
    window_data = df_copy["Alpha"].iloc[i:i+rolling_window]
    # 拟合AR(1)模型
    try:
        ar_model = AutoReg(window_data, lags=1).fit()
        rho = ar_model.params[1]  # AR(1)系数
        # 计算半衰期
        if abs(rho) < 1:
            half_life = np.log(0.5) / np.log(abs(rho))
        else:
            half_life = np.nan
    except:
        half_life = np.nan
    rolling_half_life.append(half_life)

# 将结果与原始索引对齐
df_copy = df_copy.iloc[rolling_window:].copy()
df_copy["Rolling_Half_Life"] = rolling_half_life

# 绘制滚动半衰期
plt.figure(figsize=(12,5))
plt.plot(df_copy.index, df_copy["Rolling_Half_Life"], color="red", label="Rolling Half-Life")
plt.axhline(y=145, color="darkgray", linestyle="dashed", label="Reference Half-Life")
plt.xlabel("Time")
plt.ylabel("Half-Life (Periods)")
plt.title("滚动投资组合阿尔法半衰期随时间的变化")
plt.legend()
plt.grid(True)
plt.show()

"""
投资组合回报的回报归因
"""
# "本节将投资组合回报分解为因子驱动的和无法解释的部分。它回答了两个关键问题：
#
# 投资组合的表现有多少是由已知因子解释的？
# 实际投资组合回报与因子敞口的预期回报相比如何？
# 代码通过将滚动贝塔值与因子回报相乘来计算因子贡献。然后，它将无法解释的回报（残差阿尔法），即无法由已知风险因子解释的部分分开。
#
# 三种可视化结果总结了结果：
#
# 复合因子贡献：显示不同因子如何驱动长期回报。
# 算术因子贡献：跟踪每个因子的原始累积影响。
# 实际与预期回报：比较投资组合表现与其基于因子的归因。"