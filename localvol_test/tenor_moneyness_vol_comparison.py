import QuantLib as ql
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. 基本参数设置
calculation_date = ql.Date(19, 6, 2025)
ql.Settings.instance().evaluationDate = calculation_date

spot_price = 100.0
risk_free_rate = 0.03
dividend_rate = 0.01
day_count = ql.Actual365Fixed()
calendar = ql.NullCalendar()

# 2. 构建收益率曲线
riskfree_curve = ql.FlatForward(calculation_date, risk_free_rate, day_count)
dividend_curve = ql.FlatForward(calculation_date, dividend_rate, day_count)

# 3. 创建自定义波动率微笑曲面 (moneyness-based)
tenors = [ql.Period(i, ql.Years) for i in range(1, 4)]  # 1年, 2年, 3年
strikes = [spot_price * m for m in np.arange(0.6, 1.21, 0.1)]  # moneyness 0.6-1.2

# 波动率矩阵 (行: 期限, 列: 行权价)
vol_matrix = np.array([
    # 自定义微笑曲线: 抛物线形状 (0.6->0.45, ATM->0.3, 1.2->0.4)
    [0.45, 0.38, 0.32, 0.30, 0.32, 0.38, 0.40],  # 1年
    [0.43, 0.36, 0.31, 0.29, 0.31, 0.36, 0.39],  # 2年
    [0.41, 0.35, 0.30, 0.28, 0.30, 0.35, 0.38]   # 3年
])

# 4. 创建隐含波动率曲面
vol_dates = [calculation_date + tenor for tenor in tenors]
vol_surface = ql.BlackVarianceSurface(
    calculation_date, calendar, 
    vol_dates, strikes, vol_matrix, day_count
)
vol_surface.setInterpolation("bicubic")
vol_handle = ql.BlackVolTermStructureHandle(vol_surface)

# 5. 构建局部波动率曲面
spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
local_vol_surface = ql.LocalVolSurface(
    vol_handle, riskfree_curve, dividend_curve, spot_handle
)

# 6. 在网格点上计算并比较波动率
moneyness_grid = np.arange(0.6, 1.21, 0.05)  # moneyness网格
tenor_grid = np.linspace(0.1, 3.0, 15)       # 期限网格 (0.1-3年)

# 存储结果
results = []
for m in moneyness_grid:
    for t in tenor_grid:
        strike = spot_price * m
        expiry_date = calculation_date + ql.Period(int(t*365), ql.Days)
        t_years = day_count.yearFraction(calculation_date, expiry_date)
        
        # 跳过超出曲面范围的点
        if not (vol_surface.minTime() <= t_years <= vol_surface.maxTime() and
                vol_surface.minStrike() <= strike <= vol_surface.maxStrike()):
            continue
        
        # 计算隐含波动率
        impl_vol = vol_surface.blackVol(t_years, strike)
        
        # 计算局部波动率
        try:
            local_vol = local_vol_surface.localVol(t_years, strike)
        except:
            local_vol = float('nan')
        
        results.append({
            'moneyness': m,
            'tenor': t,
            'implied_vol': impl_vol,
            'local_vol': local_vol
        })

# 7. 可视化结果
fig = plt.figure(figsize=(14, 10))

# 3D曲面图 - 隐含波动率
ax1 = fig.add_subplot(221, projection='3d')
x = [r['moneyness'] for r in results]
y = [r['tenor'] for r in results]
z1 = [r['implied_vol'] for r in results]
ax1.plot_trisurf(x, y, z1, cmap='viridis', edgecolor='none')
ax1.set_title('Implied Volatility Surface')
ax1.set_xlabel('Moneyness')
ax1.set_ylabel('Tenor (years)')
ax1.set_zlabel('Volatility')

# 3D曲面图 - 局部波动率
ax2 = fig.add_subplot(222, projection='3d')
z2 = [r['local_vol'] for r in results]
ax2.plot_trisurf(x, y, z2, cmap='plasma', edgecolor='none')
ax2.set_title('Local Volatility Surface')
ax2.set_xlabel('Moneyness')
ax2.set_ylabel('Tenor (years)')
ax2.set_zlabel('Volatility')

# 波动率差值图
ax3 = fig.add_subplot(223)
vol_diff = [lv - iv for lv, iv in zip(z2, z1)]
sc = ax3.scatter(x, y, c=vol_diff, cmap='coolwarm', 
                 vmin=-0.1, vmax=0.1, s=50)
ax3.set_title('Local Vol - Implied Vol')
ax3.set_xlabel('Moneyness')
ax3.set_ylabel('Tenor (years)')
plt.colorbar(sc, ax=ax3, label='Volatility Difference')

# ATM切片比较 (moneyness=1.0)
ax4 = fig.add_subplot(224)
atm_results = [r for r in results if 0.99 < r['moneyness'] < 1.01]
tenors_atm = sorted(set([r['tenor'] for r in atm_results]))
impl_atm = [next(r['implied_vol'] for r in atm_results if r['tenor'] == t) 
            for t in tenors_atm]
local_atm = [next(r['local_vol'] for r in atm_results if r['tenor'] == t) 
             for t in tenors_atm]

ax4.plot(tenors_atm, impl_atm, 'bo-', label='Implied Vol')
ax4.plot(tenors_atm, local_atm, 'r^-', label='Local Vol')
ax4.set_title('ATM Volatility Comparison (Moneyness=1.0)')
ax4.set_xlabel('Tenor (years)')
ax4.set_ylabel('Volatility')
ax4.legend()
ax4.grid(True)

plt.tight_layout()
# plt.savefig('vol_comparison.png', dpi=300)
plt.show()

# 8. 输出关键点比较
print("\nKey Point Comparison:")
print("Moneyness | Tenor | Implied Vol | Local Vol | Difference")
print("-" * 55)
for m in [0.6, 0.8, 1.0, 1.2]:
    for t in [0.5, 1.0, 2.0, 3.0]:
        strike = spot_price * m
        expiry_date = calculation_date + ql.Period(int(t*365), ql.Days)
        t_years = day_count.yearFraction(calculation_date, expiry_date)
        
        impl_vol = vol_surface.blackVol(t_years, strike)
        try:
            local_vol = local_vol_surface.localVol(t_years, strike)
        except:
            local_vol = float('nan')
            
        diff = local_vol - impl_vol
        print(f"{m:8.2f} | {t:5.1f} | {impl_vol:10.4f} | {local_vol:9.4f} | {diff:9.4f}")

# # 9. 保存结果到CSV
# import csv
# with open('volatility_comparison.csv', 'w', newline='') as csvfile:
#     fieldnames = ['moneyness', 'tenor', 'implied_vol', 'local_vol']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writeheader()
#     writer.writerows(results)

# print("\nResults saved to volatility_comparison.csv and vol_comparison.png")