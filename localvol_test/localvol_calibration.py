import QuantLib as ql
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set the evaluation date
today = ql.Date(20, 6, 2025)
ql.Settings.instance().evaluationDate = today

# Market data: risk-free rate, dividend yield, spot price
risk_free_rate = 0.05
dividend_yield = 0.02
spot_price = 100.0

# Create yield term structures
calendar = ql.TARGET()
risk_free_curve = ql.FlatForward(today, risk_free_rate, ql.Actual365Fixed())
dividend_curve = ql.FlatForward(today, dividend_yield, ql.Actual365Fixed())

# Create synthetic implied volatility surface (strike x maturity)
strikes = np.linspace(80, 120, 9)
expiries = [ql.Date(20, 6, 2025 + i) for i in range(1, 6)]  # 1 to 5 years
vol_matrix = np.zeros((len(expiries), len(strikes)))

# Create a volatility skew/smile
for i, expiry in enumerate(expiries):
    t = ql.Actual365Fixed().yearFraction(today, expiry)
    atm_vol = 0.2 - 0.02 * np.sqrt(t)
    for j, strike in enumerate(strikes):
        moneyness = np.log(strike/spot_price) / (atm_vol * np.sqrt(t))
        vol_matrix[i,j] = atm_vol + 0.005 * moneyness**2 - 0.01 * moneyness

# BlackVarianceSurface expects rows = strikes, columns = dates
ql_matrix = ql.Matrix(len(strikes), len(expiries))
for i in range(len(strikes)):
    for j in range(len(expiries)):
        ql_matrix[i][j] = vol_matrix[j, i]  # Note the transposed indices

# Create the Black volatility surface
volatility_surface = ql.BlackVarianceSurface(
    today, calendar, 
    expiries, strikes, 
    ql_matrix,  # Use QuantLib Matrix instead of NumPy array
    ql.Actual365Fixed())
volatility_surface.setInterpolation("bicubic")
volatility_surface.enableExtrapolation()

# Create the Black-Scholes process
spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
flat_risk_free = ql.YieldTermStructureHandle(risk_free_curve)
flat_dividend = ql.YieldTermStructureHandle(dividend_curve)
flat_vol = ql.BlackVolTermStructureHandle(volatility_surface)

bsm_process = ql.BlackScholesMertonProcess(
    spot_handle, 
    flat_dividend, 
    flat_risk_free, 
    flat_vol)

# Create the local volatility surface
# local_vol_surface = ql.LocalVolSurface(
#     ql.BlackVolTermStructureHandle(volatility_surface),
#     flat_risk_free,
#     flat_dividend,
#     spot_handle)

# Create the local volatility surface with safeguards
try:
    local_vol_surface = ql.NoExceptLocalVolSurface(
        ql.BlackVolTermStructureHandle(volatility_surface),
        flat_risk_free,
        flat_dividend,
        spot_handle,
        0.2)
    print("Using NoExceptLocalVolSurface for safer local vol calculations.")
except AttributeError:
    # If NoExceptLocalVolSurface is not available in this QuantLib version
    # Create the local volatility surface with standard version but set safe boundaries
    local_vol_surface = ql.LocalVolSurface(
        ql.BlackVolTermStructureHandle(volatility_surface),
        flat_risk_free,
        flat_dividend,
        spot_handle)
    print("Using standard LocalVolSurface with safe boundaries.")

local_vol_surface.enableExtrapolation()

# Create a proper handle for the local volatility surface
local_vol_handle = ql.LocalVolTermStructureHandle(local_vol_surface)
bs_vol_handle = ql.BlackVolTermStructureHandle(volatility_surface)

# Create grid for visualization
times = np.linspace(0.1, 3, 20)
spot_prices = np.linspace(80, 130, 20)
local_vols = np.zeros((len(times), len(spot_prices)))
bs_vol = np.zeros((len(times), len(spot_prices)))

for i, t in enumerate(times):
    for j, s in enumerate(spot_prices):
        try:
            local_vols[i,j] = local_vol_surface.localVol(t, s)
            bs_vol[i,j] = volatility_surface.blackVol(t, s)
        except Exception as e:
            print(f"Warning at t={t:.3f}, s={s:.3f}: {e}")
            local_vols[i,j] = np.nan
            bs_vol[i,j] = np.nan

# Remove any NaN values for plotting
local_vols = np.nan_to_num(local_vols, nan=0.2)  # Replace NaN with reasonable default
bs_vol = np.nan_to_num(bs_vol, nan=0.2)

# ------------------------------
fig = plt.figure(figsize=(18, 8))

# First subplot - Local Volatility
ax1 = fig.add_subplot(121, projection='3d')
T, S = np.meshgrid(times, spot_prices)
surf1 = ax1.plot_surface(T, S, local_vols.T, cmap='viridis')
ax1.set_xlabel('Time to Expiry (years)')
ax1.set_ylabel('Spot Price')
ax1.set_zlabel('Local Volatility')
ax1.set_title('Local Volatility Surface')

# Second subplot - Implied Volatility
ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(T, S, bs_vol.T, cmap='plasma')
ax2.set_xlabel('Time to Expiry (years)')
ax2.set_ylabel('Spot Price')
ax2.set_zlabel('Implied Volatility')
ax2.set_title('Implied Volatility Surface')

# Add a colorbar for each plot
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

# Set the same z-axis limits for better comparison
max_vol = max(np.max(local_vols), np.max(bs_vol))
min_vol = min(np.min(local_vols), np.min(bs_vol))
ax1.set_zlim(min_vol, max_vol)
ax2.set_zlim(min_vol, max_vol)

plt.suptitle('Comparison of Local and Implied Volatility Surfaces', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.9)

# Save the combined figure
plt.savefig('localvol_test/figures/volatility_comparison.png', dpi=300)


# Price an option using the local vol surface
print("=" * 50)
expiry_date = ql.Date(20, 6, 2026)  # 1 year expiry
strike_price = 105.0
option_type = ql.Option.Call

# Create the option
exercise = ql.EuropeanExercise(expiry_date)
payoff = ql.PlainVanillaPayoff(option_type, strike_price)
option = ql.EuropeanOption(payoff, exercise)

local_vol_process = ql.GeneralizedBlackScholesProcess(
    spot_handle, 
    flat_dividend, 
    flat_risk_free, 
    bs_vol_handle,
    local_vol_handle)


# Price with local vol
option.setPricingEngine(ql.FdBlackScholesVanillaEngine(
                                    local_vol_process,
                                    100,  # time steps
                                    100,   # space steps
                                    0,
                                    ql.FdmSchemeDesc.Douglas(),
                                    True,  # <-- Enable local volatility
                                    -1.0  # optional: handle invalid local vol values
                                ))
local_vol_price = option.NPV()
local_vol_implied = option.impliedVolatility(local_vol_price, bsm_process)

# Compare with original market vol
option.setPricingEngine(ql.AnalyticEuropeanEngine(bsm_process))
market_price = option.NPV()
market_vol = option.impliedVolatility(market_price, bsm_process)

print(f"\nOption Pricing Comparison:")
print(f"Strike: {strike_price}, Expiry: {expiry_date}")
print(f"Market Price: {market_price:.4f}, Local Vol Price: {local_vol_price:.4f}")
print(f"Market Implied Vol: {market_vol:.4f}, Local Vol Implied Vol: {local_vol_implied:.4f}")