#!/usr/bin/env python3
"""
Local Volatility Surface Calibration Example - Modified Version
- Uses tenors of 1-3 years
- Moneyness range of 0.6-1.2
- Volatility smile pattern [0.4, 0.35, 0.33, 0.3, 0.33, 0.35, 0.4]
- Compares implied volatility with local volatility
"""

import QuantLib as ql
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def setup_market_environment():
    """Setup the basic market environment"""
    # Market data
    spot = 100.0
    risk_free_rate = 0.05
    dividend_yield = 0.02
    
    # Set evaluation date
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today
    
    # Market curves
    risk_free_curve = ql.FlatForward(today, risk_free_rate, ql.Actual365Fixed())
    dividend_curve = ql.FlatForward(today, dividend_yield, ql.Actual365Fixed())
    
    return spot, risk_free_curve, dividend_curve, today

def generate_market_data(spot, risk_free_curve, dividend_curve, today):
    """Generate synthetic market data using specified volatility pattern"""
    
    # Define moneyness levels and expiries for market data generation
    moneyness_levels = np.linspace(0.6, 1.2, 50)
    # Convert moneyness to strikes
    strikes = moneyness_levels * spot
    
    # Define tenors in years (1 to 3 years)
    tenors_years = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    
    # Volatility smile pattern for each moneyness level (0.3 - 0.4 - 0.3)
    vol_pattern = np.linspace(0.4, 0.3, len(strikes) // 2).tolist() + \
                   np.linspace(0.3, 0.4, len(strikes) // 2).tolist()
    
    market_data = []
    
    for tenor_year in tenors_years:
        expiry_days = int(tenor_year * 365)
        expiry_date = today + ql.Period(expiry_days, ql.Days)
        time_to_expiry = expiry_days / 365.0
        
        for i, strike in enumerate(strikes):
            # Get volatility from our pattern
            implied_vol = vol_pattern[i]
            
            # Create option and calculate theoretical price
            exercise = ql.EuropeanExercise(expiry_date)
            payoff = ql.PlainVanillaPayoff(ql.Option.Call, float(strike))
            option = ql.VanillaOption(payoff, exercise)
            
            # Use Black-Scholes to get theoretical price
            bs_process = ql.BlackScholesMertonProcess(
                ql.QuoteHandle(ql.SimpleQuote(spot)),
                ql.YieldTermStructureHandle(dividend_curve),
                ql.YieldTermStructureHandle(risk_free_curve),
                ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.TARGET(), implied_vol, ql.Actual365Fixed()))
            )
            
            engine = ql.AnalyticEuropeanEngine(bs_process)
            option.setPricingEngine(engine)
            
            market_price = option.NPV()
            
            market_data.append({
                'strike': float(strike),
                'moneyness': float(moneyness_levels[i]),
                'expiry': expiry_date,
                'tenor_years': tenor_year,
                'time_to_expiry': time_to_expiry,
                'market_price': market_price,
                'implied_vol': implied_vol
            })
    
    return market_data

def calibrate_local_volatility_surface(market_data, spot, risk_free_curve, dividend_curve, today):
    """Calibrate local volatility surface using market data"""
    
    # Create implied volatility surface from market data
    strikes = sorted(list(set([data['strike'] for data in market_data])))
    expiry_dates = sorted(list(set([data['expiry'] for data in market_data])))
    
    print(f"Number of strikes: {len(strikes)}")
    print(f"Number of expiry dates: {len(expiry_dates)}")
    
    # Create matrix for implied volatilities (rows = strikes, cols = expiry_dates)
    vol_matrix = ql.Matrix(len(strikes), len(expiry_dates))
    
    # Fill the volatility matrix
    for i, strike in enumerate(strikes):
        for j, expiry in enumerate(expiry_dates):
            # Find corresponding market data point
            vol = None
            for data in market_data:
                if data['expiry'] == expiry and data['strike'] == strike:
                    vol = data['implied_vol']
                    break
            
            if vol is not None:
                vol_matrix[i][j] = vol
            else:
                # Interpolate or use a default value
                vol_matrix[i][j] = 0.30
    
    # Create BlackVarianceSurface
    var_surface = ql.BlackVarianceSurface(
        today,
        ql.TARGET(),
        expiry_dates,
        strikes,
        vol_matrix,
        ql.Actual365Fixed()
    )
    
    # Enable extrapolation for the surface
    var_surface.enableExtrapolation()
    
    # Create local volatility surface using Dupire formula
    local_vol_surface = ql.LocalVolSurface(
        ql.BlackVolTermStructureHandle(var_surface),
        ql.YieldTermStructureHandle(risk_free_curve),
        ql.YieldTermStructureHandle(dividend_curve),
        ql.QuoteHandle(ql.SimpleQuote(spot))
    )
    
    return local_vol_surface, var_surface

def visualize_volatility_surfaces(local_vol_surface, implied_vol_surface, spot, today):
    """Visualize both implied and local volatility surfaces"""
    
    # Create meshgrid for plotting
    moneyness_levels = np.linspace(0.7, 1.15, 40)
    strikes = moneyness_levels * spot
    times = np.linspace(1.1, 2.5, 30)
    
    Strike, Time = np.meshgrid(strikes, times)
    Moneyness, _ = np.meshgrid(moneyness_levels, times)
    
    # Calculate local volatilities
    LocalVol = np.zeros_like(Strike)
    ImpliedVol = np.zeros_like(Strike)
    
    for i, t in enumerate(times):
        expiry_date = today + ql.Period(int(t * 365), ql.Days)
        for j, k in enumerate(strikes):
            try:
                LocalVol[i, j] = local_vol_surface.localVol(t, k)
                ImpliedVol[i, j] = implied_vol_surface.blackVol(t, k)
            except:
                LocalVol[i, j] = 0.30
                ImpliedVol[i, j] = 0.30
    
    # Create subplots
    fig = plt.figure(figsize=(10, 10))
    
    # Plot implied volatility surface
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(Moneyness, Time, ImpliedVol, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('Moneyness (K/S)')
    ax1.set_ylabel('Time to Expiry (Years)')
    ax1.set_zlabel('Implied Volatility')
    ax1.set_title('Implied Volatility Surface')
    
    # Plot local volatility surface
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(Moneyness, Time, LocalVol, cmap='plasma', alpha=0.8)
    ax2.set_xlabel('Moneyness (K/S)')
    ax2.set_ylabel('Time to Expiry (Years)')
    ax2.set_zlabel('Local Volatility')
    ax2.set_title('Local Volatility Surface')
    
    plt.tight_layout()
    plt.show()



def main():
    """Main function to run the local volatility calibration example"""
    
    print("Local Volatility Surface Calibration - Moneyness & Tenor Based")
    print("=" * 70)
    
    # Setup market environment
    spot, risk_free_curve, dividend_curve, today = setup_market_environment()
    print(f"Spot price: {spot}")
    print(f"Risk-free rate: {risk_free_curve.zeroRate(1.0, ql.Continuous).rate():.2%}")
    print(f"Dividend yield: {dividend_curve.zeroRate(1.0, ql.Continuous).rate():.2%}")
    print()
    
    # Generate synthetic market data
    print("Generating synthetic market data...")
    market_data = generate_market_data(spot, risk_free_curve, dividend_curve, today)
    print(f"Generated {len(market_data)} market data points")
    print()
    
    # Display sample market data
    print("Sample Market Data:")
    print("Strike\tMoneyness\tTenor\tExpiry\t\tPrice\t\tImpl Vol")
    print("-" * 75)
    for i, data in enumerate(sorted(market_data[:15], key=lambda x: (x['tenor_years'], x['moneyness']))):
        print(f"{data['strike']:.1f}\t{data['moneyness']:.1f}\t\t{data['tenor_years']:.1f}y\t{data['expiry']}\t{data['market_price']:.4f}\t{data['implied_vol']:.2%}")
    print()
    
    # Calibrate local volatility surface
    print("Calibrating local volatility surface...")
    local_vol_surface, implied_vol_surface = calibrate_local_volatility_surface(
        market_data, spot, risk_free_curve, dividend_curve, today
    )
    print("Calibration completed!")
    print()

    visualize_volatility_surfaces(local_vol_surface, implied_vol_surface, spot, today)
    

if __name__ == "__main__":
    main()
