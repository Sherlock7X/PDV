#!/usr/bin/env python3
"""
Local Volatility Surface Calibration Example using QuantLib

This example demonstrates how to:
1. Generate synthetic market data (option prices)
2. Calibrate a local volatility surface using the Dupire model
3. Price options using the calibrated local volatility surface
4. Visualize the results
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
    """Generate synthetic market data using a known volatility surface"""
    
    # Define strikes and expiries for market data generation
    strikes = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120])
    expiries_days = np.array([30, 60, 90, 120, 180, 240, 300, 365, 400])  # Business days
    
    market_data = []
    
    # Create a synthetic volatility surface for data generation
    # Using a simple parameterized form: vol = base_vol + smile_factor * (log(K/S))^2
    base_vol = 0.20
    smile_factor = 0.5
    
    for expiry_days in expiries_days:
        expiry_date = today + ql.Period(int(expiry_days), ql.Days)
        time_to_expiry = expiry_days / 365.0
        
        for strike in strikes:
            # Generate synthetic implied volatility with smile
            log_moneyness = np.log(float(strike) / spot)
            implied_vol = base_vol + smile_factor * log_moneyness**2 + 1 * time_to_expiry * log_moneyness**2
            
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
                'expiry': expiry_date,
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
                vol_matrix[i][j] = 0.20
    
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

def price_option_with_local_vol(option, local_vol_surface, var_surface, spot, risk_free_curve, dividend_curve, today):
    """Price an option using the calibrated local vol surface using PDE method"""
    # Create Black-Scholes process with local volatility
    bs_process = ql.GeneralizedBlackScholesProcess(
    ql.QuoteHandle(ql.SimpleQuote(spot)),
    ql.YieldTermStructureHandle(dividend_curve),
    ql.YieldTermStructureHandle(risk_free_curve),
    ql.BlackVolTermStructureHandle(var_surface),
    ql.LocalVolTermStructureHandle(local_vol_surface)
)
    
    # Set the pricing engine
    option.setPricingEngine(ql.FdBlackScholesVanillaEngine(
                                    bs_process,
                                    100,  # time steps
                                    100,   # space steps
                                    0,
                                    ql.FdmSchemeDesc.Douglas(),
                                    True,  # <-- Enable local volatility
                                    -1.0  # optional: handle invalid local vol values
                                ))
 
    # Calculate NPV
    return option.NPV()

def visualize_volatility_surfaces(local_vol_surface, implied_vol_surface, spot, today):
    """Visualize both implied and local volatility surfaces"""
    
    # Create meshgrid for plotting
    strikes = np.linspace(85, 115, 20)
    times = np.linspace(0.1, 1.0, 20)
    
    Strike, Time = np.meshgrid(strikes, times)
    
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
                LocalVol[i, j] = 0.20
                ImpliedVol[i, j] = 0.20
    
    # Create subplots
    fig = plt.figure(figsize=(15, 6))
    
    # Plot implied volatility surface
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(Strike, Time, ImpliedVol, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('Strike')
    ax1.set_ylabel('Time to Expiry')
    ax1.set_zlabel('Implied Volatility')
    ax1.set_title('Implied Volatility Surface')
    
    # Plot local volatility surface
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(Strike, Time, LocalVol, cmap='plasma', alpha=0.8)
    ax2.set_xlabel('Strike')
    ax2.set_ylabel('Time to Expiry')
    ax2.set_zlabel('Local Volatility')
    ax2.set_title('Local Volatility Surface')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run the local volatility calibration example"""
    
    print("Local Volatility Surface Calibration Example")
    print("=" * 50)
    
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
    print("Strike\tExpiry\t\tTime\tPrice\t\tImpl Vol")
    print("-" * 60)
    for i, data in enumerate(market_data[:10]):  # Show first 10 points
        print(f"{data['strike']:.0f}\t{data['expiry']}\t{data['time_to_expiry']:.2f}\t{data['market_price']:.4f}\t\t{data['implied_vol']:.2%}")
    print()
    
    # Calibrate local volatility surface
    print("Calibrating local volatility surface...")
    local_vol_surface, implied_vol_surface = calibrate_local_volatility_surface(
        market_data, spot, risk_free_curve, dividend_curve, today
    )
    print("Calibration completed!")
    print()

    # Price a sample option using the local volatility surface
    strike = 100.0
    expiry = today + ql.Period(6, ql.Months)  # Use 6 months instead of 1 year to stay within our data range
    exercise = ql.EuropeanExercise(expiry)
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
    option = ql.VanillaOption(payoff, exercise)
    print(f"Pricing option with strike {strike} and expiry {expiry} using local volatility surface...")
    option_price = price_option_with_local_vol(option, local_vol_surface, implied_vol_surface, spot, risk_free_curve, dividend_curve, today)
    print(f"Option Price: {option_price:.4f}")
    print()
    
    # Display local volatility values at specific points
    print("Local Volatility Values at Specific Points:")
    print("Strike\tTime\tLocal Vol\tImpl Vol")
    print("-" * 40)
    
    test_times = [0.25, 0.5, 0.75] 
    test_strikes_lv = [90, 100, 110]
    
    for t in test_times:
        for k in test_strikes_lv:
            try:
                local_vol = local_vol_surface.localVol(t, k)
                impl_vol = implied_vol_surface.blackVol(t, k)
                print(f"{k:.0f}\t{t:.2f}\t{local_vol:.2%}\t\t{impl_vol:.2%}")
            except Exception as e:
                print(f"{k:.0f}\t{t:.2f}\tError: {str(e)}")
    
    print()
    print("Example completed successfully!")
    
    # Uncomment the following line to show volatility surface plots
    visualize_volatility_surfaces(local_vol_surface, implied_vol_surface, spot, today)

if __name__ == "__main__":
    main()