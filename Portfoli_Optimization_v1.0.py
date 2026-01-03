import sys
import subprocess
import os
import importlib.util

# ==========================================
# 0. AUTO-INSTALLER & LAUNCHER
# ==========================================
def check_and_install_dependencies():
    """Checks for required libraries and installs them if missing."""
    required_libraries = ['streamlit', 'plotly', 'yfinance', 'pandas', 'numpy', 'scipy', 'matplotlib']
    missing = []

    print("Checking dependencies...")
    for lib in required_libraries:
        if importlib.util.find_spec(lib) is None:
            missing.append(lib)
    
    if missing:
        print(f"Missing libraries found: {missing}")
        print("Installing them automatically... (This may take a minute)")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
            print("Installation complete!")
        except subprocess.CalledProcessError:
            print("Failed to install libraries. Please run: pip install " + " ".join(required_libraries))
            sys.exit(1)

# Run the check immediately
if __name__ == "__main__":
    check_and_install_dependencies()

    # Auto-launch Streamlit if run as a standard Python script
    # (Check if the script is being run by 'streamlit run' or just 'python')
    if "streamlit" not in sys.modules and os.environ.get("STREAMLIT_RUN") is None:
        print(" Launching Dashboard...")
        # Set a flag to prevent infinite loops
        os.environ["STREAMLIT_RUN"] = "true"
        # Run the streamlit command on this very file
        subprocess.run(["streamlit", "run", sys.argv[0]])
        sys.exit()

# ==========================================
# 1. IMPORTS (Safe to run now)
# ==========================================
try:
    import streamlit as st
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import scipy.optimize as sco
    import plotly.express as px
    import plotly.graph_objects as go
    from datetime import datetime
except ImportError:
    # This fallback catches any edge cases where install appeared to work but import failed
    st = None 

# ==========================================
# 2. THE LOGIC PIPELINE
# ==========================================
class PortfolioAnalyzer:
    def __init__(self, tickers, benchmark, start_date, end_date, rf_rate):
        self.tickers = tickers
        self.benchmark = benchmark
        self.start_date = start_date
        self.end_date = end_date
        self.rf = rf_rate
        self.data = None
        self.returns = None

    def fetch_data(self):
        all_tickers = self.tickers + [self.benchmark]
        try:
            raw_data = yf.download(all_tickers, start=self.start_date, end=self.end_date, interval="1mo", progress=False, auto_adjust=False)
            
            if raw_data.empty: return False

            # Robust column extraction
            try:
                if 'Adj Close' in raw_data.columns: price_data = raw_data['Adj Close']
                elif 'Close' in raw_data.columns: price_data = raw_data['Close']
                else: price_data = raw_data.xs('Adj Close', level=0, axis=1)
            except KeyError: return False

            price_data = price_data.dropna()
            valid_tickers = [t for t in self.tickers if t in price_data.columns]
            
            if not valid_tickers: return False
            
            self.data = price_data[valid_tickers]
            self.bench_data = price_data[self.benchmark] if self.benchmark in price_data.columns else None
            
            if self.bench_data is None: return False

            self.returns = self.data.pct_change().dropna()
            self.bench_returns = self.bench_data.pct_change().dropna()
            
            # Align
            common = self.returns.index.intersection(self.bench_returns.index)
            self.returns = self.returns.loc[common]
            self.bench_returns = self.bench_returns.loc[common]
            
            return not self.returns.empty
        except: return False

    def get_metrics(self):
        metrics = pd.DataFrame(index=self.data.columns)
        metrics['CAGR'] = (1 + self.returns).prod()**(12/len(self.returns)) - 1
        metrics['Volatility'] = self.returns.std() * np.sqrt(12)
        metrics['Sharpe'] = (metrics['CAGR'] - self.rf) / metrics['Volatility']
        metrics['Beta'] = self.returns.apply(lambda x: x.cov(self.bench_returns)) / self.bench_returns.var()
        
        cum_ret = (1 + self.returns).cumprod()
        metrics['Max Drawdown'] = ((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min()
        return metrics

    def optimize(self):
        mean_ret = self.returns.mean()
        cov_mat = self.returns.cov()
        
        def neg_sharpe(weights):
            p_ret = np.sum(mean_ret * weights) * 12
            p_var = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights))) * np.sqrt(12)
            return -(p_ret - self.rf) / p_var

        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bnds = tuple((0.0, 1.0) for _ in range(len(self.tickers)))
        res = sco.minimize(neg_sharpe, len(self.tickers)*[1./len(self.tickers)], method='SLSQP', bounds=bnds, constraints=cons)
        return res.x

    def simulate(self):
        results = np.zeros((3, 2000))
        mean_ret = self.returns.mean() * 12
        cov_mat = self.returns.cov() * 12
        for i in range(2000):
            w = np.random.random(len(self.tickers)); w /= np.sum(w)
            p_std = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
            p_ret = np.sum(mean_ret * w)
            results[:,i] = [p_std, p_ret, (p_ret - self.rf)/p_std]
        return results

# ==========================================
# 3. DASHBOARD UI
# ==========================================
if st: # Only run if streamlit loaded successfully
    st.set_page_config(page_title="Auto-Portfolio", layout="wide")

    with st.sidebar:
        st.header("Setup")
        tickers = st.text_area("Tickers", "AAPL MSFT GLD TLT SPY").upper().split()
        bench = st.text_input("Benchmark", "SPY").upper()
        start = st.date_input("Start", datetime(2015,1,1))
        rf = st.number_input("Risk Free %", 4.5) / 100
        run = st.button("Run Analysis")

    st.title("Quantitative Portfolio Dashboard")

    if run:
        with st.spinner("Processing..."):
            eng = PortfolioAnalyzer(tickers, bench, start, datetime.today(), rf)
            if eng.fetch_data():
                metrics = eng.get_metrics()
                weights = eng.optimize()
                sim = eng.simulate()

                t1, t2, t3 = st.tabs(["Overview", "Optimization", "Risk"])
                
                with t1:
                    st.line_chart((eng.data/eng.data.iloc[0])*100)
                    st.dataframe(metrics.style.format("{:.4f}")) # Clean table without gradient for safety

                with t2:
                    c1, c2 = st.columns([1,2])
                    with c1: 
                        st.write("Optimal Weights")
                        st.plotly_chart(px.pie(values=weights, names=tickers, hole=0.4))
                    with c2:
                        df = pd.DataFrame(sim.T, columns=['Vol', 'Ret', 'Sharpe'])
                        fig = px.scatter(df, x='Vol', y='Ret', color='Sharpe', title="Efficient Frontier")
                        opt_ret = np.sum(eng.returns.mean()*weights)*12
                        opt_std = np.sqrt(np.dot(weights.T, np.dot(eng.returns.cov()*12, weights)))
                        fig.add_trace(go.Scatter(x=[opt_std], y=[opt_ret], marker=dict(color='red', size=15), name='Max Sharpe'))
                        st.plotly_chart(fig)

                with t3:
                    cum = (1+eng.returns).cumprod()
                    dd = (cum - cum.cummax()) / cum.cummax()
                    st.plotly_chart(px.area(dd, title="Drawdowns"))
                    st.plotly_chart(px.imshow(eng.returns.corr(), text_auto=True, title="Correlations"))
            else:
                st.error("Could not fetch data. Check tickers.")
    else:
        st.info("Click 'Run Analysis' in the sidebar.")