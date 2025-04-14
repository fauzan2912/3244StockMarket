"""
Fixed version of correlation_analysis.py with error handling for division by zero
and relative paths instead of absolute paths.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.cluster import KMeans
import os
import warnings
warnings.filterwarnings('ignore')

def load_stock_data(stocks, data_dir):
    """
    Load stock data for multiple stocks.
    
    Args:
        stocks: List of stock symbols
        data_dir: Directory containing stock data files
        
    Returns:
        Dictionary of DataFrames with stock data
    """
    stock_dfs = {}
    
    for stock in stocks:
        try:
            # In a real scenario, you would load from actual files
            # For this example, we'll create some dummy data
            dates = pd.date_range(start='2010-01-01', end='2017-12-31')
            np.random.seed(hash(stock) % 2**32)  # Different seed for each stock
            
            # Generate random price data with some correlation structure
            n = len(dates)
            close_prices = np.random.normal(loc=100, scale=1, size=n).cumsum()
            daily_volatility = 2
            
            df = pd.DataFrame({
                'Date': dates,
                'Open': close_prices + np.random.normal(0, daily_volatility, n),
                'High': close_prices + np.random.normal(daily_volatility, daily_volatility/2, n),
                'Low': close_prices - np.random.normal(daily_volatility, daily_volatility/2, n),
                'Close': close_prices,
                'Volume': np.random.normal(1000000, 200000, n).astype(int),
                'OpenInt': np.zeros(n)
            })
            
            df.set_index('Date', inplace=True)
            stock_dfs[stock] = df
            
            print(f"✅ Loaded data for {stock}")
        except Exception as e:
            print(f"❌ Error loading data for {stock}: {e}")
    
    return stock_dfs

def calculate_pairwise_correlations(stock_dfs, window=None):
    """
    Calculate pairwise correlations between stocks.
    
    Args:
        stock_dfs: Dictionary of DataFrames with stock data
        window: Rolling window size (optional)
        
    Returns:
        DataFrame with pairwise correlations
    """
    # Extract close prices
    close_prices = pd.DataFrame({stock: df['Close'] for stock, df in stock_dfs.items()})
    
    # Calculate returns
    returns = close_prices.pct_change().dropna()
    
    if window is None:
        # Calculate static correlation matrix
        corr_matrix = returns.corr()
    else:
        # Calculate rolling correlation matrix
        corr_matrices = {}
        for i in range(window, len(returns) + 1):
            window_returns = returns.iloc[i-window:i]
            corr_matrices[returns.index[i-1]] = window_returns.corr()
        
        # Convert to 3D array (time x stocks x stocks)
        dates = list(corr_matrices.keys())
        stocks = list(stock_dfs.keys())
        corr_3d = np.array([corr_matrices[date].values for date in dates])
        
        # Return the last correlation matrix
        corr_matrix = corr_matrices[dates[-1]]
    
    return corr_matrix

def visualize_correlation_matrix(corr_matrix, title="Stock Correlation Matrix", output_file=None):
    """
    Visualize correlation matrix as a heatmap.
    
    Args:
        corr_matrix: Correlation matrix
        title: Plot title
        output_file: Output file path (optional)
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title(title)
    plt.tight_layout()
    
    if output_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file)
        print(f"Saved correlation matrix to {output_file}")
    else:
        plt.show()
    
    plt.close()

def visualize_correlation_network(corr_matrix, threshold=0.5, title="Stock Correlation Network", output_file=None):
    """
    Visualize correlation matrix as a network.
    
    Args:
        corr_matrix: Correlation matrix
        threshold: Correlation threshold for drawing edges
        title: Plot title
        output_file: Output file path (optional)
    """
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    for stock in corr_matrix.index:
        G.add_node(stock)
    
    # Add edges for correlations above threshold
    for i, stock1 in enumerate(corr_matrix.index):
        for j, stock2 in enumerate(corr_matrix.index):
            if i < j:  # Avoid duplicate edges
                correlation = corr_matrix.iloc[i, j]
                if abs(correlation) >= threshold:
                    G.add_edge(stock1, stock2, weight=correlation)
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Position nodes using spring layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
    
    # Draw edges with colors based on correlation
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    # Positive correlations in red, negative in blue
    pos_edges = [(u, v) for u, v, w in G.edges(data='weight') if w >= 0]
    neg_edges = [(u, v) for u, v, w in G.edges(data='weight') if w < 0]
    
    pos_weights = [G[u][v]['weight'] for u, v in pos_edges]
    neg_weights = [abs(G[u][v]['weight']) for u, v in neg_edges]
    
    # Draw positive edges
    nx.draw_networkx_edges(G, pos, edgelist=pos_edges, width=[w*5 for w in pos_weights], 
                          edge_color='red', alpha=0.7)
    
    # Draw negative edges
    nx.draw_networkx_edges(G, pos, edgelist=neg_edges, width=[w*5 for w in neg_weights], 
                          edge_color='blue', alpha=0.7)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    
    if output_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file)
        print(f"Saved correlation network to {output_file}")
    else:
        plt.show()
    
    plt.close()

def detect_correlation_regimes(stock_dfs, window=60, n_regimes=3):
    """
    Detect correlation regimes using clustering.
    
    Args:
        stock_dfs: Dictionary of DataFrames with stock data
        window: Rolling window size
        n_regimes: Number of correlation regimes
        
    Returns:
        DataFrame with regime labels and regime centroids
    """
    # Extract close prices
    close_prices = pd.DataFrame({stock: df['Close'] for stock, df in stock_dfs.items()})
    
    # Calculate returns
    returns = close_prices.pct_change().dropna()
    
    # Calculate rolling correlation matrices
    corr_matrices = {}
    for i in range(window, len(returns) + 1):
        window_returns = returns.iloc[i-window:i]
        corr_matrices[returns.index[i-1]] = window_returns.corr()
    
    # Convert correlation matrices to vectors (flatten upper triangular part)
    corr_vectors = []
    dates = []
    
    for date, corr_matrix in corr_matrices.items():
        # Get upper triangular indices
        indices = np.triu_indices_from(corr_matrix, k=1)
        
        # Extract values
        vector = corr_matrix.values[indices]
        
        corr_vectors.append(vector)
        dates.append(date)
    
    # Convert to array
    X = np.array(corr_vectors)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_regimes, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # Create DataFrame with regime labels
    regimes_df = pd.DataFrame({
        'Date': dates,
        'Regime': labels
    })
    regimes_df.set_index('Date', inplace=True)
    
    # Calculate regime centroids (average correlation matrix for each regime)
    regime_centroids = {}
    
    for regime in range(n_regimes):
        # Get dates for this regime
        regime_dates = regimes_df[regimes_df['Regime'] == regime].index
        
        # Get correlation matrices for these dates
        regime_matrices = [corr_matrices[date] for date in regime_dates if date in corr_matrices]
        
        # Calculate average correlation matrix
        if regime_matrices:
            avg_matrix = sum(regime_matrices) / len(regime_matrices)
            regime_centroids[regime] = avg_matrix
    
    return regimes_df, regime_centroids

def visualize_correlation_regimes(regimes_df, regime_centroids, output_dir=None):
    """
    Visualize correlation regimes.
    
    Args:
        regimes_df: DataFrame with regime labels
        regime_centroids: Dictionary of regime centroids
        output_dir: Output directory for plots (optional)
    """
    # Plot regime distribution over time
    plt.figure(figsize=(15, 5))
    plt.plot(regimes_df.index, regimes_df['Regime'], marker='o', linestyle='-', alpha=0.7)
    plt.title('Correlation Regimes Over Time')
    plt.xlabel('Date')
    plt.ylabel('Regime')
    plt.grid(True)
    
    if output_dir:
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/correlation_regimes_time.png")
        print(f"Saved regime time plot to {output_dir}/correlation_regimes_time.png")
    else:
        plt.show()
    
    plt.close()
    
    # Plot regime centroids
    for regime, centroid in regime_centroids.items():
        plt.figure(figsize=(10, 8))
        sns.heatmap(centroid, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title(f'Correlation Matrix for Regime {regime}')
        plt.tight_layout()
        
        if output_dir:
            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f"{output_dir}/correlation_regime_{regime}.png")
            print(f"Saved regime {regime} plot to {output_dir}/correlation_regime_{regime}.png")
        else:
            plt.show()
        
        plt.close()

def detect_lead_lag_relationships(stock_dfs, max_lag=5):
    """
    Detect lead-lag relationships between stocks using simple correlation analysis.
    
    Args:
        stock_dfs: Dictionary of DataFrames with stock data
        max_lag: Maximum lag to test
        
    Returns:
        DataFrame with lead-lag relationships
    """
    # Extract close prices
    close_prices = pd.DataFrame({stock: df['Close'] for stock, df in stock_dfs.items()})
    
    # Calculate returns
    returns = close_prices.pct_change().dropna()
    
    # Test lead-lag relationships for each pair of stocks
    results = []
    
    for stock1 in returns.columns:
        for stock2 in returns.columns:
            if stock1 != stock2:
                # Test if stock1 leads stock2
                best_lag = 1
                best_corr = 0
                min_p_value = 1.0  # Initialize with maximum possible p-value
                
                for lag in range(1, max_lag + 1):
                    # Calculate lagged correlation
                    corr = returns[stock1].shift(lag).corr(returns[stock2])
                    
                    # Simple p-value approximation (not using actual statistical test)
                    # This is a placeholder - in a real implementation, use proper statistical tests
                    n = len(returns) - lag
                    t_stat = corr * np.sqrt((n - 2) / (1 - corr**2))
                    p_value = 2 * (1 - abs(np.minimum(0.999, t_stat) / np.sqrt(n)))  # Simplified p-value calculation
                    
                    if abs(corr) > abs(best_corr):
                        best_lag = lag
                        best_corr = corr
                        min_p_value = p_value
                
                # Check if the relationship is significant
                is_significant = min_p_value < 0.05 and abs(best_corr) > 0.2
                
                results.append({
                    'Leader': stock1,
                    'Follower': stock2,
                    'Best_Lag': best_lag,
                    'Correlation': best_corr,
                    'P_Value': min_p_value,
                    'Is_Significant': is_significant
                })
    
    # Convert to DataFrame
    lead_lag_df = pd.DataFrame(results)
    
    # Filter significant relationships
    significant_df = lead_lag_df[lead_lag_df['Is_Significant']]
    
    return lead_lag_df, significant_df

def visualize_lead_lag_network(significant_df, title="Lead-Lag Network", output_file=None):
    """
    Visualize lead-lag relationships as a directed network.
    
    Args:
        significant_df: DataFrame with significant lead-lag relationships
        title: Plot title
        output_file: Output file path (optional)
    """
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes
    all_stocks = set(significant_df['Leader']).union(set(significant_df['Follower']))
    for stock in all_stocks:
        G.add_node(stock)
    
    # Add edges with error handling for division by zero
    for _, row in significant_df.iterrows():
        # Avoid division by zero
        weight = 1.0 / max(row['P_Value'], 1e-10)  # Add small epsilon to avoid division by zero
        G.add_edge(row['Leader'], row['Follower'], weight=weight, lag=row['Best_Lag'])
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Position nodes using spring layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
    
    # Draw edges with colors based on lag
    edges = G.edges()
    lags = [G[u][v]['lag'] for u, v in edges]
    
    # Normalize lags for color mapping
    max_lag = max(lags) if lags else 1  # Avoid division by zero if lags is empty
    norm_lags = [lag/max_lag for lag in lags]
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=2, edge_color=norm_lags, edge_cmap=plt.cm.cool, 
                          arrowsize=20, connectionstyle='arc3,rad=0.1')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    
    if output_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file)
        print(f"Saved lead-lag network to {output_file}")
    else:
        plt.show()
    
    plt.close()

def create_correlation_features(stock_dfs, target_stock, window_sizes=[5, 10, 20, 50, 100]):
    """
    Create correlation-based features for a target stock.
    
    Args:
        stock_dfs: Dictionary of DataFrames with stock data
        target_stock: Target stock symbol
        window_sizes: List of rolling window sizes
        
    Returns:
        DataFrame with correlation features
    """
    # Get target stock data
    target_df = stock_dfs[target_stock].copy()
    
    # Calculate returns for all stocks
    returns = {}
    for stock, df in stock_dfs.items():
        returns[stock] = df['Close'].pct_change()
    
    # Create correlation features
    for stock in stock_dfs:
        if stock != target_stock:
            # Price correlation
            for window in window_sizes:
                target_df[f'Corr_{stock}_Price_{window}d'] = target_df['Close'].rolling(window).corr(
                    stock_dfs[stock]['Close'])
            
            # Return correlation
            for window in window_sizes:
                target_df[f'Corr_{stock}_Return_{window}d'] = returns[target_stock].rolling(window).corr(
                    returns[stock])
            
            # Volume correlation
            for window in window_sizes:
                target_df[f'Corr_{stock}_Volume_{window}d'] = target_df['Volume'].rolling(window).corr(
                    stock_dfs[stock]['Volume'])
            
            # Beta (systematic risk)
            for window in window_sizes:
                # Calculate covariance
                cov = returns[target_stock].rolling(window).cov(returns[stock])
                
                # Calculate variance of other stock
                var = returns[stock].rolling(window).var()
                
                # Calculate beta with error handling for division by zero
                target_df[f'Beta_{stock}_{window}d'] = cov / var.replace(0, np.nan)
            
            # Lead-lag features
            for lag in [1, 2, 3, 5]:
                # Lagged correlation
                target_df[f'Lead_Lag_{stock}_{lag}d'] = returns[target_stock].corr(returns[stock].shift(lag))
                
                # Lagged return
                target_df[f'Lagged_Return_{stock}_{lag}d'] = returns[stock].shift(lag)
    
    # Add correlation regime features
    try:
        regimes_df, _ = detect_correlation_regimes(stock_dfs, window=60, n_regimes=3)
        
        # Merge regimes with target dataframe
        target_df = target_df.join(regimes_df)
        
        # Create dummy variables for regimes
        for regime in range(3):
            target_df[f'Regime_{regime}'] = (target_df['Regime'] == regime).astype(int)
        
        # Drop the original Regime column
        target_df.drop('Regime', axis=1, inplace=True)
    except Exception as e:
        print(f"Warning: Could not add regime features: {e}")
    
    # Drop rows with NaN values
    target_df.dropna(inplace=True)
    
    return target_df

def analyze_feature_importance(model, feature_names, top_n=20, output_file=None):
    """
    Analyze feature importance from XGBoost model.
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        top_n: Number of top features to display
        output_file: Output file path (optional)
        
    Returns:
        DataFrame with feature importance
    """
    # Get feature importance
    importance = model.get_score(importance_type='gain')
    
    # Convert to DataFrame
    importance_df = pd.DataFrame({
        'Feature': list(importance.keys()),
        'Importance': list(importance.values())
    })
    
    # Sort by importance
    importance_df.sort_values('Importance', ascending=False, inplace=True)
    
    # Get top N features
    top_features = importance_df.head(top_n)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title(f'Top {top_n} Feature Importance')
    plt.tight_layout()
    
    if output_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file)
        print(f"Saved feature importance plot to {output_file}")
    else:
        plt.show()
    
    plt.close()
    
    return importance_df

def analyze_correlation_feature_impact(importance_df):
    """
    Analyze the impact of correlation features on model performance.
    
    Args:
        importance_df: DataFrame with feature importance
        
    Returns:
        Dictionary with correlation feature impact metrics
    """
    # Identify correlation features
    corr_features = importance_df[importance_df['Feature'].str.contains('Corr_|Beta_|Lead_Lag_|Regime_')]
    
    # Calculate total importance
    total_importance = importance_df['Importance'].sum()
    
    # Calculate correlation feature importance
    corr_importance = corr_features['Importance'].sum()
    
    # Calculate percentage
    corr_percentage = (corr_importance / total_importance) * 100
    
    # Group correlation features by type
    feature_types = {
        'Price_Correlation': corr_features[corr_features['Feature'].str.contains('Corr_.*_Price_')],
        'Return_Correlation': corr_features[corr_features['Feature'].str.contains('Corr_.*_Return_')],
        'Volume_Correlation': corr_features[corr_features['Feature'].str.contains('Corr_.*_Volume_')],
        'Beta': corr_features[corr_features['Feature'].str.contains('Beta_')],
        'Lead_Lag': corr_features[corr_features['Feature'].str.contains('Lead_Lag_')],
        'Regime': corr_features[corr_features['Feature'].str.contains('Regime_')]
    }
    
    # Calculate importance by type
    type_importance = {}
    for feature_type, features in feature_types.items():
        type_importance[feature_type] = features['Importance'].sum()
        type_importance[f'{feature_type}_pct'] = (features['Importance'].sum() / total_importance) * 100
    
    # Return results
    return {
        'total_features': len(importance_df),
        'correlation_features': len(corr_features),
        'total_importance': total_importance,
        'correlation_importance': corr_importance,
        'correlation_percentage': corr_percentage,
        'type_importance': type_importance
    }

# Example usage
if __name__ == "__main__":
    # This is just an example of how to use these functions
    # In practice, you would integrate them into the main XGBoost optimization script
    
    # Define stocks
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB']
    
    # Create output directory
    output_dir = "correlation_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load stock data
    stock_dfs = load_stock_data(stocks, "data")
    
    # Calculate pairwise correlations
    corr_matrix = calculate_pairwise_correlations(stock_dfs)
    
    # Visualize correlation matrix
    visualize_correlation_matrix(corr_matrix, output_file=f"{output_dir}/correlation_matrix.png")
    
    # Visualize correlation network
    visualize_correlation_network(corr_matrix, threshold=0.5, output_file=f"{output_dir}/correlation_network.png")
    
    # Detect correlation regimes
    regimes_df, regime_centroids = detect_correlation_regimes(stock_dfs, window=60, n_regimes=3)
    
    # Visualize correlation regimes
    visualize_correlation_regimes(regimes_df, regime_centroids, output_dir=output_dir)
    
    # Detect lead-lag relationships
    lead_lag_df, significant_df = detect_lead_lag_relationships(stock_dfs, max_lag=5)
    
    # Visualize lead-lag network
    visualize_lead_lag_network(significant_df, output_file=f"{output_dir}/lead_lag_network.png")
    
    # Create correlation features for a target stock
    target_stock = 'AAPL'
    target_df = create_correlation_features(stock_dfs, target_stock)
    
    # Print feature names
    print(f"Total features: {len(target_df.columns)}")
    print("Correlation feature names:")
    for col in target_df.columns:
        if 'Corr_' in col or 'Beta_' in col or 'Lead_Lag_' in col or 'Regime_' in col:
            print(f"- {col}")
    
    # Save correlation features
    target_df.to_csv(f"{output_dir}/{target_stock}_correlation_features.csv")
    print(f"Saved correlation features to {output_dir}/{target_stock}_correlation_features.csv")
