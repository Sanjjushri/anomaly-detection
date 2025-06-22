#!/usr/bin/env python3
"""
Trade Dataset Generator - CSV Format
Generates high-dimensional financial trade data with embedded anomalies in CSV format.
"""

import csv
import random
import math
from datetime import datetime, timedelta
from typing import List, Dict, Any

class TradeDataGenerator:
    def __init__(self):
        self.symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 
                       'JPM', 'BAC', 'WMT', 'JNJ', 'PG', 'KO', 'PFE', 'XOM']
        self.sectors = ['Technology', 'Consumer', 'Finance', 'Healthcare', 'Energy']
        
    def generate_dataset(self, size: int = 1000) -> List[Dict[str, Any]]:
        """Generate a complete dataset with normal and anomalous trades."""
        trades = []
        base_time = datetime.now()
        
        # Generate normal trades
        for i in range(size):
            trade = self._generate_normal_trade(i, base_time)
            trades.append(trade)
        
        # Add anomalies (approximately 5% of dataset)
        anomaly_count = max(1, int(size * 0.05))
        anomaly_indices = random.sample(range(size), anomaly_count)
        
        for idx in anomaly_indices:
            trades[idx] = self._generate_anomalous_trade(trades[idx])
            
        return trades
    
    def _generate_normal_trade(self, trade_id: int, base_time: datetime) -> Dict[str, Any]:
        """Generate a normal trade record with realistic financial metrics."""
        symbol = random.choice(self.symbols)
        base_price = 50 + random.random() * 400  # $50-$450 range
        
        # Generate timestamp within last 30 days
        days_back = random.random() * 30
        timestamp = base_time - timedelta(days=days_back)
        
        trade = {
            'id': f'trade_{trade_id:06d}',
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': symbol,
            'price': round(base_price * (0.95 + random.random() * 0.1), 2),  # ±5% variation
            'volume': int(1000 + random.random() * 50000),  # 1K-51K shares
            'volatility': round(0.1 + random.random() * 0.3, 4),  # 10-40% volatility
            'rsi': round(30 + random.random() * 40, 2),  # RSI 30-70 (normal range)
            'macd': round(-2 + random.random() * 4, 4),  # MACD -2 to 2
            'bollinger_position': round(0.2 + random.random() * 0.6, 4),  # 20%-80% of Bollinger band
            'market_cap': int(1e9 + random.random() * 1e12),  # $1B-$1T market cap
            'pe_ratio': round(10 + random.random() * 30, 2),  # P/E ratio 10-40
            'volume_profile': round(0.8 + random.random() * 0.4, 4),  # Volume profile 0.8-1.2
            'price_momentum': round(-0.05 + random.random() * 0.1, 4),  # ±5% momentum
            'liquidity_ratio': round(0.5 + random.random() * 0.5, 4),  # Liquidity ratio 0.5-1.0
            'order_book_imbalance': round(-0.1 + random.random() * 0.2, 4),  # Order book imbalance ±10%
            'cross_correlation': round(0.3 + random.random() * 0.4, 4),  # Cross-correlation 0.3-0.7
            'sector_beta': round(0.8 + random.random() * 0.6, 4),  # Beta 0.8-1.4
            'vwap_deviation': round(-0.02 + random.random() * 0.04, 4),  # VWAP deviation ±2%
            'time_of_day_factor': round(0.7 + random.random() * 0.6, 4),  # Time factor 0.7-1.3
            'is_anomaly': 0,  # Use 0/1 for CSV compatibility
            'anomaly_type': '',
            'anomaly_reason_1': '',
            'anomaly_reason_2': '',
            'anomaly_reason_3': ''
        }
        
        return trade
    
    def _generate_anomalous_trade(self, normal_trade: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a normal trade into an anomalous one."""
        anomaly_types = [
            'price_spike', 'volume_surge', 'extreme_volatility', 
            'unusual_rsi', 'liquidity_crisis', 'momentum_anomaly'
        ]
        
        anomaly_type = random.choice(anomaly_types)
        anomaly_reasons = []
        anomalous_trade = normal_trade.copy()
        
        if anomaly_type == 'price_spike':
            # Extreme price increase with volume surge
            multiplier = 1.5 + random.random() * 2  # 150%-250% increase
            anomalous_trade['price'] = round(anomalous_trade['price'] * multiplier, 2)
            anomalous_trade['volume'] = int(anomalous_trade['volume'] * (2 + random.random() * 3))
            anomalous_trade['volatility'] = min(0.95, anomalous_trade['volatility'] * 2)
            
            anomaly_reasons.extend([
                f'Extreme price spike: {((multiplier-1)*100):.1f}% increase',
                f'Volume surge accompanying price movement: {anomalous_trade["volume"]:,} shares',
                'Volatility spike detected'
            ])
            
        elif anomaly_type == 'volume_surge':
            # Massive volume spike
            volume_multiplier = 10 + random.random() * 20  # 10x-30x normal volume
            anomalous_trade['volume'] = int(anomalous_trade['volume'] * volume_multiplier)
            anomalous_trade['volatility'] = min(0.95, anomalous_trade['volatility'] * 1.5)
            anomalous_trade['liquidity_ratio'] = max(0.05, anomalous_trade['liquidity_ratio'] * 0.3)
            
            anomaly_reasons.extend([
                f'Massive volume surge: {volume_multiplier:.1f}x normal volume',
                f'Total volume: {anomalous_trade["volume"]:,} shares',
                'Liquidity strain detected'
            ])
            
        elif anomaly_type == 'extreme_volatility':
            # Extremely high volatility
            anomalous_trade['volatility'] = 0.8 + random.random() * 0.2  # 80%-100% volatility
            anomalous_trade['bollinger_position'] = 0.05 if random.random() < 0.5 else 0.95
            anomalous_trade['price_momentum'] = -0.2 + random.random() * 0.4  # Extreme momentum
            
            anomaly_reasons.extend([
                f'Extreme volatility: {(anomalous_trade["volatility"]*100):.1f}%',
                'Price outside normal Bollinger Bands',
                f'Extreme momentum: {(anomalous_trade["price_momentum"]*100):.1f}%'
            ])
            
        elif anomaly_type == 'unusual_rsi':
            # Extreme RSI values
            anomalous_trade['rsi'] = 5 + random.random() * 15 if random.random() < 0.5 else 85 + random.random() * 10
            anomalous_trade['price_momentum'] = -0.15 if anomalous_trade['rsi'] < 30 else 0.15
            anomalous_trade['macd'] = -8 if anomalous_trade['rsi'] < 30 else 8
            
            condition = 'Oversold' if anomalous_trade['rsi'] < 30 else 'Overbought'
            anomaly_reasons.extend([
                f'Extreme RSI: {anomalous_trade["rsi"]:.1f} ({condition} condition)',
                f'MACD divergence: {anomalous_trade["macd"]:.2f}',
                'Strong momentum divergence'
            ])
            
        elif anomaly_type == 'liquidity_crisis':
            # Liquidity crisis indicators
            anomalous_trade['liquidity_ratio'] = 0.05 + random.random() * 0.15  # Very low liquidity
            anomalous_trade['order_book_imbalance'] = 0.3 + random.random() * 0.4  # High imbalance
            anomalous_trade['volume'] = max(100, int(anomalous_trade['volume'] * 0.1))  # Very low volume
            anomalous_trade['vwap_deviation'] = -0.1 + random.random() * 0.2  # High VWAP deviation
            
            anomaly_reasons.extend([
                f'Liquidity crisis: ratio {anomalous_trade["liquidity_ratio"]:.3f}',
                f'Order book imbalance: {(anomalous_trade["order_book_imbalance"]*100):.1f}%',
                f'Low volume: {anomalous_trade["volume"]:,} shares'
            ])
            
        elif anomaly_type == 'momentum_anomaly':
            # Extreme momentum with correlation breakdown
            anomalous_trade['price_momentum'] = -0.25 + random.random() * 0.5  # ±25% momentum
            anomalous_trade['macd'] = -8 + random.random() * 16  # Extreme MACD
            anomalous_trade['cross_correlation'] = 0.05 + random.random() * 0.15  # Low correlation
            anomalous_trade['sector_beta'] = 2.0 + random.random() * 2.0  # High beta
            
            anomaly_reasons.extend([
                f'Extreme momentum: {(anomalous_trade["price_momentum"]*100):.1f}%',
                f'MACD anomaly: {anomalous_trade["macd"]:.2f}',
                f'Low market correlation: {anomalous_trade["cross_correlation"]:.3f}'
            ])
        
        anomalous_trade['is_anomaly'] = 1  # Use 1 for anomaly in CSV
        anomalous_trade['anomaly_type'] = anomaly_type
        
        # Fill anomaly reason columns (up to 3 reasons)
        for i, reason in enumerate(anomaly_reasons[:3]):
            anomalous_trade[f'anomaly_reason_{i+1}'] = reason
        
        return anomalous_trade

    def save_to_csv(self, data: List[Dict[str, Any]], filename: str) -> None:
        """Save dataset to CSV file."""
        if not data:
            print("No data to save!")
            return
        
        # Define column order for CSV
        columns = [
            'id', 'timestamp', 'symbol', 'price', 'volume', 'volatility', 'rsi', 'macd',
            'bollinger_position', 'market_cap', 'pe_ratio', 'volume_profile', 'price_momentum',
            'liquidity_ratio', 'order_book_imbalance', 'cross_correlation', 'sector_beta',
            'vwap_deviation', 'time_of_day_factor', 'is_anomaly', 'anomaly_type',
            'anomaly_reason_1', 'anomaly_reason_2', 'anomaly_reason_3'
        ]
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            writer.writerows(data)
        
        print(f"Dataset saved to {filename}")

def main():
    """Generate and save the trade dataset in CSV format."""
    print("Trade Dataset Generator - CSV Format")
    print("=" * 50)
    
    generator = TradeDataGenerator()
    
    # Generate datasets of different sizes
    datasets = {
        'small': generator.generate_dataset(500),
        'medium': generator.generate_dataset(1000),
        'large': generator.generate_dataset(2000)
    }
    
    for size, data in datasets.items():
        filename = f'trade_dataset_{size}.csv'
        generator.save_to_csv(data, filename)
        
        # Print dataset statistics
        anomalies = sum(1 for trade in data if trade['is_anomaly'])
        print(f"\n{size.upper()} Dataset ({filename}):")
        print(f"  Total trades: {len(data)}")
        print(f"  Anomalies: {anomalies} ({(anomalies/len(data)*100):.1f}%)")
        print(f"  Normal trades: {len(data) - anomalies}")
        
        # Show anomaly types
        anomaly_types = {}
        for trade in data:
            if trade['is_anomaly']:
                atype = trade.get('anomaly_type', 'unknown')
                anomaly_types[atype] = anomaly_types.get(atype, 0) + 1
        
        print(f"  Anomaly types: {dict(anomaly_types)}")
    
    print(f"\n✅ All datasets generated successfully!")
    print(f"📊 Files created:")
    for size in datasets.keys():
        print(f"   - trade_dataset_{size}.csv")
    
    print(f"\n📋 CSV Format Features:")
    print(f"   - Easy to import into pandas, Excel, R, etc.")
    print(f"   - Anomaly flags as 0/1 for easy filtering")
    print(f"   - Separate columns for anomaly reasons")
    print(f"   - Standardized timestamp format")
    print(f"   - All numeric data properly formatted")

if __name__ == "__main__":
    main()