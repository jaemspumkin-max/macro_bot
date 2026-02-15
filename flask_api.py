"""
Flask API for Macro Bias Engine
================================

A simple REST API that exposes the macro bias engine for web integration.

Run with: python flask_api.py
Access at: http://localhost:5000/api/bias

Endpoints:
    GET /api/bias - Get current bias analysis (dashboard format)
    GET /api/bias/detailed - Get detailed analysis with all metrics
    GET /api/health - Health check endpoint
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime
import json

# Import the macro bias engine
try:
    from macro_bias_engine import MacroBiasEngine, OutputFormatter
except ImportError:
    print("Error: macro_bias_engine.py not found in the same directory")
    print("Please ensure macro_bias_engine.py is in the same folder as flask_api.py")
    exit(1)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for web dashboard access

# Cache for results (avoid fetching data on every request)
_cache = {
    'last_update': None,
    'results': None,
    'cache_duration_minutes': 15  # Cache results for 15 minutes
}


def get_cached_results():
    """Get cached results or fetch new ones if cache is stale."""
    now = datetime.now()
    
    # Check if cache is valid
    if _cache['last_update'] is not None:
        time_diff = (now - _cache['last_update']).total_seconds() / 60
        if time_diff < _cache['cache_duration_minutes']:
            return _cache['results']
    
    # Fetch new results
    print(f"[{now}] Fetching new macro bias analysis...")
    engine = MacroBiasEngine()
    results = engine.run_analysis()
    
    # Update cache
    _cache['last_update'] = now
    _cache['results'] = results
    
    return results


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'Macro Bias Engine API',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0'
    })


@app.route('/api/bias', methods=['GET'])
def get_bias():
    """
    Get current macro bias analysis (dashboard-ready format).
    
    Returns:
        JSON with overall bias, strength, confidence, volatility, regime, and factors
    """
    try:
        results = get_cached_results()
        dashboard_data = OutputFormatter.to_dashboard_dict(results)
        
        return jsonify({
            'success': True,
            'data': dashboard_data,
            'cache_age_minutes': int((datetime.now() - _cache['last_update']).total_seconds() / 60)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/bias/detailed', methods=['GET'])
def get_bias_detailed():
    """
    Get detailed macro bias analysis with all metrics.
    
    Returns:
        Complete JSON with all factor scores, volatility components, etc.
    """
    try:
        results = get_cached_results()
        
        return jsonify({
            'success': True,
            'data': results,
            'cache_age_minutes': int((datetime.now() - _cache['last_update']).total_seconds() / 60)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/bias/factors', methods=['GET'])
def get_factors():
    """
    Get just the factor breakdown.
    
    Returns:
        JSON array of factor scores and contributions
    """
    try:
        results = get_cached_results()
        
        return jsonify({
            'success': True,
            'factors': results['factor_scores'],
            'timestamp': results['timestamp']
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/bias/refresh', methods=['POST'])
def refresh_analysis():
    """
    Force refresh of the analysis (bypass cache).
    
    Returns:
        Fresh analysis results
    """
    try:
        # Clear cache
        _cache['last_update'] = None
        _cache['results'] = None
        
        # Get fresh results
        results = get_cached_results()
        dashboard_data = OutputFormatter.to_dashboard_dict(results)
        
        return jsonify({
            'success': True,
            'message': 'Analysis refreshed successfully',
            'data': dashboard_data
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/bias/config', methods=['GET', 'POST'])
def get_or_update_config():
    """
    Get or update engine configuration (weights, cache duration).
    
    GET: Returns current configuration
    POST: Updates configuration (requires JSON body with 'weights' and/or 'cache_duration_minutes')
    """
    if request.method == 'GET':
        return jsonify({
            'success': True,
            'config': {
                'cache_duration_minutes': _cache['cache_duration_minutes'],
                'default_weights': {
                    '10Y_Treasury_Yield': 2.0,
                    'DXY_Dollar_Index': 2.0,
                    'M2_Money_Supply': 2.0,
                    'Credit_Spreads_BAA_AAA': 1.5,
                    'VIX_Index': 1.5,
                    'Economic_Surprises': 1.0
                }
            }
        })
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            
            if 'cache_duration_minutes' in data:
                _cache['cache_duration_minutes'] = int(data['cache_duration_minutes'])
            
            # If weights are provided, they would need to be stored and used
            # For now, weights are handled at engine initialization
            
            return jsonify({
                'success': True,
                'message': 'Configuration updated',
                'config': {
                    'cache_duration_minutes': _cache['cache_duration_minutes']
                }
            })
        
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 400


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': [
            '/api/health',
            '/api/bias',
            '/api/bias/detailed',
            '/api/bias/factors',
            '/api/bias/refresh',
            '/api/bias/config'
        ]
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': str(error)
    }), 500


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 MACRO BIAS ENGINE API")
    print("="*70)
    print("\nAPI Endpoints:")
    print("  • GET  /api/health              - Health check")
    print("  • GET  /api/bias                - Dashboard-ready bias data")
    print("  • GET  /api/bias/detailed       - Detailed analysis")
    print("  • GET  /api/bias/factors        - Factor breakdown only")
    print("  • POST /api/bias/refresh        - Force refresh analysis")
    print("  • GET  /api/bias/config         - Get configuration")
    print("  • POST /api/bias/config         - Update configuration")
    print("\n" + "="*70)
    print(f"\n✅ Server starting on http://localhost:5000")
    print("📊 Access dashboard data at: http://localhost:5000/api/bias")
    print("\n" + "="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
