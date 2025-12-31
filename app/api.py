"""
MeenaSetu - Flask API with Landing Page
Complete REST API with documentation page
"""

from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS
import sys
sys.path.append('.')
from app.fish_database import FishDatabase

app = Flask(__name__)
CORS(app)
db = FishDatabase()

# HTML Landing Page Template
LANDING_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🐟 MeenaSetu Fish API</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        .header h1 { font-size: 3em; margin-bottom: 10px; }
        .header p { font-size: 1.2em; opacity: 0.9; }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 40px;
            background: #f8f9fa;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stat-card h3 { font-size: 2.5em; color: #667eea; margin-bottom: 5px; }
        .stat-card p { color: #666; }
        .content { padding: 40px; }
        .section { margin-bottom: 40px; }
        .section h2 {
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }
        .endpoint {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            border-left: 4px solid #667eea;
        }
        .endpoint-header {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
        .method {
            background: #28a745;
            color: white;
            padding: 5px 15px;
            border-radius: 5px;
            font-weight: bold;
            margin-right: 15px;
            font-size: 0.9em;
        }
        .path {
            font-family: 'Courier New', monospace;
            font-size: 1.1em;
            color: #333;
        }
        .description { color: #666; margin-top: 10px; }
        .example {
            background: #263238;
            color: #aed581;
            padding: 15px;
            border-radius: 5px;
            margin-top: 10px;
            font-family: 'Courier New', monospace;
            overflow-x: auto;
        }
        .try-button {
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin-top: 10px;
            font-size: 1em;
        }
        .try-button:hover { background: #5568d3; }
        .response {
            background: #e8f5e9;
            padding: 15px;
            border-radius: 5px;
            margin-top: 10px;
            display: none;
            max-height: 300px;
            overflow-y: auto;
        }
        .footer {
            background: #263238;
            color: white;
            text-align: center;
            padding: 30px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🐟 MeenaSetu Fish API</h1>
            <p>West Bengal Freshwater Fish Database</p>
            <p style="margin-top: 10px; font-size: 0.9em;">v1.0 - Production Ready</p>
        </div>

        <div class="stats">
            <div class="stat-card">
                <h3>{{ stats.total_species }}</h3>
                <p>Total Species</p>
            </div>
            <div class="stat-card">
                <h3>{{ stats.families }}</h3>
                <p>Families</p>
            </div>
            <div class="stat-card">
                <h3>{{ stats.orders }}</h3>
                <p>Orders</p>
            </div>
            <div class="stat-card">
                <h3>{{ stats.with_local_names }}</h3>
                <p>With Local Names</p>
            </div>
        </div>

        <div class="content">
            <div class="section">
                <h2>📡 API Endpoints</h2>

                <div class="endpoint">
                    <div class="endpoint-header">
                        <span class="method">GET</span>
                        <span class="path">/api/species</span>
                    </div>
                    <div class="description">Get all species (with optional limit)</div>
                    <div class="example">curl http://localhost:5000/api/species?limit=10</div>
                    <button class="try-button" onclick="tryEndpoint('/api/species?limit=5', 'resp1')">Try It!</button>
                    <div id="resp1" class="response"></div>
                </div>

                <div class="endpoint">
                    <div class="endpoint-header">
                        <span class="method">GET</span>
                        <span class="path">/api/species/{scientific_name}</span>
                    </div>
                    <div class="description">Get specific species by scientific name</div>
                    <div class="example">curl http://localhost:5000/api/species/Labeo%20rohita</div>
                    <button class="try-button" onclick="tryEndpoint('/api/species/Labeo rohita', 'resp2')">Try It!</button>
                    <div id="resp2" class="response"></div>
                </div>

                <div class="endpoint">
                    <div class="endpoint-header">
                        <span class="method">GET</span>
                        <span class="path">/api/search?q={query}</span>
                    </div>
                    <div class="description">Search species by any name (scientific, local, or common)</div>
                    <div class="example">curl http://localhost:5000/api/search?q=rohu</div>
                    <button class="try-button" onclick="tryEndpoint('/api/search?q=rohu', 'resp3')">Try It!</button>
                    <div id="resp3" class="response"></div>
                </div>

                <div class="endpoint">
                    <div class="endpoint-header">
                        <span class="method">GET</span>
                        <span class="path">/api/family/{family_name}</span>
                    </div>
                    <div class="description">Get all species in a family</div>
                    <div class="example">curl http://localhost:5000/api/family/Cyprinidae</div>
                    <button class="try-button" onclick="tryEndpoint('/api/family/Cyprinidae', 'resp4')">Try It!</button>
                    <div id="resp4" class="response"></div>
                </div>

                <div class="endpoint">
                    <div class="endpoint-header">
                        <span class="method">GET</span>
                        <span class="path">/api/stats</span>
                    </div>
                    <div class="description">Get database statistics</div>
                    <div class="example">curl http://localhost:5000/api/stats</div>
                    <button class="try-button" onclick="tryEndpoint('/api/stats', 'resp5')">Try It!</button>
                    <div id="resp5" class="response"></div>
                </div>

                <div class="endpoint">
                    <div class="endpoint-header">
                        <span class="method">GET</span>
                        <span class="path">/api/random?n={count}</span>
                    </div>
                    <div class="description">Get random species (for quiz/training)</div>
                    <div class="example">curl http://localhost:5000/api/random?n=3</div>
                    <button class="try-button" onclick="tryEndpoint('/api/random?n=3', 'resp6')">Try It!</button>
                    <div id="resp6" class="response"></div>
                </div>
            </div>

            <div class="section">
                <h2>🔝 Top Families</h2>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
                    {% for family, count in top_families.items() %}
                    <div class="stat-card">
                        <h3>{{ count }}</h3>
                        <p>{{ family }}</p>
                    </div>
                    {% endfor %}
                </div>
            </div>

            <div class="section">
                <h2>💻 Quick Start</h2>
                <div class="endpoint">
                    <h3>Python</h3>
                    <div class="example">
import requests

# Get all species
response = requests.get('http://localhost:5000/api/species?limit=10')
species = response.json()

# Search for fish
response = requests.get('http://localhost:5000/api/search?q=rohu')
results = response.json()
                    </div>
                </div>

                <div class="endpoint">
                    <h3>JavaScript</h3>
                    <div class="example">
// Fetch all species
fetch('http://localhost:5000/api/species?limit=10')
  .then(res => res.json())
  .then(data => console.log(data));

// Search for fish
fetch('http://localhost:5000/api/search?q=rohu')
  .then(res => res.json())
  .then(data => console.log(data));
                    </div>
                </div>
            </div>
        </div>

        <div class="footer">
            <p>🐟 MeenaSetu Fish Database API</p>
            <p style="margin-top: 10px; opacity: 0.8;">Built for fish identification and conservation</p>
        </div>
    </div>

    <script>
        async function tryEndpoint(path, responseId) {
            const responseDiv = document.getElementById(responseId);
            responseDiv.style.display = 'block';
            responseDiv.innerHTML = '⏳ Loading...';
            
            try {
                const response = await fetch(path);
                const data = await response.json();
                responseDiv.innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
            } catch (error) {
                responseDiv.innerHTML = '<pre style="color: red;">Error: ' + error.message + '</pre>';
            }
        }
    </script>
</body>
</html>
"""

# Root endpoint - Landing page
@app.route('/')
def index():
    """API landing page with documentation"""
    stats = db.get_statistics()
    top_families = dict(list(stats['top_families'].items())[:6])
    return render_template_string(LANDING_PAGE, stats=stats, top_families=top_families)

# API Endpoints
@app.route('/api/species', methods=['GET'])
def get_all_species():
    """Get all species with optional limit"""
    limit = request.args.get('limit', type=int)
    species = db.get_all_species(limit=limit)
    return jsonify({'count': len(species), 'species': species})

@app.route('/api/species/<path:scientific_name>', methods=['GET'])
def get_species(scientific_name):
    """Get specific species"""
    species = db.get_species(scientific_name)
    if species:
        return jsonify(species)
    return jsonify({'error': 'Species not found'}), 404

@app.route('/api/search', methods=['GET'])
def search():
    """Search species by name"""
    query = request.args.get('q', '')
    if not query:
        return jsonify({'error': 'Query parameter "q" is required'}), 400
    
    results = db.search_by_name(query)
    return jsonify({'count': len(results), 'results': results})

@app.route('/api/family/<family_name>', methods=['GET'])
def get_family(family_name):
    """Get all species in a family"""
    species = db.filter_by_family(family_name)
    return jsonify({'count': len(species), 'family': family_name, 'species': species})

@app.route('/api/order/<order_name>', methods=['GET'])
def get_order(order_name):
    """Get all species in an order"""
    species = db.filter_by_order(order_name)
    return jsonify({'count': len(species), 'order': order_name, 'species': species})

@app.route('/api/iucn/<status>', methods=['GET'])
def get_iucn(status):
    """Get species by IUCN status"""
    species = db.filter_by_iucn_status(status)
    return jsonify({'count': len(species), 'status': status, 'species': species})

@app.route('/api/habitat/<habitat_type>', methods=['GET'])
def get_habitat(habitat_type):
    """Get species by habitat"""
    species = db.filter_by_habitat(habitat_type)
    return jsonify({'count': len(species), 'habitat': habitat_type, 'species': species})

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get database statistics"""
    return jsonify(db.get_statistics())

@app.route('/api/random', methods=['GET'])
def get_random():
    """Get random species"""
    n = request.args.get('n', default=1, type=int)
    if n > 50:
        return jsonify({'error': 'Maximum 50 random species allowed'}), 400
    
    species = db.get_random_species(n)
    return jsonify({'count': len(species), 'species': species})

# Health check
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'database': 'connected',
        'species_count': len(db.df)
    })

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🐟 MeenaSetu Fish API Server")
    print("="*70)
    print(f"\n✅ Database loaded: {len(db.df)} species")
    print(f"\n🌐 Server starting at: http://localhost:5000")
    print(f"📖 API Documentation: http://localhost:5000")
    print(f"\n📡 Available endpoints:")
    print("   GET  /                    - API documentation")
    print("   GET  /api/species         - Get all species")
    print("   GET  /api/species/<name>  - Get specific species")
    print("   GET  /api/search?q=...    - Search by name")
    print("   GET  /api/family/<name>   - Get family")
    print("   GET  /api/order/<name>    - Get order")
    print("   GET  /api/iucn/<status>   - Get by IUCN status")
    print("   GET  /api/stats           - Get statistics")
    print("   GET  /api/random?n=5      - Get random species")
    print("   GET  /health              - Health check")
    print("\n" + "="*70)
    print("Press CTRL+C to quit\n")
    
    app.run(debug=True, port=5000)