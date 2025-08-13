from __future__ import annotations
from flask import Flask
from pathlib import Path
from config import settings

from routes.ui import ui_bp
from routes.api import api_bp

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['OUTPUT_DIR'] = str(settings.OUTPUT_DIR)

# ensure output dir exists
Path(app.config['OUTPUT_DIR']).mkdir(exist_ok=True)

# blueprints
app.register_blueprint(ui_bp)
app.register_blueprint(api_bp, url_prefix='/api')

if __name__ == '__main__':
    # NOTE: change port if 5000 is busy
    app.run(debug=True, host='127.0.0.1', port=5000)