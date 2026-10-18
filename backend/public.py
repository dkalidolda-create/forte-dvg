"""One-origin deployment: jury API first, web interface at the remaining paths."""
import os
from dataclasses import replace

from backend.app.config import Settings
from backend.app.main import create_app
from frontend.server import UISettings
from frontend.server import create_app as create_web_app


def create_public_app(settings=None, client_factory=None):
    settings = settings or Settings.load()
    public_port = int(os.getenv('PUBLIC_PORT', '9000'))
    internal = os.getenv('PUBLIC_INTERNAL_API_URL', f'http://127.0.0.1:{public_port}')
    # The public process already serves the API. Its UI must not depend on a
    # second development server on port 8000 staying alive.
    ui = replace(UISettings.load(), api_key=settings.api_key, backend_url=internal,
                 max_upload_bytes=settings.max_upload_bytes)
    app = create_app(settings)
    app.mount('/', create_web_app(ui, client_factory), name='web')
    return app


app = create_public_app()
