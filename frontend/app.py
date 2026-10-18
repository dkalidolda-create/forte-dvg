"""Run the custom web interface: python -m frontend.app (http://127.0.0.1:8502)."""
import os

import uvicorn

from frontend.server import create_app

app = create_app()

if __name__ == '__main__':
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        streamlit_context = get_script_run_ctx(suppress_warning=True)
    except ImportError:
        streamlit_context = None
    if streamlit_context is not None:
        import streamlit as st
        st.set_page_config(page_title='Forte DVG', page_icon='✧')
        st.title('Forte DVG')
        st.write('Новый веб-интерфейс открывается отдельно.')
        st.link_button('Открыть новый интерфейс ↗', 'http://127.0.0.1:8502')
        st.caption('Если страница не открывается, выполните в новом терминале:')
        st.code('python -m frontend.app', language='bash')
    else:
        uvicorn.run(app, host='127.0.0.1', port=int(os.getenv('FRONTEND_PORT', '8502')))
