def test_gradio_import():
    from dashboard.gradio_app import demo
    assert demo is not None
