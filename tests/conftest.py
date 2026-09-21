def pytest_configure(config):
    config.addinivalue_line("markers", "commerce: 상권 단계(stages.commerce=true) 회귀 — v11 기본 비활성 경로가 아님")
