from nox_poetry import session


@session(python=["3.10", "3.11", "3.12", "3.13"])
def tests(session):
    session.run("poetry", "install", "--with", "test")
    # session.install("joblib", ".")
    session.run("pytest", "tests/")
