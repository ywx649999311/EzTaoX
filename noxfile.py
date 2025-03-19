from nox_poetry import session

PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13"]


@session(python=PYTHON_VERSIONS)
def tests(session):
    tmpPath = session.poetry.export_requirements()
    session.run(
        "poetry",
        "export",
        "--without-hashes",
        "--with",
        "test",
        "-o",
        f"{tmpPath}",
    )
    session.install("-r", f"{tmpPath}")
    session.install(".")
    session.run("pytest", "tests/")
