# Releasing matchup

## PyPI trusted publishing

The release workflow publishes to PyPI without a long-lived API token. Before
the first PyPI release, add a pending trusted publisher in the PyPI account:

- PyPI project name: `matchup`
- GitHub owner: `hurd-git`
- GitHub repository: `matchup`
- Workflow name: `release.yml`
- Environment name: `pypi`

Also create the `pypi` environment in the GitHub repository. The workflow has
`id-token: write` permission and uses the official PyPI publish action.

## Version release

1. Update the version in `pyproject.toml`, `src/matchup/__init__.py`, and the
   version test.
2. Move the changelog entries into a dated version section.
3. Run the tests and build checks locally.
4. Commit and push the release changes.
5. Create and push `v<version>`. The tag builds CPython 3.10 through 3.14
   Windows x64 wheels and an sdist, then creates the GitHub Release.
6. Verify the GitHub Release artifacts.
7. Run the `Release` workflow manually with `publish_pypi=true`. The workflow
   derives `v<version>` from `pyproject.toml`, downloads that existing GitHub
   Release, verifies its asset names and metadata, and publishes it through
   trusted publishing.

PyPI versions and uploaded files cannot be replaced. Always verify the version
and artifacts before running the publishing job.
