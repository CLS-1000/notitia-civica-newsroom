#!/usr/bin/env bash
# Unpack pgvector into .pgdev/ for the ephemeral test cluster, without root and
# without touching the system PostgreSQL install. Not needed when the
# postgresql-18-pgvector package is installed system-wide, or when
# NOTITIA_TEST_DSN points at a server that already has pgvector.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="${repo_root}/.pgdev/pgvector"
work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT

(cd "${work}" && apt-get download postgresql-18-pgvector)
dpkg-deb -x "${work}"/postgresql-18-pgvector_*.deb "${work}/root"

rm -rf "${dest}"
mkdir -p "${dest}/share/extension" "${dest}/lib"
cp "${work}"/root/usr/share/postgresql/18/extension/vector* "${dest}/share/extension/"
cp "${work}"/root/usr/lib/postgresql/18/lib/vector.so "${dest}/lib/"

# '$libdir/vector' would only resolve inside the system libdir. A bare module
# name is looked up through dynamic_library_path, which the test cluster sets.
sed -i "s|^module_pathname = .*|module_pathname = 'vector'|" "${dest}/share/extension/vector.control"

echo "pgvector unpacked to ${dest}"
