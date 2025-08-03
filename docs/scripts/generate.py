from pathlib import Path
import mkdocs_gen_files
import logging

log = logging.getLogger("mkdocs")

src_package_dir = Path("y5gfunc")

doc_api_root = Path("API")
package_name = src_package_dir.name

log.info(f"Starting reference page generation from source directory: {src_package_dir}")

for path in sorted(src_package_dir.rglob("*.py")):

    if "test" in path.parts:
        log.debug(f"Skipping test folder file: {path}")
        continue
    
    module_path_rel_pkg = path.relative_to(src_package_dir)
    module_path_parts = module_path_rel_pkg.with_suffix("").parts

    if module_path_rel_pkg.name == "__init__.py":
        continue
    if any(part.startswith("_") for part in module_path_parts):
        log.debug(f"Skipping private module: {module_path_rel_pkg}")
        continue

    full_module_path = ".".join([package_name] + list(module_path_parts))

    short_module_name = module_path_parts[-1]

    doc_path = doc_api_root / module_path_rel_pkg.with_suffix(".md")

    log.info(
        f"Generating doc page for module: {full_module_path} at {doc_path} with nav title: '{short_module_name}'"
    )

    with mkdocs_gen_files.open(doc_path, "w") as fd:
        print("---", file=fd)
        print(f"title: {short_module_name}", file=fd)
        print("---", file=fd)
        print("", file=fd)

        print(f"# `{full_module_path}`", file=fd)
        print("", file=fd)

        identifier = f"::: {full_module_path}"
        print(identifier, file=fd)

    mkdocs_gen_files.set_edit_path(
        doc_path,
        (
            Path("../..") / path
            if src_package_dir.name == package_name
            else Path("../../..") / path
        ),
    )


log.info("Finished reference page generation.")
