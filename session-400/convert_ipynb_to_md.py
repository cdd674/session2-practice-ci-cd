from pathlib import Path
import subprocess
import sys


def convert_ipynb_to_md():
    cwd = Path.cwd()
    notebooks = list(cwd.glob("*.ipynb"))

    if not notebooks:
        print("No .ipynb files found in current directory.")
        return

    for ipynb_file in notebooks:
        md_file = ipynb_file.with_suffix(".md")

        # Skip existing markdown
        if md_file.exists():
            print(f"Skipping: {ipynb_file.name} -> {md_file.name} already exists")
            continue

        try:
            print(f"Converting: {ipynb_file.name} -> {md_file.name}")

            # Disable problematic contrib nbextensions exporters
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "markdown",
                    str(ipynb_file),
                    "--Exporter.exclude_input_prompt=True",
                    "--Exporter.exclude_output_prompt=True",
                ],
                check=True,
                env={
                    **dict(__import__("os").environ),
                    "JUPYTER_PATH": "",
                    "JUPYTER_CONFIG_DIR": "",
                },
            )

            print(f"Done: {md_file.name}")

        except subprocess.CalledProcessError:
            print(
                f"nbconvert failed for {ipynb_file.name}, trying manual conversion..."
            )

            # Fallback: pure python conversion without nbconvert CLI
            try:
                import nbformat

                nb = nbformat.read(ipynb_file, as_version=4)

                lines = []

                for cell in nb.cells:
                    if cell.cell_type == "markdown":
                        lines.append(cell.source)
                        lines.append("\n")
                    elif cell.cell_type == "code":
                        lines.append("```python")
                        lines.append(cell.source)
                        lines.append("```")
                        lines.append("\n")

                md_file.write_text("\n".join(lines), encoding="utf-8")

                print(f"Manual conversion done: {md_file.name}")

            except Exception as e:
                print(f"Failed manual conversion for {ipynb_file.name}: {e}")


if __name__ == "__main__":
    convert_ipynb_to_md()
