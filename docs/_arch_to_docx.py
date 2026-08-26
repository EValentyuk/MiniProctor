"""Подготовка architecture.md для конвертации в DOCX: замена mermaid-блоков на ссылки на PNG.

Запускается из c:\\Projects\\MiniProctor. Создаёт временный md, конвертирует через md2docx_plain.py, удаляет временный md.
"""
import re
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).parent.parent  # c:\Projects\MiniProctor
SRC = ROOT / "docs" / "architecture.md"
TMP = ROOT / "docs" / "_architecture_with_images.md"
DST = ROOT / "docs" / "architecture.docx"

IMAGES = [
    "diagrams/images/context.png",
    "diagrams/images/container.png",
    "diagrams/images/inference-flow.png",
    "diagrams/images/training-flow.png",
]

text = SRC.read_text(encoding="utf-8")

pattern = re.compile(r"```mermaid\n.*?\n```", re.DOTALL)
matches = pattern.findall(text)
print(f"Найдено mermaid-блоков: {len(matches)}")
print(f"PNG в очереди: {len(IMAGES)}")

if len(matches) != len(IMAGES):
    print("Не совпадает количество mermaid-блоков и PNG. Прерываю.")
    sys.exit(1)

it = iter(IMAGES)
def repl(_m):
    img = next(it)
    return f"![]({img})"

new_text = pattern.sub(repl, text)
TMP.write_text(new_text, encoding="utf-8")
print(f"Временный файл записан: {TMP}")

script = r"c:\Projects\md2docxScripts\scripts\md2docx_plain.py"
result = subprocess.run(
    ["python", "-X", "utf8", script, str(TMP), str(DST)],
    cwd=str(ROOT),
    capture_output=True,
    text=True,
    encoding="utf-8",
)
print("STDOUT:", result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

TMP.unlink(missing_ok=True)
print(f"Готово: {DST}")
