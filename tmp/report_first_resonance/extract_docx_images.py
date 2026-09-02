from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("docx", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    doc = Document(args.docx)
    counter = 0
    for paragraph_index, paragraph in enumerate(doc.paragraphs):
        for blip in paragraph._p.xpath(".//a:blip"):
            rid = blip.get(qn("r:embed"))
            if not rid:
                continue
            part = doc.part.related_parts[rid]
            ext = Path(str(part.partname)).suffix or ".bin"
            counter += 1
            out = args.output_dir / f"{counter:02d}_p{paragraph_index:03d}{ext}"
            out.write_bytes(part.blob)
            digest = hashlib.sha256(part.blob).hexdigest()
            print(f"{counter}\tparagraph={paragraph_index}\trid={rid}\tpart={part.partname}\tsha256={digest}\tout={out}")


if __name__ == "__main__":
    main()
