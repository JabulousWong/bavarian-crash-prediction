"""Read cached XLSX cell values with Python's standard library, read-only."""
from pathlib import Path
import posixpath
import xml.etree.ElementTree as ET
import zipfile

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


def read_sheets(path: Path) -> dict[str, dict[str, object]]:
    with zipfile.ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared = ["".join(n.itertext()) for n in ET.fromstring(
                archive.read("xl/sharedStrings.xml")).findall("s:si", NS)]
        relations = {r.attrib["Id"]: r.attrib["Target"] for r in ET.fromstring(
            archive.read("xl/_rels/workbook.xml.rels"))}
        sheets = {}
        for sheet in ET.fromstring(archive.read("xl/workbook.xml")).findall("s:sheets/s:sheet", NS):
            target = relations[sheet.attrib[REL]]
            member = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
            cells = {}
            for cell in ET.fromstring(archive.read(member)).findall(".//s:c", NS):
                value = cell.find("s:v", NS)
                if cell.attrib.get("t") == "inlineStr":
                    v = "".join(cell.find("s:is", NS).itertext())
                elif value is None or value.text is None:
                    continue
                elif cell.attrib.get("t") == "s":
                    v = shared[int(value.text)]
                else:
                    try:
                        v = float(value.text)
                        if v.is_integer():
                            v = int(v)
                    except ValueError:
                        v = value.text
                cells[cell.attrib["r"]] = v
            sheets[sheet.attrib["name"]] = cells
        return sheets


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--sheet")
    args = parser.parse_args()
    for name, cells in read_sheets(args.path).items():
        if args.sheet and name != args.sheet:
            continue
        print(name)
        for address, value in cells.items():
            print(address, repr(value))
