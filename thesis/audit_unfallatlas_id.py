"""Audit identifier date and proposed HHMM against published fields. Read-only inputs."""
import argparse, collections, csv, datetime, hashlib, io, json, pathlib, zipfile
parser = argparse.ArgumentParser()
parser.add_argument("--source", default=str(pathlib.Path(__file__).resolve().parents[1] / "accidents"))
parser.add_argument("--output", default="audit_results")
args = parser.parse_args()
out = pathlib.Path(args.output)
out.mkdir(parents=True, exist_ok=True)
summaries, anomalies, offsets, sources = [], [], [], []
examples = []
for year in range(2016, 2026):
    path = pathlib.Path(args.source) / f"Unfallorte{year}_EPSG25832_CSV.zip"
    sources.append({"file": str(path), "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest()})
    with zipfile.ZipFile(path) as archive:
        member = next(n for n in archive.namelist() if n.lower().endswith((".csv", ".txt")))
        reader = csv.DictReader(io.TextIOWrapper(archive.open(member), encoding="utf-8-sig"), delimiter=";")
        field = next((f for f in ("UIDENTSTLAE", "UIDENTSTLA") if f in reader.fieldnames), None)
        c = collections.Counter()
        lengths, hours, blocks, matches = (collections.Counter() for _ in range(4))
        for row in reader:
            c["all_records"] += 1
            ident = row.get(field, "") if field else ""
            if ident in ("01250509171013352025", "01240526125013102024"):
                examples.append({"year": year, "id": ident, "USTUNDE": row["USTUNDE"], "UMONAT": row["UMONAT"], "UWOCHENTAG": row["UWOCHENTAG"]})
            if int(row["ULAND"]) != 9:
                continue
            c["bavaria_records"] += 1
            if not field:
                continue
            lengths[len(ident)] += 1
            if len(ident) < 12 or not ident[:12].isdigit():
                c["unparseable_prefix"] += 1
                continue
            c["numeric_prefix"] += 1
            c["state_match"] += int(int(ident[:2]) == int(row["ULAND"]))
            flags = []
            try:
                date = datetime.date(2000 + int(ident[2:4]), int(ident[4:6]), int(ident[6:8]))
                c["valid_date"] += 1
                checks = {"year": date.year == int(row["UJAHR"]), "month": date.month == int(row["UMONAT"]),
                          "weekday": (date.weekday() + 1) % 7 + 1 == int(row["UWOCHENTAG"])}
                for key, ok in checks.items():
                    c[key + "_match"] += int(ok)
                    if not ok:
                        flags.append(key)
                c["date_all_match"] += int(all(checks.values()))
            except ValueError:
                flags.append("invalid_date")
            if flags:
                anomalies.append({"year": year, "id": ident, "flags": ",".join(flags),
                                  **{k: row[k] for k in ("UJAHR", "UMONAT", "UWOCHENTAG", "USTUNDE")}})
            h, m, published = int(ident[8:10]), int(ident[10:12]), int(row["USTUNDE"])
            hours[ident[8:10]] += 1
            blocks[ident[8:12]] += 1
            c["valid_hhmm"] += int(h < 24 and m < 60)
            c["hour_match"] += int(h == published)
            c["valid_hhmm_hour_match"] += int(h < 24 and m < 60 and h == published)
            for start in range(8, len(ident)-1):
                if ident[start:start+2].isdigit():
                    matches[start+1] += int(int(ident[start:start+2]) == published)
        summaries.append({"year": year, "identifier_field": field, **dict(c), "id_lengths": dict(lengths),
                          "proposed_hours": dict(hours), "top_proposed_hhmm": blocks.most_common(10)})
        offsets.append({"year": year, "suffix_hour_matches_by_one_based_start": dict(matches)})
report = {"method": "ULAND=09, date positions 3-8 as YYMMDD; proposed HHMM at 9-12. Weekday Sunday=1. No crash records modified.",
          "sources": sources, "years": summaries, "provided_examples": examples, "suffix_checks": offsets}
(out / "identifier_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
with (out / "date_conflicts.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["year","id","flags","UJAHR","UMONAT","UWOCHENTAG","USTUNDE"])
    w.writeheader()
    w.writerows(anomalies)
lines = ["# Bavarian Unfallatlas identifier audit", "", report["method"], "",
         "| Year | Bavaria records | ID field | Date fields all agree | Proposed hour agrees | Valid proposed HHMM |",
         "|---|---:|---|---:|---:|---:|"]
for r in summaries:
    n = r["bavaria_records"]
    def fmt(k):
        return f'{r.get(k,0):,} ({100*r.get(k,0)/n:.3f}%)' if r.get("identifier_field") else "N/A"
    lines.append(f'| {r["year"]} | {n:,} | {r["identifier_field"] or "absent"} | {fmt("date_all_match")} | {fmt("hour_match")} | {fmt("valid_hhmm")} |')
lines += ["", "## Conclusion", "",
          "The proposed HHMM decoding at positions 9-12 is contradicted by the published hour. A numerically valid HHMM is not evidence of a timestamp. No crash-minute rounding distribution can be inferred from these digits.",
          "Date agreement tests validate consistency with published year/month/weekday, not independent accuracy of the police-recorded date. Flag conflicting records; no automatic correction is justified.",
          "Testing adjacent two-digit suffix positions is diagnostic only, not evidence that a different encoding cannot exist. A state-specific format description is required.",
          "", "## Provided examples", ""]
for e in examples:
    lines.append(f'- {e["id"]}: USTUNDE={e["USTUNDE"]}, proposed HHMM={e["id"][8:12]}.')
(out / "README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
print("\n".join(lines))

