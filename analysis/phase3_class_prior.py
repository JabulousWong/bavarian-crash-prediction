"""Extract actual official counts and keep local class-prior assumptions explicit."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
from phase3_xlsx_read import read_sheets

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/results/phase3/class_prior"
GENESIS = "https://genesis.destatis.de/datenbank/online/statistic/46241/table/46241-0020"
POLICE_2024 = "https://www.stmi.bayern.de/assets/stmi/med/reden/pressetermin_m_anl._verkehrsunfallstatistik_2024_am_20.02.2025_in_m%C3%BCnchen.pdf"
POLICE_2025 = "https://www.stmi.bayern.de/media/presse-und-medien/reden/schutz-und-sicherheit/260225-rede-herrmann-verkehrsunfallstatistik-2025.pdf"
MOTORWAY = {2024: {"total": 28004, "other_intoxicant_pdo": 138, "remaining_pdo": 22952},
            2025: {"total": 27321, "other_intoxicant_pdo": 122, "remaining_pdo": 22456}}


def number(value):
    if value in ("-", "–"):
        return 0
    if isinstance(value, (float, int)):
        return int(value)
    raise ValueError(f"Not a count: {value!r}")


def fingerprint(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records, sources, annual, monthly_district = [], [], {}, []
    for year in (2024, 2025):
        month_path = ROOT / f"h1101c/h1101c_{year}12.xlsx"
        annual_path = ROOT / f"h1100d/h1100d_{year}00.xlsx"
        monthly = read_sheets(month_path)
        detailed = read_sheets(annual_path)
        sources.extend([fingerprint(month_path), fingerprint(annual_path)])
        state = {}
        mapping = {"total": "J14", "injury": "J15", "pdo_total": "J17",
                   "serious_pdo": "J20", "other_intoxicant_pdo": "J24", "remaining_pdo": "J27"}
        for category, cell in mapping.items():
            count = number(monthly["S.7"][cell])
            state[category] = count
            records.append({"year": year, "geography": "Bavaria", "road_scope": "all roads",
                            "category": category, "count": count, "source": str(month_path),
                            "sheet": "S.7", "cell": cell, "status": "published December cumulative figures; source may be provisional"})
        assert state["total"] == state["injury"] + state["pdo_total"]
        assert state["pdo_total"] == sum(state[c] for c in ("serious_pdo", "other_intoxicant_pdo", "remaining_pdo"))
        motorway = dict(MOTORWAY[year])
        for category, cell in (("injury", "C34"), ("serious_pdo", "K34")):
            motorway[category] = number(detailed["21"][cell])
            records.append({"year": year, "geography": "Bavaria", "road_scope": "motorways",
                            "category": category, "count": motorway[category], "source": str(annual_path),
                            "sheet": "21", "cell": cell, "status": "annual detailed table"})
        assert motorway["injury"] == number(monthly["S.15"]["D39"])
        assert motorway["total"] == sum(motorway[c] for c in ("injury", "serious_pdo", "other_intoxicant_pdo", "remaining_pdo"))
        motorway["pdo_total"] = motorway["total"] - motorway["injury"]
        motorway["all_police_to_injury_ratio"] = motorway["total"] / motorway["injury"]
        motorway["injury_plus_serious_to_injury_ratio"] = 1 + motorway["serious_pdo"] / motorway["injury"]
        for category in ("total", "other_intoxicant_pdo", "remaining_pdo"):
            records.append({"year": year, "geography": "Bavaria", "road_scope": "motorways",
                            "category": category, "count": motorway[category], "source": GENESIS,
                            "sheet": "46241-0020", "cell": f"Bayern/{year}/{category}/auf Autobahnen",
                            "status": "actual official count transcribed from indexed table; total cross-checked in Interior Ministry publication"})
        annual[str(year)] = {"bavaria_all_roads": state, "bavaria_motorways": motorway}
        # Export available district motorways for finer geography, preserving the
        # fact that these monthly detailed totals exclude remaining/minor PDO.
        for month in range(1, 13):
            path = ROOT / f"h1101c/h1101c_{year}{month:02d}.xlsx"
            if not path.exists():
                continue
            sources.append(fingerprint(path))
            sheets = read_sheets(path)
            extracted_bavaria_injury = None
            for name in (f"S.{page}" for page in range(9, 15)):
                cells = sheets[name]
                for address, value in cells.items():
                    if address.startswith("B") and isinstance(value, str) and value.strip() == "auf Autobahnen":
                        row = int(address[1:])
                        region = cells.get(f"B{row-3}")
                        if not region or f"F{row}" not in cells:
                            continue
                        total, injury, pdo = (number(cells[f"{c}{row}"]) for c in "FGH")
                        assert total == injury + pdo
                        monthly_district.append({"year": year, "month": month, "geography": region,
                            "injury": injury, "serious_pdo_plus_intoxicant_pdo": pdo,
                            "detailed_table_total_excluding_remaining_pdo": total,
                            "source": str(path), "sheet": name, "cells": f"F{row},G{row},H{row}"})
                        if region == "Bayern":
                            extracted_bavaria_injury = injury
            assert extracted_bavaria_injury == number(sheets["S.15"]["D18"])
    for name, data in (("actual_official_counts.csv", records), ("district_motorway_monthly.csv", monthly_district)):
        with (OUT / name).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    population = json.loads((ROOT / "analysis/results/phase3/population/population_summary.json").read_text())
    training = population["years"]["2024"]
    # Only 2024 aggregate statistics enter training-prior scenarios.
    ratios = annual["2024"]["bavaria_motorways"]
    scenarios = {}
    for threshold, counts in training["thresholds"].items():
        n = training["population_edge_hours"]
        k = counts["official_positive_edge_hours"]
        injury_records = counts["official_crash_records"]
        scenarios[threshold] = {"population_edge_hours": n, "observed_injury_edge_hours": k,
            "observed_injury_crash_records": injury_records,
            "observed_injury_edge_hour_fraction": k/n, "transport_scenarios": {}}
        for label, multiplier in (("injury_plus_serious_pdo", ratios["injury_plus_serious_to_injury_ratio"]),
                                  ("all_police_recorded_crashes", ratios["all_police_to_injury_ratio"])):
            additional = injury_records * (multiplier - 1)
            scenarios[threshold]["transport_scenarios"][label] = {
                "actual_2024_motorway_count_ratio": multiplier,
                "implied_additional_local_crash_count_under_transport": additional,
                "conditional_occupancy_lower": k/n,
                "conditional_occupancy_upper": min(1.0, (k + additional)/n),
                "meaning": "conditional occupancy range if the statewide event ratio transfers exactly, official injury records are complete and spatially valid, and the implied added events lie inside this same frame. No collision/multiplicity distribution is assumed; these are not statistical confidence limits or identified local priors.",
            }
    release_audit = {}
    for year in (2024, 2025):
        first_release_sum = sum(row["injury"] for row in monthly_district
                                if row["year"] == year and row["geography"] == "Bayern")
        annual_injury = annual[str(year)]["bavaria_motorways"]["injury"]
        release_audit[str(year)] = {"sum_of_monthly_release_motorway_injury_counts": first_release_sum,
            "december_cumulative_and_annual_motorway_injury_count": annual_injury,
            "annual_minus_monthly_release_sum": annual_injury-first_release_sum,
            "use": "Use annual cumulative/revised counts for annual totals; do not silently sum incompatible monthly release vintages."}
    output = {"actual_counts": annual, "sources": sources,
        "release_vintage_audit": release_audit,
        "web_sources": [{"url": GENESIS, "verified_on": "2026-09-29"},
                        {"url": POLICE_2024, "supports": "2024 motorway total 28004"},
                        {"url": POLICE_2025, "supports": "2025 motorway total 27321 and 2024 comparator 28004"}],
        "training_prior_constraints_2024": scenarios,
        "identified_local_all_crash_class_prior": None,
        "not_identified_reason": "Actual aggregate counts are known, but they do not locate all PDO events inside these corridor edge-hours or identify incident multiplicity, capture completeness, or representative positive ascertainment.",
        "target_choice": "Official injury endpoint and all-police-recorded-crash endpoint are different targets; Waze may also include events never recorded by police.",
        "year_isolation": "2025 counts are descriptive checks only; 2024 ratios are used for training scenarios.",
        "regional_table_caveat": "H1101C district table F=G+H; H includes serious PDO i.e.S. and other intoxicant PDO. Remaining PDO is excluded. Monthly sums may differ from final annual figures after revisions.",
        "source_completeness": {"unfallatlas_bavaria_2024": 46446, "official_statistical_injury_2024": 49374,
                                "unfallatlas_bavaria_2025": 46938, "official_statistical_injury_2025": 49925,
                                "interpretation": "Counts differ; no automatic local correction without auditing source coverage."},
        "PU_identification": "Knowing a prior does not make injury-only positives representative of all crashes. Standard SCAR/nnPU assumptions require separate justification.",
    }
    (OUT / "class_prior_constraints.json").write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Official crash statistics and class-prior constraints", "",
        "The h1100d/h1101c workbooks provide official crash counts, so these numbers are neither unknown nor subjective assumptions. What has to be assumed is the mapping from the statistical area to the edge-hours of the study corridor.", "",
        "| Scope | 2024 | 2025 |", "|---|---:|---:|"]
    for scope, label in (("bavaria_all_roads", "Bavaria, all roads"), ("bavaria_motorways", "Bavaria, motorways")):
        for category, text in (("total", "all crashes"), ("injury", "injury crashes"), ("pdo_total", "property-damage-only crashes")):
            lines.append(f"| {label}: {text} | {annual['2024'][scope][category]:,} | {annual['2025'][scope][category]:,} |")
    lines += ["", "Cells: h1101c December files S.7!J14/J15/J17 give annual all-road totals / injury / property damage; h1100d annual files 21!C34/K34 give motorway injury / serious property damage. All motorway crashes and the remaining property-damage crashes come from GENESIS 46241-0020, and the crash totals were cross-checked against the annual release of the Bavarian Ministry of the Interior. The CSV records the source of every item.", "",
        "District detail: motorway detail for districts, cities, administrative regions and the state was extracted for 24 months (it includes higher-level totals and must not be summed as a whole). Column F, insgesamt, covers only injury + serious property damage / specified property damage such as drink-driving and excludes the remaining property-damage crashes (übrige Sachschadensunfälle).", "",
        "Release check: the monthly release files sum to 2,771 motorway injury crashes in 2024, while the December cumulative and annual tables give 3,236; for 2025 the figures are 2,825 and 3,199. Annual statistics use the cumulative/revised figures; the sum of earlier monthly files cannot replace the actual annual total. The state values of every monthly district table agree with the road-type table of the same file.", "",
        "2024 motorway all crashes / injury crashes = 28,004/3,236 = 8.653894; (injury + serious property damage) / injury = (3,236+1,678)/3,236 = 1.518541. The actual ratio and its transferability to the local corridor are two separate questions.", "",
        "The class prior is P(at least one target crash in a random edge-hour), not a crash-category share and not the 20% of the 1:4 sample. Geographic scope, time coverage and event definition must be aligned first, then crash counts converted into hours with at least one crash. Several crashes in the same hour must not be counted twice in the numerator.", "",
        "The JSON gives the observed injury fraction over the 2024 training-population denominator and occupancy ranges under explicit regional-transfer/completeness assumptions. The ranges are not confidence intervals and do not identify the true corridor prior. The 2025 statistics play no part in choosing the 2024 prior. Even with exact totals, injury crashes as positives for all crashes may violate SCAR.", "",
        f"Official sources: [GENESIS]({GENESIS}); [Ministry of the Interior release 2024]({POLICE_2024}); [Ministry of the Interior release 2025]({POLICE_2025}).",
        "PU references: [Elkan & Noto (2008)](https://cseweb.ucsd.edu/~elkan/posonly.pdf); [Kiryo et al. (2017)](https://papers.nips.cc/paper/6765-positive-unlabeled-learning-with-non-negative-risk-estimator.pdf)."]
    (OUT / "REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"actual_counts": annual, "district_monthly_rows": len(monthly_district),
                      "training_prior_constraints_2024": scenarios}, indent=2))


if __name__ == "__main__":
    main()
