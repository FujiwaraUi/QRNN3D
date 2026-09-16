from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ICVL_PATHS = (
    "../ICVL_test_complex.txt",
    "../ICVL_test_gauss.txt",
    "../ICVL_train.txt",
)
MAT_PATH = "mat_ls_result.txt"
RESULT_PATH = BASE_DIR / "check_mat_data_result.txt"


def read_names(path: Path) -> tuple[list[str], set[str]]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return lines, set(lines)


icvl_paths = tuple(BASE_DIR / path for path in ICVL_PATHS)
if not icvl_paths:
    raise FileNotFoundError("ICVL_PATHS must contain at least one path")

icvl_names: set[str] = set()
duplicate_lines: list[str] = []
for path in icvl_paths:
    lines, names = read_names(path)
    icvl_names.update(names)
    duplicates = sorted(name for name in names if lines.count(name) > 1)
    duplicate_lines.extend(f"{path.name}: {name}" for name in duplicates)

mat_path = BASE_DIR / MAT_PATH
mat_lines, mat_names = read_names(mat_path)
mat_duplicates = sorted(name for name in mat_names if mat_lines.count(name) > 1)
duplicate_lines.extend(f"{mat_path.name}: {name}" for name in mat_duplicates)

missing_from_mat = sorted(icvl_names - mat_names)
extra_in_mat = sorted(mat_names - icvl_names)

report: list[str] = []
report.append(f"ICVL_t*.txt files: {len(icvl_paths)}")
report.append(f"ICVL unique entries: {len(icvl_names)}")
report.append(f"mat_ls_result.txt unique entries: {len(mat_names)}")
report.append("")
report.append("Entries listed in ICVL_t*.txt but missing from mat_ls_result.txt:")
report.extend(missing_from_mat or ["(none)"])
report.append("")
report.append("Entries listed in mat_ls_result.txt but missing from ICVL_t*.txt:")
report.extend(extra_in_mat or ["(none)"])
report.append("")
report.append("Duplicate entries:")
report.extend(duplicate_lines or ["(none)"])

RESULT_PATH.write_text("\n".join(report) + "\n")