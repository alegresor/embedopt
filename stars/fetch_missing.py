# fetch_missing.py
import starfile, pathlib
here = pathlib.Path(__file__).resolve().parent
root = here / "10810"
p = starfile.read(here / "10810_shiny.star")["particles"]
f = p.rlnImageName.str.split("@").str[1].str.extract(r"(Movies/.*)")[0].unique()
base = "https://ftp.ebi.ac.uk/empiar/world_availability/10810/data/job177/"
miss = [base + x for x in f if not (root / x).exists()]
(here / "missing_urls.txt").write_text("\n".join(miss) + "\n")
print(len(miss), "files to fetch")