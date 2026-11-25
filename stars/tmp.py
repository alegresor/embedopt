import starfile, mrcfile, pathlib
here = pathlib.Path(__file__).resolve().parent              # .../stars
root = here / "10810"
p = starfile.read(here / "10810_shiny.star")["particles"]
f = p.rlnImageName.str.split("@").str[1].str.extract(r"(Movies/.*)")[0]
u = f.unique()
print(len(p), "particles;", len(u), "stacks;", sum(not (root / x).exists() for x in u), "missing")
print(p.rlnClassNumber.value_counts())

def nz(x):
    with mrcfile.open(root / x, header_only=True, permissive=True) as m:
        return int(m.header.nz)
ok = f.map(lambda x: (root / x).exists())
q = p[ok.values].reset_index(drop=True)
print(len(q), "particles with images;", f[ok].nunique(), "stacks")
print(f.str.extract(r"Movies/([^/]+)/")[0][ok].value_counts())        # sessions kept
print(f.str.extract(r"Movies/([^/]+)/")[0][~ok].value_counts())       # sessions missing
print("images on disk:", sum(nz(x) for x in f[ok].unique()), "(should equal", len(q), ")")
q.to_csv(here / "10810_particles_present.csv", index=False)

bad = [x for x in f[ok].unique() if (root/x).stat().st_size < 1024 + nz(x)*226*226*4]
print(len(bad), "truncated")