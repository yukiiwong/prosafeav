"""The fitted EVT parameters at the end of each run: the EVT diagnostics table.

Takes the last successful refit in each run's log, which is the model the policy
finished under, rather than the first one it happened to produce.
"""
import glob
import os
import re
import sys

PAT = re.compile(
    r"\[EVT\] update #(\d+) n=(\d+) \| "
    r"u_ttc=(-?[\d.]+) xi=(-?[\d.]+) sigma=([\d.]+) n_exc=(\d+) \| "
    r"u_drac=(-?[\d.]+) xi=(-?[\d.]+) sigma=([\d.]+) n_exc=(\d+) \| "
    r"(\w+)=([\d.]+) P_crash=([\d.e+-]+)"
)

logdir = sys.argv[1] if len(sys.argv) > 1 else "logdir"
rows = []
for d in sorted(glob.glob(os.path.join(logdir, "*_s[0-9]"))):
    log = os.path.join(d, "run.log")
    if not os.path.exists(log):
        continue
    last = None
    for line in open(log, errors="ignore"):
        m = PAT.search(line)
        if m:
            last = m
    if last:
        rows.append((os.path.basename(d), last))

if not rows:
    print("no completed EVT fits found")
    sys.exit(0)

print(f"{'run':26s} {'fits':>4s} {'n':>6s} "
      f"{'TTC thr':>8s} {'xi':>7s} {'sigma':>7s} {'n_exc':>6s} "
      f"{'DRAC thr':>9s} {'xi':>7s} {'sigma':>7s} {'n_exc':>6s} "
      f"{'alpha':>7s} {'P_crash':>11s}")
print("-" * 122)
for name, m in rows:
    (upd, n, u_t, xi_t, sg_t, ne_t, u_d, xi_d, sg_d, ne_d, fam, par, pc) = m.groups()
    print(f"{name:26s} {int(upd):4d} {int(n):6d} "
          f"{-float(u_t):8.3f} {float(xi_t):+7.3f} {float(sg_t):7.3f} {int(ne_t):6d} "
          f"{float(u_d):9.3f} {float(xi_d):+7.3f} {float(sg_d):7.3f} {int(ne_d):6d} "
          f"{float(par):7.4f} {float(pc):11.3e}")
print("\nTTC threshold shown as a positive time; the model works on its negation.")
