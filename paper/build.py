import subprocess, os
os.chdir("E:/MIDL_2/paper")
s = open("paper_src.tex", encoding="utf8").read()
for n in ("main", "paired", "main_v2", "paired_v2", "v3", "busi_neg", "ablK"):
    s = s.replace(r"\relax\input{tables/%s.tex}" % n, open(f"tables/{n}.tex", encoding="utf8").read().strip())
open("paper.tex", "w", encoding="utf8").write(s)
env = dict(os.environ); env["PATH"] = "C:/Users/pc/AppData/Local/Programs/MiKTeX/miktex/bin/x64" + os.pathsep + env["PATH"]
BIN = "C:/Users/pc/AppData/Local/Programs/MiKTeX/miktex/bin/x64/"
for cmd in ([BIN + "pdflatex.exe"], [BIN + "bibtex.exe"], [BIN + "pdflatex.exe"], [BIN + "pdflatex.exe"]):
    a = cmd + (["paper"] if "bibtex" in cmd[0] else ["-interaction=nonstopmode", "paper.tex"])
    r = subprocess.run(a, capture_output=True, text=True, env=env)
log = open("paper.log", encoding="utf8", errors="ignore").read()
print("errors:", log.count("\n!"), "| undefined refs/cites:", log.count("undefined"), "|", [l for l in log.split("\n") if "Output written" in l])
