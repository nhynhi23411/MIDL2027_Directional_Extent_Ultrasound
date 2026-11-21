import re
BS = chr(92)
s = open("E:/MIDL_2/paper/paper_src.tex", encoding="utf8").read()
blocks = []
for lab in ("tab:data", "tab:paired"):
    pat = re.escape(BS + "begin{table}[htbp]\n" + BS + "floatconts\n  {" + lab + "}") + r".*?" + re.escape(BS + "end{table}\n\n")
    m = re.search(pat, s, re.S); blocks.append(m.group(0)); s = s.replace(m.group(0), "")
mark = BS + "section{Second-study tables}" + BS + "label{app:v2}"
s = s.replace(mark, BS + "section{Supplementary tables}" + BS + "label{app:sup}\n" + "".join(blocks) + mark)
for t in ("data", "paired"):
    s = s.replace("(" + BS + "tableref{tab:" + t + "})", "(Appendix~" + BS + "ref{app:sup}, " + BS + "tableref{tab:" + t + "})")
s = s.replace("width=0.9" + BS + "linewidth]{figures/overlap", "width=0.78" + BS + "linewidth]{figures/overlap")
open("E:/MIDL_2/paper/paper_src.tex", "w", encoding="utf8").write(s)
