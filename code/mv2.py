import re
BS=chr(92)
s=open("E:/MIDL_2/paper/paper_src.tex",encoding="utf8").read()
blocks=[]
for env,lab in (("figure","fig:qual"),("figure","fig:overlap"),("table","tab:v3")):
    pat=re.escape(BS+"begin{"+env+"}[htbp]\n"+BS+"floatconts\n  {"+lab+"}")+r".*?"+re.escape(BS+"end{"+env+"}\n\n")
    m=re.search(pat,s,re.S); blocks.append(m.group(0)); s=s.replace(m.group(0),"")
mark=BS+"section{Second-study tables}"+BS+"label{app:v2}"
s=s.replace(mark,"".join(blocks)+mark,1)
for lab,cmd in (("fig:qual","figureref"),("fig:overlap","figureref"),("tab:v3","tableref")):
    s=s.replace(BS+cmd+"{"+lab+"}","Appendix~"+BS+"ref{app:sup}, "+BS+cmd+"{"+lab+"}")
open("E:/MIDL_2/paper/paper_src.tex","w",encoding="utf8").write(s)
