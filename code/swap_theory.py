BS=chr(92)
s=open("E:/MIDL_2/paper/paper_src.tex",encoding="utf8").read()
a=s.index(BS+"subsection{Properties of the loss}"); b=s.index(BS+"subsection{Measurement protocol}")
s=s[:a]+open("E:/MIDL_2/paper/theory_main.tex",encoding="utf8").read()+s[b:]
open("E:/MIDL_2/paper/paper_src.tex","w",encoding="utf8").write(s)
