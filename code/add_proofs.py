BS=chr(92)
s=open("E:/MIDL_2/paper/paper_src.tex",encoding="utf8").read()
s=s.replace(BS+"section{Results}",BS+"section{Results}"+BS+"label{sec:results}",1)
P=open("E:/MIDL_2/paper/proofs.tex",encoding="utf8").read()
mark=BS+"section{Data audit}"
s=s.replace(mark,P+"\n"+mark,1)
open("E:/MIDL_2/paper/paper_src.tex","w",encoding="utf8").write(s)
