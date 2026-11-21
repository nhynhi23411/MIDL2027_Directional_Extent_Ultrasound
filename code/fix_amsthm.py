BS=chr(92)
s=open("E:/MIDL_2/paper/paper_src.tex",encoding="utf8").read()
s=s.replace(BS+"usepackage{amsmath,amssymb,amsthm}\n"+BS+"newtheorem{proposition}{Proposition}\n"+BS+"theoremstyle{remark}"+BS+"newtheorem{remark}{Remark}\n",BS+"usepackage{amsmath,amssymb}\n")
open("E:/MIDL_2/paper/paper_src.tex","w",encoding="utf8").write(s)
