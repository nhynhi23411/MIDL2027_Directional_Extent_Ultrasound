BS=chr(92)
s=open("E:/MIDL_2/paper/paper_src.tex",encoding="utf8").read()
a=s.index(BS+"midlauthor{"); b=s.index(BS+"begin{document}")
new=(BS+"midlauthor{"+BS+"Name{Nhi-Yen Nguyen-Huynh"+BS+"nametag{$^{1}$}} "+BS+"Email{nhinhy23411@st.uel.edu.vn}"+BS+BS+"\n"
 +BS+"addr $^{1}$ Vietnam National University, Ho Chi Minh City -- University of Economics and Law "+BS+"AND\n"
 +BS+"Name{Nguyen Quoc Khanh Le"+BS+"nametag{$^{2}$}} "+BS+"Email{khanhlee@tmu.edu.tw}"+BS+BS+"\n"+BS+"addr $^{2}$ Taipei Medical University}\n\n")
open("E:/MIDL_2/paper/paper_src.tex","w",encoding="utf8").write(s[:a]+new+s[b:])
