import re,json,urllib.request,urllib.parse
b=open('E:/MIDL_2/paper/refs.bib',encoding='utf8').read()
for e in re.split(r'\n@',b)[1:]:
    key=e.split('{')[1].split(',')[0]; t=re.search(r'title=\{(.*)\},?\n',e)
    if not t: continue
    title=re.sub(r'[{}]','',t.group(1)); au=re.search(r'author=\{(.*?)\}',e,re.S).group(1).split(' and ')[0].split(',')[0]
    try:
        u='https://api.crossref.org/works?rows=1&query.bibliographic='+urllib.parse.quote(title+' '+au)
        r=json.load(urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'refcheck (generic client)'}),timeout=20))['message']['items'][0]
        print(key,'|',title[:50],'->',r.get('title',[''])[0][:50],'|',r.get('container-title',[''])[0][:30],r.get('volume'),r.get('page'),r.get('issued',{}).get('date-parts',[[None]])[0][0])
    except Exception as ex: print(key,'ERR',ex)
