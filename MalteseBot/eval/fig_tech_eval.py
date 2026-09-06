import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

s = json.loads((Path("eval/summary.json")).read_text(encoding="utf-8"))
RAG="#1a7a3e"; BASE="#a02020"
groups=["Overall","English","Maltese"]
keys=[("overall",None),("by_language","en"),("by_language","mt")]
def get(metric):
    rag=[]; base=[]
    for sec,k in keys:
        node = s[sec] if k is None else s[sec][k]
        rag.append(node["RAG"][metric]); base.append(node["Baseline"][metric])
    return rag,base
fig,axes=plt.subplots(1,2,figsize=(9,4))
x=range(len(groups)); w=0.36
for ax,metric,title,fmt in [
    (axes[0],"keyword_precision","Fact precision","{:.2f}"),
    (axes[1],"hallucination_rate","Hallucination rate","{:.0%}")]:
    rag,base=get(metric)
    b1=ax.bar([i-w/2 for i in x],rag,w,label="RAG",color=RAG)
    b2=ax.bar([i+w/2 for i in x],base,w,label="Baseline",color=BASE)
    ax.set_xticks(list(x)); ax.set_xticklabels(groups)
    ax.set_ylim(0,1); ax.set_title(title)
    ax.set_ylabel(title)
    for bars in (b1,b2):
        for r in bars:
            h=r.get_height()
            ax.text(r.get_x()+r.get_width()/2,h+0.02,fmt.format(h),ha="center",va="bottom",fontsize=8,fontweight="bold")
    if metric=="hallucination_rate":
        ax.legend(loc="upper left",fontsize=9)
fig.tight_layout()
fig.savefig("eval/fig_tech_eval.png",dpi=150)
print("wrote eval/fig_tech_eval.png")
