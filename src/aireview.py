#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "backoff>=2.2,<3",
#   "numpy>=2,<3",
#   "openai>=2,<3",
#   "pymupdf>=1.26,<2",
#   "pymupdf4llm>=0.2,<1",
#   "pypdf>=5,<7",
# ]
# ///
"""Standalone implementation of The AI Scientist's Automated Reviewer."""

from __future__ import annotations

import argparse
import base64
import bz2
import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import backoff
import numpy as np
import openai
import pymupdf
from openai import OpenAI
from pypdf import PdfReader

with contextlib.redirect_stdout(io.StringIO()):
    import pymupdf4llm

MODEL = "gpt-5.6-terra"
REVIEWER_COUNT = 5
REVIEW_ATTEMPTS = 2
MAX_COMPLETION_TOKENS = 16384
MIN_PAPER_CHARS = 100
REVIEW_RETRY_CORRECTION = (
    "\n\nRetry correction: The previous response did not satisfy the required review JSON "
    "contract. Return one complete strict JSON review with every required field and the "
    "specified field types and score ranges."
)

REVIEWER_SYSTEM_PROMPT = (
    "You are an AI researcher who is reviewing a paper that was submitted to a prestigious ML venue."
    "Be critical and cautious in your decision."
)

TEMPLATE_INSTRUCTIONS = """
Respond in the following format:

THOUGHT:
<THOUGHT>

REVIEW JSON:
```json
<JSON>
```

In <THOUGHT>, first briefly discuss your intuitions and reasoning for the evaluation.
Detail your high-level arguments, necessary choices and desired outcomes of the review.
Do not make generic comments here, but be specific to your current paper.
Treat this as the note-taking phase of your review.

In <JSON>, provide the review in JSON format with the following fields in the order:
- "Summary": A summary of the paper content and its contributions.
- "Strengths": A list of strengths of the paper.
- "Weaknesses": A list of weaknesses of the paper.
- "Originality": A rating from 1 to 4 (low, medium, high, very high).
- "Quality": A rating from 1 to 4 (low, medium, high, very high).
- "Clarity": A rating from 1 to 4 (low, medium, high, very high).
- "Significance": A rating from 1 to 4 (low, medium, high, very high).
- "Questions": A set of clarifying questions to be answered by the paper authors.
- "Limitations": A set of limitations and potential negative societal impacts of the work.
- "Ethical Concerns": A boolean value indicating whether there are ethical concerns.
- "Soundness": A rating from 1 to 4 (poor, fair, good, excellent).
- "Presentation": A rating from 1 to 4 (poor, fair, good, excellent).
- "Contribution": A rating from 1 to 4 (poor, fair, good, excellent).
- "Overall": A rating from 1 to 10 (very strong reject to award quality).
- "Confidence": A rating from 1 to 5 (low, medium, high, very high, absolute).
- "Decision": A decision that has to be one of the following: Accept, Reject.

For the "Decision" field, don't use Weak Accept, Borderline Accept, Borderline Reject, or Strong Reject. Instead, only use Accept or Reject.
This JSON will be automatically parsed, so ensure the format is precise.
"""

NEURIPS_FORM = (
    """
## Review Form
Below is a description of the questions you will be asked on the review form for each paper and some guidelines on what to consider when answering these questions.
When writing your review, please keep in mind that after decisions have been made, reviews and meta-reviews of accepted papers and opted-in rejected papers will be made public.

1. Summary: Briefly summarize the paper and its contributions. This is not the place to critique the paper; the authors should generally agree with a well-written summary.
  - Strengths and Weaknesses: Please provide a thorough assessment of the strengths and weaknesses of the paper, touching on each of the following dimensions:
  - Originality: Are the tasks or methods new? Is the work a novel combination of well-known techniques? (This can be valuable!) Is it clear how this work differs from previous contributions? Is related work adequately cited
  - Quality: Is the submission technically sound? Are claims well supported (e.g., by theoretical analysis or experimental results)? Are the methods used appropriate? Is this a complete piece of work or work in progress? Are the authors careful and honest about evaluating both the strengths and weaknesses of their work
  - Clarity: Is the submission clearly written? Is it well organized? (If not, please make constructive suggestions for improving its clarity.) Does it adequately inform the reader? (Note that a superbly written paper provides enough information for an expert reader to reproduce its results.)
  - Significance: Are the results important? Are others (researchers or practitioners) likely to use the ideas or build on them? Does the submission address a difficult task in a better way than previous work? Does it advance the state of the art in a demonstrable way? Does it provide unique data, unique conclusions about existing data, or a unique theoretical or experimental approach?

2. Questions: Please list up and carefully describe any questions and suggestions for the authors. Think of the things where a response from the author can change your opinion, clarify a confusion or address a limitation. This can be very important for a productive rebuttal and discussion phase with the authors.

3. Limitations: Have the authors adequately addressed the limitations and potential negative societal impact of their work? If not, please include constructive suggestions for improvement.
In general, authors should be rewarded rather than punished for being up front about the limitations of their work and any potential negative societal impact. You are encouraged to think through whether any critical points are missing and provide these as feedback for the authors.

4. Ethical concerns: If there are ethical issues with this paper, please flag the paper for an ethics review. For guidance on when this is appropriate, please review the NeurIPS ethics guidelines.

5. Soundness: Please assign the paper a numerical rating on the following scale to indicate the soundness of the technical claims, experimental and research methodology and on whether the central claims of the paper are adequately supported with evidence.
  4: excellent
  3: good
  2: fair
  1: poor

6. Presentation: Please assign the paper a numerical rating on the following scale to indicate the quality of the presentation. This should take into account the writing style and clarity, as well as contextualization relative to prior work.
  4: excellent
  3: good
  2: fair
  1: poor

7. Contribution: Please assign the paper a numerical rating on the following scale to indicate the quality of the overall contribution this paper makes to the research area being studied. Are the questions being asked important? Does the paper bring a significant originality of ideas and/or execution? Are the results valuable to share with the broader NeurIPS community.
  4: excellent
  3: good
  2: fair
  1: poor

8. Overall: Please provide an "overall score" for this submission. Choices:
  10: Award quality: Technically flawless paper with groundbreaking impact on one or more areas of AI, with exceptionally strong evaluation, reproducibility, and resources, and no unaddressed ethical considerations.
  9: Very Strong Accept: Technically flawless paper with groundbreaking impact on at least one area of AI and excellent impact on multiple areas of AI, with flawless evaluation, resources, and reproducibility, and no unaddressed ethical considerations.
  8: Strong Accept: Technically strong paper with, with novel ideas, excellent impact on at least one area of AI or high-to-excellent impact on multiple areas of AI, with excellent evaluation, resources, and reproducibility, and no unaddressed ethical considerations.
  7: Accept: Technically solid paper, with high impact on at least one sub-area of AI or moderate-to-high impact on more than one area of AI, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations.
  6: Weak Accept: Technically solid, moderate-to-high impact paper, with no major concerns with respect to evaluation, resources, reproducibility, ethical considerations.
  5: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject, e.g., limited evaluation. Please use sparingly.
  4: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.
  3: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility and incompletely addressed ethical considerations.
  2: Strong Reject: For instance, a paper with major technical flaws, and/or poor evaluation, limited impact, poor reproducibility and mostly unaddressed ethical considerations.
  1: Very Strong Reject: For instance, a paper with trivial results or unaddressed ethical considerations

9. Confidence:  Please provide a "confidence score" for your assessment of this submission to indicate how confident you are in your evaluation. Choices:
  5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.
  4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
  3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
  2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
  1: Your assessment is an educated guess. The submission is not in your area or the submission was difficult to understand. Math/other details were not carefully checked.
"""
    + TEMPLATE_INSTRUCTIONS
)

META_REVIEWER_SYSTEM_PROMPT = """You are an Area Chair at a machine learning conference.
You are in charge of meta-reviewing a paper that was reviewed by {reviewer_count} reviewers.
Your job is to aggregate the reviews into a single meta-review in the same format.
Be critical and cautious in your decision, find consensus, and respect the opinion of all the reviewers."""

# Exact text extracted from the source reviewer's first few-shot paper:
# ai_scientist/fewshot_examples/132_automated_relational.txt. The final method
# uses this paper/review pair as its single in-context example. It is bz2-
# compressed and Base85-encoded solely to keep this implementation one file;
# this is reversible packaging, not encryption.
_FEWSHOT_PAPER_B85 = (
    "LRx4!F+o`-Q&~QCo6`UivS0mR5PSds`oI7G`oI7G{!P$YjspNb#2g8b0AO41q3^p~J!l?1*5D3+Jv^|A1f)*d4zPp*cEqK#5CBfzyX+aT0kk^u0P^S8+yi(316A&`zWcuRKKjGYZ=Lr(_r2cDk9Iq--#gNG%iQYhwjR!X>pEP$V)y}DIv+mwdpm%5d+r*1dtu*Tp0&HT(DbPS`p35&9`|?Kd}BSvYN*>F_QE^wuJ5s?*FE(6<;eRr)1})@>o0lU^WC>=yFJQuEvWQ;KHzVSCb9!$7>2RCxmW{UkA3g1Z+N@x2YK`FEP^*W?(X`dUtasvd)zx__dQPyG$5J)KmrMcJre}M8X9O#G%{gLPtz(rq<W@LP#H0(07_^Ggb4)E6MBY<Y3hEaDBh&?0g!r{003wJ0002OK|~ZxYMw~c{FF4&;%a+T)6#m6L}baKhyV=$&;SO24FsA=6H_LMl*WXo=}%0iCxtysjD~>IL7)Ht8U}+v00Jb4OhB4wCYodfJw}5O8A<9i&<z@SMxLhiKmY&(A?gI0h=Bnnni`EXgHhy}G<t@a41u5;00006fHVN2^Z*c_ko&wn9>5n?qyi*=AJvPepO3{y8aDk-ze2*FG{N8G&WKJuTtnu0P_T`)?_cA72i7XatB+QoyF@4eU7y(qV!vnZmvNO*VJ^0BA+W{kX;t+?TfejLs8WF9R|B!8XPgq8d|oS{f_5mOY7{rJeAX=XD5!%7FTxdnSd>6#`5UC5fWuZ!P57=jtO^(@Ayr8l<CSh9?<Y5(&-GoZc#C-`U_ZHci(D%DT~atrSVt+eqU{mO+<uP+#oer2jS&$>{7G?pcgPda{N1*lwVl3idZp9nk3<0IiWtDc8r1~btZ={lzG^>dhcqX(-3+I^&bQk9Sqo{8`cXO|Z&V7xV9Jmm$NAX>#5@r3W_Ywyo5i58u?(PL32c+c6)(?A{f8)6_P+go=0OiGnnntbb?-"
    "v=PTcMl1zC}29798)cg_TLV@?lj@T#R*J**}|i7(+S?iHWDJZRs+t`-P`Y0N@yo1j@%I8p3PcI1A8Eb6eJiLSaq#eT02DPZ8SY?o=CS81HAzZs~|X;zzCGcqEptip=3Vzf9zn`M^`A{iU%dktdlF8kM=`hv|l9X`+{&pU&UUb*@oP1pD-3tgSZn{;s72phj0GBAdndFP`mbUgFLMYd8FM<`9E(s<cbQ?E_9&gM1Og3GV7zHe!_pLyB9of+|x0|Fq32mtnif2F;3fyiT{fUmux3NLfmGa-l!NrUKjf>$xtF7d5fe_TEbc5jgLm}p<f+hU5lO=GM)N?|{-hi)rY5~GWOA_02t+~L!~DYZ6a4t=l?H%KgL2E?=gS}KoUol~pNWp{E{AD80M?)Q{z_SuFW0g=IH<Ug?5m3D`B?AzrbnOR6?83XaKvdZm0<?$WI;y$B2im6l|pnL)#0Ejy^$aZj`?7}Q20U-h+TlEh0g6pxu@#G#+5us})^UE%B#rU3`r2xMOqj)Css2gN#{I>jT8k5IQ`0VnQO+=`GYXC0=s~&BKDOF<}MKOxHMSggVMEjthY*XC;!bm9yueFgy><+aCUbnX9rj3#+z>{`PYyjbaaF|FIm{wXrN=RyKJJssV&{+i$1F8UU$&8#cWT_50Ert+w#40bRoj3Dg-_?VJbJH_T^L`Rks;);D*z}M+&^}(J)=|)Q2zebBdf;PGkT#DXhGTf{huy~Cx9-(C(1CT{t^9o;@dHNw>nyQZc*8kCBN#AX2r!Xmt2xSRXVcq+DzJC#HEa<n<rm@dPN-i>=TLQO8(1YoOhKUa5Q-H=W)%R`5Cp&w`-bUDzgp}2wECE1T0lT&5r-@<FTi(C&0xk>wvz5V{VAct+!cPr2CaI4>>-D?p7#AeetvHEKuk7r!8LH;`lZpMV@L?Du{WvpGPi~iOy<6CMmD`H7;bQ*SomTr)vvYiz4gAcb)%S#44S|q1F{OhCPfoj&=+9SrQkMg&f-"
    "&4@D=XyjR@A@2O9Fe>E`Z6g{7FzxkKjWe)!ZX4RKJNuH3@xr1g>TX~**QC{JczpA;BY3KaW6`bNJy_py-awWRs16p&F-Itm3Sn6QgRvSgu9gS(rLng~14jGS`Le)IG5@J;^cv&)Wr)(8npFf4>J{n$tFTria1kM`=7hMQZzpmpo`Pgb7LA^@20=DRH<fHya9CSjF>w<p9OcdlvWH(~^WN%Fn7S7p+?0-pIvY?8#RU<e&+)8e&|LECSK)}(GZOl~<06)zSW0)@j~rt)Zh*Nl{Qdc2)JJ#IKaRfMDpieRv>ZwndCDy4|Qg&a_98}!_aQq@>?4nxawi^Mh#ZlTh76ZNLwU0OF`b))Pe5nGE55*kx+sELh_SAX0Q>El+dI(oYYW<p!x;?4B)89V7i%%NKlG>IeW&HJE&B0+T2D(;8oMT1a)#$5nb?ZhW@cD)fUa!zEc*w$kZ5)%MbyET4*f@B%{y{IO>e^=eZd831F{$^c9!ahLJ$}lhQsA5>*ii?ywLu_iSJmG3xa{=V~ah{un-Pa(r`brWBZeiiMohHGhX>)R_^{bdThS%=%mi$ex<$QR<*w!B)H(obwd=A623b6bA?8Stv)%!O;MN?-2q^12ySWW1Ydjmy?Qnx5Zih=v4&7{DKyW63jl^zXZ1JNN4mza1}DaxZ+yT67+l18NA74mn7%WuG%I=bSyiLyD8DJU)ja4*6;c8NN=4R+f4qw_(Rr1?Apn<a1cu6Uhtf)=<8<L}cHFAIfH9ck+Pbw-"
    "aQHJuDIaOw~`hjFy~o37P7Om?5kowyLMx9k3io57;EzpbH`i>#DlR|Q<csSoo>heE=5eH_E5!(M?7GY(`Fh2B@?QIP{)ceM!ik%0##mI|I<r!l?)==w;<9(>N+u6+ZSI=1G|MR`;uIK=Q&&Jk!#e#e9^zV{yFYi+?xSyQiUGr>on!LIww8M}#uN%KST_Q%WoHKKK`-4_Vqp{wVyZt~f!x-QOI3vO2txIlFS<V~VoMg5AI9%3|*#VACG<}Wa!s}~>9MrJUvf2-gK@rS(MqM$>IM31S!y|~S1I=J`jra0XiO_sG4|1ucV@#9j5M?AgO#nXE0!W=4FRJw)+@hGiMmvVscv+6x}QlyVLh4xztlj4Pc%2?h_#T${WH(zALpjeISmQoSIlmszo5*r;6atw)56lhy-dC}D23!b<g1!u1Y6USb8<2O!a2-CV<Uu#s>>Q~}S-D3PTGr_->E%}$yrKEsHzt~xQdkubSee)k3`{t@C!BFEu(3TxmR`M0-jKr*qRjpuhDV?xpoW~L!4a5tJ2Ii_xl7Hsawzuff39HOSr7ky){5y7&q~El{BsI5>#q__c>KStQ#)l~B+H1us&!%ELiX$}52LYj6r5_#z-lSbmruiyu7?-J1Hz;z%CY+Yc6kUe6S!M!ezz8A$((E8F=8tuC=c6pZdQRpNXF%5Iwef5y6wts_4FO<SiWw-U8A92MnJt^?b<Vl`GJCHGQ0<~{jxE|<!jW!L?jRssCS%u~uM&Oq-DBzB6C={;L!)<N_Whl+HHvxh$L-NiUqjO6S(^`?>94N%rv;|iqVYnlqSJ}fqpU1T*ehVbT=BV{0q2Pv_>pP{O*RWFg=&z~A&iJR@e4ef!?s$$oqc_IbgQ@TgQz;)G!vfp?=R+ejI$L^1BnTVLM}x(RI)I^s%12THK&-"
    "ZdMy=UeK+4HgF*08tzFPoJ@~?qXra?^SpSv$3?*c{bY$;BL_@MM&~Arx1D2yT=;~Q?Ga#=}&CQ@l#Fthee43O~K)R_AYK2M`_GI2sZBZLZDpZVk#t0%s7VK+`A_CQI#<&h0INK;ka;~cSCkz1s(4|1KAfnFnl@XB`RIEmHwh&Xg@yr8snCKTIrBa$pFdQk>um&k|itt&MVKmL|1-da*^fKk%yY0~ZqsY8*$H#3+x!s`hw6C=f<@7G{=ViY7%W7EnHD8)2iR@|DM{YG<|2%7j#tuKq9tBjiy}j;Y<yU>o{Jn5$T`KabN_dz()lc*u`t=^@aRx1LIteI)gw}U{J#h#@SCrW>;80HYxMWs6cuD$w2i%>yrpdLfeY$kY0>OZ9KGo5pjLy6+T68x1HOI#mmHW^1o!am9*(%+X*2Y=_sTCGZwK0qwm>p#WLgC15nzu8}p)J%s6<JV(U6@2$4WXlM%A-SFZy=<N%C~YC0*P(y6JV9LAcbm^K3&9b^u~DJ-d~)q*HefJ#yda1mO9Hh?)~)qc9IW1xrfqXIBHbLE`+$_ZC&+RO7na!N<wnl`ESkl-&k$BHy(EpDEXsW_6nT>2u72nfg~i1MIf7Yq@H3%LS(bU$Jgx*e%Cis?EDl&JU&~>VR!bOPVO=KRBv}X+oF<(JM#2!=Y3BW?vi>;(&L|K{P}s`M=#rpX;6;+Ts@{jNy3i)d|fMyD_q{!g3XL^4<`M`N=|v+)10CHchBs7XP&%THPf*Pg(WIykIDfU4mId(v<cBW*6z|Qj${ohW2IIVk&VHoXA2$0k^R2>I;m8^=c32#3%V*FV_f1@BE?w2VDLi?@El+@7{0~DRiwfkM8J=Y-hC$*cfRKuSF2iOYs5_MDN9jr37@|VJJ)LX=7sNmVjcF5AdzpsOSxqI0IUivhK)_ICw*~_aaK(*qbE&C6KUU!rAaY7B_VpiLa}P-Xfzg-"
    "QG5+dX`%+vuO@otZf^csV1n%L8Bk)XnGo!txC)H<1`sg3C@4%;+2Xn&9M8GO8V;F-&-z-qk#85X82pm=4#_hnep9e6{|vv3MF>bp`_A^)>U4VUbvV7q<&fJ#&Gfn~6x2eR=qA<ZRzOxMLcxY`lqZGFT_Z$UfgEt0_0y_--%to-@;|e#H(!=}nGDaI_w5W8gQRL5c6yJL&$WB2St^w59U&d~JY9B2)3C-_Ezz#-EQ&!w0kKC+Fky6P)%$Osh8ad$kpqusKBdBwx)up%rm&H&-&+C2=Zs-qUVgeeyj0gaP>ELkgW&uD+-HVMDD7Rf)|+hjLp+}+JI6~4j)rY_ehT{_ekIx0gp(T)^YHSAm8V8-9bl+}!VTeHzSc?M-=(iXg>!6aonWc4gbY~FOdZZi<|ta2*%nu2R(4Wz3qsIPbwOcWsE=wiBN|kp7gC7V7N#l`fldNKV1tTOX)%{Nal(A>`ihNqbKa4p$ng&6cXhEUyXj(?e=nPkA-2zIAzH%i;YPT3GF|VrR@CZg@)a6U?DbtLn2WiNvpJo6amGX%AztH9?cF*V35~ji1@`TA8}hO$x#QLK&iK&%tml>6o@TWA<EJ+zs4sCQQ$V}NIXXEx*GSv(*#)WL%R_U_Q==O$dk<!A4*j~V8QIQzVa4!{&!RhV5j#fVpA{p3>2peA?m3*Pc%3TD$-kjy_`c7=*D(CX`QwKdT$_I}z9(0m>j#%nTd6qhQ`?r?Y3a8xB4dZy%vUK?M81_(MZ3z9r+eYrxl3TK5=JLR{9|zw(OblF_@KU>^ip#8x92+H+f}oC9pVuprBM>Och?U_RPn}?3FdIg#92wNyN+Xw6&p?It2d9fsrg&wpFaPw7w@>`N{=0f)n9BQ-1B`?!uCq5+2;1LM@QGsx^r%x%`~l}X}OfL%5#;HdYW`rS80^0X4|A0FjP*QqSX*5rh?MoP#YOf$7<f-"
    "Mk>Tmfmsf3I_;f04A`A3N^L1Cj8o6juTRN#(#=ci<F{H()`;dkcyQ{TL?n;ZcP8ES9QL!5$k>^H(q;5aBA%#e6t|umvaI97F85~WE)fhPXt}0h(>B*T?vDPGk+u0XET$K?1ol_LIVtAHEXHj1y-zxCy1H8#3;)9VS{OuKk*8X<lnLwMkWUg&q*?2G&kvv9z`L{7(<H}?s?v;uGm;q4qtrDUpTdI6Y%c87X(=RHBvff$Ou<)pRm4OZf!WX3gEmyOXe}$GX=(Os#|@KHE>RHhvqF1yrYOfYSwxy7f&v0UQe=Z<4&uRVq&wLJ4nC4b7Fn`LAU08=iDxuuBEpFvp%w^(_D1{6o8$X_J^tX8eYyPpVdFHIp$3Z5;5fNc<0EU`7?U^$pXPr`dl35%WSkUEn_MvLyI+ZX_>~Y|=Pb`Y|Mw5e^iS<y<Q$&>{~q>IM{~aWldN~Gyhh)DQoTGHYEHOajYf5oCa{Ql2||EC;v-g7QHJDHwn1{UPZMpWx|=QbYfjH|wPh>}G}vKDZw(nKoO0TqX|SxhGkZ5s8=@xh=pLY(G=(YmWG5zG)Rnra6Wi7a(l@CuNsv_{ZDhL}sBPJ$8loD+fo5(Nu-47CtU?y|Ws|5d*>5?$WM1P~M&8^ggKu*<6lBS%l>4WhVWIV{wNBf5CkJ?JZM&6Vp_4VZ<J473o2O}envJ-$^-oG)Zqo5KWVs`ojf;6Y)2-xW)U|5Wr%zRw@LtO$^-m8vCJEjTHft$r%?@bQ<J?YN-e%`hdekub9H+b9UMt9_hEqF>iWR=CUh=S0Dh6kS=Nh8*BRmP6U$)>ZAwIG1`FVPQf+8mg9GSsS&4#TAG-GJ>Qx~#%nWB2efXg(g7-IJb41^DSBCoWCAQA!+J$_%TCu{!NUSn)G<GXJjFTaN_?76jgKd-"
    "sZ`gd|qZ{gu_Xh?1&EjjAflW7gAEvj+z(WF103D<v)r*HI;YJ@jgyKOg07|aPpe^a)1@YTpO+PYE2ZR@gFzk1GyKGXafR%wO3wBXZ5pTFK-XbhuY5b=9CzkDP|KA6TuwdkBRE>-*e5|itSa3h2oht0P>{Biw=3CH?-WrZoa17+&4X*;sVF^++ElZe}gMu7F~=d6A!j*+*bRCDlQ_6vQ!d-i&K+ywc5GV9+ym<tp{f{}p4ML(>xiU=qwd=dmKB|aJ|jJNgRG@Trn<;#ywuC3$t+h)@HXkoTKOee>X>Pd^kk!SPoJ54bl3|1@I*#IDUYMUbzB%i9Gc}~wpH_0YZ0NuWM5m$-vkVu5kfQx^S@%6-yqV4~<cWdFd$-23>d)}XW@E^v%ygOfL__Tr79h{}`+OpAwkPwG}2E5($>HE$4a+_cp`tQ3PylcL!6wdvPpP()=d;c+Za;_HFN}<2QNRLYUotvY!Ldn)D#dTYO&MSDL`K*?x#j814R4r%<;?i3t!0nJ(Y3UV^2+H!-y3FOOx}yF`3hOw<V-tA>b%=q{TGw$BY8$GQUrm<ll|rvw#;2rk$iZ%Q$Bbumea>OuDuj$^VP&`ZIbV3?PnNh>sas93(821=Rxh*0wE`XLuoY`3rEf1UQOaMe^}?Kf5gNWfkMsFVo9&!`ZhGgfzR}?pb4SOQm5KjRcCKrDms%d~=D6oCcZgW-zB>CiF?%wTRk~O&xrPoL#2#up;6=f>HdBOMw@6OEa#G5DxM8O6PCDioZ680JFE&qkLR3ixYngL=uhxP3Ac6Rxe+q!ofcy4s{AT~B!{Gf=e7ij{gppKLV2q*$+&K^BpSiBsG@nD+e&^&TzFE+w5r{<#lh@;c?LJ#Na;6y}ZHNe4DRHWDff?;r>GLjvSsX|wJcGvIj_;AOFSK#6K|h?J91#Esr?Rhu4~4v5QD2YefM0cab$Z?5IbikoKe9gT7AyGA%J4YFrlV<SS6c=M-"
    "Eil6qV)xLJEy{+#5+LzERPVQFKYL|;0h#q-%;~czUEq63W%->`r~l210FBf^S>)ZXzd~0tIcJF@s+uE8}C9h%|eqw;(n(NZHQpYjchC<H-*%&)f!OBVoF^`{#G<Sl-m0CQ#CiT%I2V+Gil??o}AQ5h$AqxuCZLhNwF|@-N~xvs<+Oyb3A9RapJeQgxuW)cwby<j)<xQ7hzvLQhhom^o4fn<z2DS)m8r;14K`q?1jye<vEvXY`fHmpCAX~*1%_kvyQ@n7}w*<tkpC|0ekfMk2&j4u<?&)tfz_jy-a;$wqnPul!>}|3;a6>SB;Y7zRFiJv|(~iXo^y+dorHL2>g@y=X)*4Ipo;(5y>PvqmzxgCQ@YBibiIPaxR|-rKPFgGSxdQvPyY0K>Ey!d7<dwvo7wZP!uFh@dT1oNmkg?#<-Y+3EtaTq4w-ipmLQ4I8tsIZlHi@Kv<X(xN0g{B2#WI8qgb^z}0#Y1R!ju4ei;b^)lb5hhZiy-)Jo|g*$~JodhvUR5);EJRB&8DBFZ;i!5@8$ski>kWG?qw{Nr(VC}ZanhVs3F>ho^RVK$bvW2=d_Q@JoYYbI^1uU}ntYoxcK@3}M*xO4!LzJ5woFZA04J(Hj$reCTB404kxovAik0uG3(*%agsRpzfK-P*wLyj{xJF_xKMrrkJK&-gULTt3ojg~tVqR3(FklA?_+f3Do&?^&~A-57iyf&s>k~S7)T!OdJa{_8BLLp$7)9l4sb~6oaJ<+?Fnz|kB(NPfU1ri!d&&cT_%I0_ult46~8pq<BKN=7C6!^69&mh+;jnMg^v-y|wp8ufWa^K<d`bsD&(V3_fT_*`t<taUl6fEA!#@;|j6-2qoxo=7V5$wcJ{-"
    "T1suT(I9>gsH&Oqs>uN{eZ7nFOpXnM2O1WRYYjOARw9!IlN2fC@}zjBT5c%Y{sFwP4CPa*qN5B?!o(5gpMERZPnR$EDdL#tS5*Gw2Ekfw9t$37pRk)PCo9vH$Sl!NJ?H>&y|aM2<Npb3lxG)hzRAi<0crR%zREkxGfhkW9MmuY-IOe`{QJ<C$MBPIUyv?DIW$g$U<htp%2qx&fVmQ0$HZlwf5#p`_U%4H}>mNNOcPf@vYFnRyhXrfthJnn0A5WMQ0TW<jjv?dRhM>0kAZt&6Is5|9_ZW>`L{h-QQg^ehVzfdoAe3RVc!^0F6-H-nZq$a$ls6JSVVw|>N{es&7UN2)4A?NpW2MS!5tED=!va*v6D1|JZCchR=Jr`BOP4>Cb1$|cRFl}OSvS5a0vw2eFEK?~qFqg9?-J9rfCK7uk`?1~(fDJD}tMNft-fe`(ia%HX692Z)SElTWqIfvLag|aMy$VCBUfq-n2aRRa)8(4A@3LC@3q%(3*`s~upJvU&WdZ&ptijw?i8+SA|bE4>bI1fg&Bd}M&&p}6#9>K)MxPW!DTb<_|X%CKQjYuVliA+NxQwD9=x?9m2-A$At)S*Pd@L^7L-Mf4k=KAbyq?=<PfE_v|xgx}f@7==Ny}1a)hlnx95+g~PVMyDy;U>$8OK22~O(_>!7}*_ix);E~2EmjLb^xR<R<TBDa4e}rNMfMTa*fk*f?Yj8weZx{N}&*QOoE_Mkgy|$9nA%j@<7;vj2>^*lv>=l5o4H;W|p^cbBHyVc22F1JWgIl5Zfv)W;jjknAO~caBC1a?P|6#ajd5qdJT4ncT%TKg{2N~8B5XAa0wd&5r*itNFXvp!pZ|9CWbbWjG-7-G7!#ebKA+{UOuYt!Q|JXdo!`<OVQxKQua?f3qrF|dod73?rkta-nFTdPec`1#TeT2j}w{Crze8R-"
    "6oNia_KI1t*Mj;BIQx(AuvWaVxY=hB^w!G79RlAMsZ45JrQU~EK>?r*P)TR?4+~^WEl#CQour1Pcek1Ah0MEK(K`pGP7014*c!h>)vUT81K+-b@1(@l1C`8i6W7FkKoD3mT}yAb1m&@DK!yJUSqe8mNs+P+^z~*iczHuVhmq2+6Q|mB|t`4Qj5sQ7ER9Egc2}}j06M$JF+Y)5MiPM$7T=^nWMw3YxBPSLq~J7IFAJ^3ML@V<}Wdzfw+o?jlh)da+);Jxz*fGsp3TuscI`x=R+K#Ik*nZt=AY4BhOCK$#u7CmbWPsgP{vi2RAq*xwz2SHa2tP2^2KX3uINo6U}iHHq-)K*{ZQ!>!1Q5AviNncV=VNV>gH5(B&h{PlmC~4+0<`#ayBbAfib;wqJ+USx)YaAk<#2gIe<xYg(Kg<dPhzcElT+wJFFflz3#4t>@OO3PG+}ROSs*#h1)?YZCz8aaB%YD~o8IFzeT04T#mG2Q|835SV7>oC2Kj&9HJfYqT5(ameW|3MmdjCsJFiBPg*+*$|-N^KXU`hU(xbdu>Ez%#Lk4ci;mzy~xKpH@wnzfR`R%P&`T=7CEUTw`(e<CWKR_k{TiBUD+~dAazLAONB7v2?GfSZjH&y8g<-7?d|fEPWPfWQ1)eV=(7y7rPOLwI!@0w%L?J8d#8DYzV$4e?yfAkjTNh(v+Y>k79O$+_*1;2PwCljk(r>Ws=QWkUG_{m>Ty{OeHX@TfwYdcCG~tmZQ)SU(Cm2rpTce^{pn%!pr2v7`qoL1`;RF6gSuFUTHM=TU>)<guzKh{YsQctTLRRq!oDqe48CYK1ZYfR<QNoeN5_PIlRRAhq2w+LPD7Z!z0E%7+1NS<4|5KpPWYocgwTd1)?uRL$tq$1Rb~uYh>b1#R>$#QAzV;XT`VDD{hwu&9!J~tVqyjucCh%~7(Hul@UHHfJT_!Dk9$eQfXFW&dAN9S!WIJMV=$j>*WocT;sFj+)<HX-"
    "XHWXOP9`Y!N0A_94{{&KGsSxpgYWuU_9s8JUp3kIHN)`FfuQLK(OWtP^yU001vp?T5p_&#q7B3-6iZNGP?5MTEUQH)y2Ge*v)}$7y2<XmDypg<3{ngViz1%7(5xq^H2YJ;e(%-GG?PZJbmReZm`e_Q)bb&l%;%833jz}W;zNFxq^aP-r>tKql<%1#xWL^Fki;<)rWTgQyJnn%tP~vB6M$S@+`By79dXl2fWv_C=cWwqaej5KGz?^}7^*0+`%n^(E!ng<d$+at<0p$E$q@yL5k(R*L{B<YSUh=WqotO4Rv?EePQ<CFs7c_CZosq-l(i(Epkycsk16N!KJiYSf5sY^miF6sM{cNY#48h{C^AU!q}$A=lN3fugzS(xLi=nw&u1>;s;dKgCmPV}$J3*csBqU)1JYVrP^FZ$C<PRifGn+4x4y(O75i+{5e13vcz|*@>)Viud6EZmN$&_$lG;jIp|r6E5=W@d0}X}1N0q`F`4BKG1129Wg|T?^_Am59BOGMy&$S`B{^TS=EDGt3;UaUEmW@Q+=9E42)5HVPZU*sryuBXHU3P5i2pyG&RwL$>_uWqT(NXkHKHs}P#9zNG-|otuP%xflomnBxJ2sTa2f#9u;Rh_D$_f$peFxT<OW_@TEtnZnLp>w#eA)(PYto`CHjf-bx=(t};epkNueX@jY_kkP+ikAt$8r)*#vqs-"
    "zCCc<kJis=OX$=6%JYZ{*8x+{@}P`}bB3elX}p1+cN0YTz5{~hd(TEBRUoIZP1!AqzELGG@?{~SZPhVwx!Bbb2Una!-;!JmBG&!oy7jY${W|U4*XezeVs0j)r!5Zp^k)^sse9+|J@bE0fl91ZRztl|dS6ol*r23Xt02ZfK*BFLHi%;Y>DGGT&(+z*fEKr5(_1s^H;7Z8Uo75-m^C!`7RE*(DFp?5*07`PJswz2$ii|Gc};@PPOJm50;OWo6g7c+xAGb~^ylBq!O&ovA-+6eVh5QJHT6i2j|9tI*Izc8lA1Ha7HD|dMKa)qL3?ZSvuPkeAT}MbpEpun?>{oXu5Ku^!BQ}u0t&q1p|{k(UECS(+A#M!a!iBt8O<^?>u9qhzNII(Jp?iTCim~QLqOGeT}+~-%4^<*1j{HO&8JuCq3ZF5yK=XV+<?Om3HqUwWRYKDk#m;Zj84mc2uJ&8?dALY<xg8++C1M8`7F25@7MezXBwG(pT^}*!R;f!JGt@=rr-IoKV0AScKW?NwSC`r3w!;&pX1RD9m*R#&-JSE8z6yYA@5yLB#|-GfO&|=5B7Z^#lW%0i|_bu?7>e!P(bPbmMos`=g;__?wQRVYCObl_CjCr=&AeDuq3O8BU;`=jDD>2^?V=Sy~nrr-Y&ZR;$1Kw-"
    "lF&)u2oeDi4O>F)a^k%P~E^d)I*f@TLu>PkJiJx{lcXU?(HL=*T{5zA3BDLbOUhJoIPIDh|g%qj(sqiS(6MowE2=}zw!K`z7g}9AxFEsK^~FnV8_v9`%h;;*Up_pLFu&$ueadjpW(~$s<BnNf?SMHk-y`QZOPUn>+^L&XqI{F_sOzp;u`b9q1wxNL}nWK#*jec+E7SHrd2OjKcDno*mUFjN^}anok*MtY#twt82G?nlD*wl`fQ~BZhKdUsC|E4(_}OLFVRPC5B_6o*N<i26NVqLAeBhp$nnCBzO(;R7!N7-pl`VIK6tZ?cyd`CU+d-iz#nJ?q7X?SmPQLAh{&)az+(|qV1lF~!a@=O{+}=2^#%7=m&+_4Ms#{DvdL&9f>aQwiFh;axA}gIviH^qNT?^g$e;}WKaIKPthel%5Eybjq!)lergb>@OT9)lZecIJ&LkZe<;aO7`h%DuarXRN5ZIbNSf>+B<6zqWyml8VCcAju57!qc;)9GzZ+d@km=Eb&tj~naLhKJYz<Dkw#M2PGwJuP!XbMfk+wZq8L86&GXx4Pfs6lZ=EQw!M!R&{s^RMfmf2xnQ-_x^4&%Hn?lmkK(I6G8sPqrvgzuapl>Df{j^V61?c7GAK7?T#{7;8BGod_K`)1m9@)j}!`dFNv;8c~X;i^tbbSumaXCkPagIrd(8YVC6=e*BB!><c5V@vH>nz-ZticUNiXG9Ol6*i#ZC6!vk`&v<z~Q-$GscYTK24dFbOpu*(hyNC`=Z!q5^$R5OE9@BYX<BOsb(27VY2#g|%DF~v;7D9Uvl)@^^4Y>;g<@KKm9-cJi<oo}>`JZCa@y^WV-8kljpVpIJo+iEA(?z`PJTj%$7bT6>tun{mIsoT4n5PW$bHmHrxC$=1=aQ6ngP0Z$o5k^iZ?dRAY}auGdApW-cHl?-XFzPI_(Tq(C7|d*@q<t%$rX=9jpw(w0I(-C%8w2z2z%edPamD9>NOhA(+W-"
    "`N7bzx*?8}S2n_n^1I@G1Vfno$((9N|3)w9S$Q0cL^+rG$6hj9=$|`R0=zT${u0U^$AV`xbN#eahzN$hDfCf*e(s!Hdud4a(zh>aMQKt`^<ji7n6Jw9MbbGiRF~jDGoS7Yc>()AP?e@4^$3Ef%fNR$YnR2&v_D>laj-t(WW@j)OUerftd|X^vws)pQ${rIk7cW;yi?Pjwfbk`z1G^7A#86O)Q#sY(!G~fHQ&%2Y8LLo|0MW8G3W#l#0A`pBH`IbUzuQUb%JJxQ*ie)SsEi~6_9ool67}lm)OPz5TQo)yA|0q5{Ofi{svCa5@RSi)@KbF{0)i{u1D%1fEKgw#h(LHQ@`K(**>osSs5mzk{W9k4YDZ}UJdy+z-4Qda6}GU3A(R_2gaSBH0mRL843R}Z)DsXL;fUF*$kQTJ0W#g@?%y%XA_eS0{C{m&{LCwB6D1E*L?cR&1~prVrW^2t1ftvy0m?S`JaY28<~V^+qf%5ZSSgGYRTR4Cq0&(}S(}k$iXD<33^dl{++-0R8Dhva;K06yR5fa0LR2G$3lmBF1*VMwg9}_4h5k|vw6KO6h7E{C*`ZDFbTI;#H4X3yGKNEHT5>^U0ZB0+vk@63L19+dELjQyF)B&dMIp(i6v@~D=k%00@;98J#o!)%#Xv(5DMd;!D#1xhQ*xn7v@D9s+64|xW-3-pASgzVu}N)}M-l@)CRYG1R$|-)++n_+zAQm^J$&)Vadj@nm4L`w3jvaf!VX^a(<M~WNT!%;5U2vnV5JBw5L7E-f{{d2wnH)niiwRFL#jfHA|zo1WHes<+XIXWP1-9J5eOoJF^D1%WY;QaNi`zDBMn=&T;yD`>6!W7QzAIo+R%t}fOAzv0b+~b_Os(BOTnh5Hj<(06OA(k`lrxB&o2WuE)Q;c*VLFW8LcxmCW=hUC5U3>{Ua&OLeSp^F#{Z8uxg450<L%oB&sW<3CPr&F=X2~>2g>m?bz@_78-(U1PcV5T_CA7D-"
    "ci{44tUuHSw*kp%K*&BU0KLLt}t;o|&gUwZ^#olzfqA#X9(ORdoCqWwsYEPXteBZ)TQfu?e7Ok8Rf&Tr{dGh>uIgRt1gW@qMG)&*$>s&9+Sogvrn4-iY0_@~QdN#pNUJ!>dO;)mXft4<s>nGcluJuwqm^9=l%|%t3g96@e=i0z}~bRo=?rOeXNjhrVJbIXQWAx?~p4hhJY>Ar)}XYB4b_GE2nEJEqcR$Z+)KD7YYoJ1B;ZCNsZiNRSt&W;GcD!}dps4a15H;tRljT*xPm{a;y`-83T9vDY(N=QD)QH4aXQbWD&Ikfe$b6E7Ro)tF0KhZ5x6e7M6N+;JV#a-kj7kv#EUox1hrn0Jj%j^Vg%tW4VwVq^)J4JHndc#JO`FtL(q^wNPtNlWO|?a;||?%zyN%hBBF(P;Anm%h#9MjdR%c~=>xcEk)yGuohz*IjtYY~H9iW{GWfT-lbY%ZghGgo82K^uj5{)P>5KW~_y<=f8!|3><HW9a2HB4mxe5i}T$I-QnPDld-9NPDc}#TOjqGNOyB@5xmNmt>lVbEL_9FPK4wtMWW}=Ijt<oQ%NR=n2$$ojMloo{y61}?D+xGwSd@Rsbqx0XiS~Wxpi3g5S?_1#Ll<e+OJkQU%^mkv&NJ6M7?1mR-HU0T$^BP@v+adxv=NCtLs<POr1UiZ+X4C>@a=i4o3^VhmT`tfY;dpNKSkCVeGTy7-|^-e4>LQ$p{h(VuoqSIz6U`Oi!b**;B;mZ3!`<or{Fi>T%7CHgFU&ogDY<23%=akTPMdqfNz{ig*^Cno#Nz`8(P#p~y`LNlZhcDaJj^CZJDqB2D7AGOQIrO(K;Ekq>c8Ml$BISx_u;8pJ~h3J!$0*wi)X@aKl4iX!shrQ_W=(VaCk#L~@fY%^7G@tQBUqE9`cAw5w64e!dLWkgxD$nH`_Aq3aa95Yb%NJ7A5Hyu(c?9#uJPdo?kzf{!FPgd0Q#O;S<7-"
    "3DiXlHe7Gkp?8Sq17&M(Zj50tLj^h=X!KDYpB5lib+DNF(#j_w@yCDFZ`&4B9|MlMXBXFASVtR$Bzr%*_2P+tfRzz?zb(i5g;)IY3%bQRNfU<A)5d=G`n6&(wCGAaqE)Um{29qWew1b+4@<)1c}J$QT_SuF?36)IrsQ^~hVH0FVT18}CKPMKcW@?#>h_FhHtqp3L2MG&W_s%?6`WTF6VbZ2>|k$umfq>@jRq*wO1VPU#^#$;aCv&mVB;j>71tt_gs&X06ICkvPiv>a(M-<5-e7Q5muHrbWVKoG~;p-NqQ~!`Z=%c<M^PTH>5Sd3+_N997z%nx(g6kDkc4ND$7Jjy097*LwEe#*o-0q*y(Ro?^Rkr*6a#g!;y$DIlPGA;{P-w=#*va>#VaMW`7({-yQIhJzv0_0DH3PX)E~-k7f(+^}4wDoz=x*Ta{NBtA4GS5F`_H}rVwCOqFx+gWE2=ZB9tv-7#HlGfmjJrvzOt9qD6M&Fx%hH|5x(8U^h<;>=gp|!b`+|ljU=}`myF*C!_y)5O<YmMxFW1e;!BCC}&Njl;h>J&ZCn9DC%UmP?aLpXtofM;d$pH=+P6Hzr80!c+_VC2ij3nSafthyfSk5^7FqB1EWoH@@q@(JH!%LEFnr)&e@1R{Ym0Kk)YqA|EOB&3B2_n>zP0Z0NdU<xXbSt1jF@g<V|xFwjcA2Sf1xdbNY8!$ZB8Q_!DL#M-Fj7_cpMBYKQDMW$5IPOz6HarG^%(dG*c>P_;K;h}vrWF`SL`Zv}geDw;)3{hMi`rIvP5NDMhl36JTuJSQps_t6>kb-"
    "tZ#Vg`0^UD3Sb~IrDb6DPsGIpnC%?d0PXwQLtibe=4#oC*K8?5mX2$GLOdrk&a^3+|8coqu*am4vNIN2va*P*Lg$;%e%aR);`A{n)P=@|lJFY+*fJC74L~)2G0Wl<j2tOf^+)#1x#W9LxKx9<l9~IgF9|(b*Rm0svo5-TLOF*<tA_sKtfWimhz|DF0xN+7_hYMC$R>Vnt`^kjWB?6=%RJ`aVD;80;3M4BiA?2B%I}@$NHcWyiRtQ`(&zXo2pwI^B#Jw;uqZ4%{0Aq?KBlOzg$Om-;u@)$G6@(HbsUR$oV#Tdir0ogbNzjvj*1ez%K0kZbt*Ex~O9xiMEFwz~HD#_;6=IfJ#TBYF7&bD*sYXOC6%j3z#+Zp46k(*yu@M-I79$YGWnn21hE|eW^>rSNEY!WBuGf-aTb;I!F=#xa_u}+N>b8~$0r_G)L*=RtHwyXVB#J>`EEENRBL#{`u~rKtPz4CGDFuR&NT4XB0Z0goBv`NogcOLyfd!HaBv=XpmKjEF7E9KAxf@HZ(GHysdrXz00g$+n9VxJB0J4zAS#Sg|oK5!{sC^hOyKX*cnl7fHAV?{g8g9WnH{u4u0CGHe^jt#<7;JB~+k9}51o&;Sb0|R!l_6kx3iAN$hva8xmd5*Xks#B=kA5EaV0BNQFI!9Mp3h7Z_GqmB68F3w2-DL$P*;MW3MeDiqCLC$mYP(E7(N`d6(&O%t$vj?U!ys+xsYkEP{X}lLggmQ>vQOU%KNQHl{4Gm1cE>Wk&1pE4yQLLJ@O<W89tYyplqTMp6=rw3Lcl`qv-"
    "tM`yY?-d*2t>H0BP%6Y%)@t+7N>sgkX<b`XSw6B{d+!L}JuNe1XG@o;4Z;s}0>kPCtw%zvC>n3nQNc7HzjKLgnDgO;`r8&QTSeuDfqIR?!7_Kb=Vu=+kX<mFIzY?Hy*J9p%vfvAs*z(T~<K-A17{DL6I;B77%9~z{T{c;_N#rpY^W^zE~r_rOhK=T|jS$AISO!4!ygi|2vht=1wmdo}#dSv^`cmc7bP3HBv`i*lR-jC%IgYJpuA2)FbF$vDtvymM>bkZAoJH*(Oe6(Q%tAjo_yCs)YGj3(sqw7y*3X|h6t>FcdXor||MTVHnF?HB6m8e1p<uj*0p_5TA(0J2H90O51q@FS@BZZl&Ny6WvF(aM<n?NCfREs76j5S3PL<gYn6bWe>CMDN%3YC2I>Q)=%qazdis=t=K*S;4p!#@rdhe-rRIR$Ool~mDy>CN(Z%kpq>*!0bxmhr6+z4SEBiAHwY5djeRxdhr1QX&#%H@Xg%qn-BO3CNC9@wbnonU?5X@4ZqE>94<2EZKveQy;XMw1$++b%Q7e@7oh|+Ka*uS6pO~dvzO!dIiG-"
    "wCz5fmJ<q|<SB<<pFM2DJ`n5W$Oqii-*yTF5h5^(5ELh+0-~y_41!29Aqb+9B3V1o<J=!JhtrFq&kud<_WYQ^1(Ei8&|;i2{6V(m_IHRL#eF@II<S?3`8Fi1y=*;hDx%4aB`^pCB9ND{W?Q{a6Lznw!*SU~`%L$QG&?eGSoZetLcn+WHSyGXqH~2CYb^ZtLc}A69MN12;51y~t+<l6qi(04+-b0Z#N8c^#2+ByZ5!T+YQvD6PCFbPe6`*eAQOf$Hfnq9#FOMQgz)gqjbb2DZk>7x$C4ERTcFBoG1OeTOftl#ru9&kl_YXJxE~$!qQAQ*hsnUCUoIif+utF$*VQ~d!_>p?ki*j1p(udCV*#<uS}<Z3FosaS$SC<w4))CwX~AK}XTuIM$)BVL&M4YJl4%KvSs<c@5;87QkIl<*4#hz3LWl}hjFl=QCl8;E6zUsh@-BSFW+oUZYb~lwCP_>+G@=SJu)%3^RuK9vm{1l4Nm-"
    "WQ)vYT<i7O$tZ03aX>}GF_;hHi$!9Z?4YqP+FvB48ejF~Fb@geerg5Vd%qMB8U3SjUR!FJ|jU^7v~ww6TdU<A4+iY%OSIN6-dCt(5ROFX9z)Fzp#$OJI@Fz~A{0wR(-c!G3bvQIdmbV9}oHo&X}Aj2*>1~{oj1r<<H73)42oKBsFY?nRAi;`oixHLXQ%dX%UM|2G2Q)_GuH2sJY-a{dh22BlxProUn3kxgWhn}_D9$76yPG+37hM|d+Z=80-G9r)xKv6q7IAaR4B@7uXPV22OG?nRwB*t3@F@iFM(t-gfQ7E}}0BNML*@@GE$ceK?7l$J2oL&$joHxMZS&J?tNfP?1#t>ivB$DB<KCnkBMG>m(2-{AKaUBb|$9tVLyJiTEI@T4H<a1pKb9-+^fh;p<V7;yOqiYfo)*hw6l)Q(<3Xf?HzA|!_a_VMsMKP&8sz#0Yc$J|Or<64$m~@grKA%4VxiA8uB2{UhVq+FRh9X7|eRmAucTmD@hV47$fiIE8CpUqcq*ove0K|c|4%O!aL$Mu$<<N!=F=v&mxX*rzdl}d`ueg-RD#%5G#aJv-3PDIH3IifYgn${K#Ry8JZ=D$k5Hc0hJn=QxNiE8VFk*=_5{rb(Af`!yAR~|`!LZQ9;saz89Epa(9ADXqfvB{XZ4iu9vel?40xull2*`&TP-^CZLAJ<Bkb9X6Sw%%hfx;ox+}7IEFy6y<M{`C1stI<Xi7}$co6%<@7Ym?MDwl?0L6@d;h~^O#IOC>4btK5exa1*s(E^~Q2vD@r$S$SHp-S4M1qr@tA&P+-lOkaQ%-Twly2cD@IUAE0n>>NDFzrNDhP8G~W+2JKj%+0v!!ijWz2AGLAe-!9Mxj`kZ<)y^iJ!*YAynS-"
    "VC=5R8zgE?SzdDrJg~%-;a6zLp&Z5<rLA~tyC%X-tIrO5?Q4c)Y^YWiwL(}+Njfuz2POu#!80hk*q2d>l#ro>75Ro#6zX={I%4u81vL<<A_Fd|k_BY50}LTzCqR;hu^?jrCL}5;Ay`aSL1aWBSZHBbQ&_;(lS)kil1@;#9lKL%P!=l3t(>w{CkI$g+!f7GbE2ejaU`uiSXRM_#-L~lAlM9u$c(c9pfKw5E;j|x&;%%$2;MT4?cuTkj15GPlBy6Ckgyqv0T^Lb87dVjfkdq?FkuD-S(COky5<-lpbDZyn=%z+1qD_h!2}VARs}_bQ+*f#Y$-G`BAYe<!et}4@&LY@jO<k7JX>ZqH^ep}l}cx!4}=9Ab?L`(lw0ti4i*TIG+0n68fdaYcjjtU5K@el2&JV#0}@mblNeC|Z3zSo(HUW_sWw(<42lUA2#kzutRo^gfJI0U6hyB!^zV_!gYb<}y8Zat`))ApPS(FyJ-8JIMUxQ9Sfnvsx+01q(8o+NmK$3oO@7^w^nwR`T2&%h7U1&X2hge&Av93nLS`e4X1Tp|wzlc664;dx;uwcO2Qrn5O(H`saOdo2$RY$|ATcQo-z*i_IE#$cFQS3%fcc{!0|<&pkAeDoeOz?fr5nE!XyCa>{ycUaSoJd8%&h$Pxz+UCktz5NQOhU0#+5r*4Yy$3U7x?7D?cfM8wQM`<rxuy9v!g8A)G?dkqCydq>*}QDUlLNGr;FSH+D!UC$7R7lmS2y5cv{N3khy?H@Fk93^Ihn0+L86>gG+rR&E>|SIENE41>r%|59@L-9fH>Hi6jI6BNJ{2(T-?*2EAjWQzbe^2fXGI2(i0^HvCT`+8nTP9yxCU#7|(p>Ctu2)(oJ-"
    "RX8%d#A&9DK$y+Qp6BZgiyzkn>LO|Iim8>nxUknS+<m$_g{R$w_)ddquNGC28=%V@}EBqz*(=QeqKO5J5NHc<?&9UkxnPN3D>`OhI6oW&l@G@gcmPqJtxs&(G)@0Lzz8D4;J<F6FLO30R&jWNgck_vgQ!)UdEy3fv~hU2q&i(ctALsVKVa8`1D)CZyEv6_&r}2vt;&&DeqNw?mWB@9{tI>zex;+?vGOvWc6<px|FFr{XL^{<Frv}GwZF9!7$!j>c5}p<>5I($>&P3L%ID1xZL8W*Nxug?8I+floD4Okv-B87?uXF45NMj#7E?=5f?>Itavmp%QW54aRNDkq8I8hBzMJ;YC}c9+FqJO;gttVnCTRe0CEW2CZ_F%gm&JE131(~CF-0immO|}!#ALy)Y=yyu+)Z-ACMXwSQJIFWF#Vi2s=*641va_1BOB=;vqzns{&O-VHAPNL0UrIT5vMEQ4P!@6i|0Sys9jPdd)bJ{*HZK>wd4wN%ne+uB%hKJdW~D&P*Z9qrWdHrLF5x6dEOM8ZbiG(lj!dsIm8Hr%be!)xfn7@w7E=GQbeZ6(*#xs~c=zsMw1lxXR5?*$j-vrhslG<b^;)j0J-i5W29<IjO1TmMn(W$YB-2$XIy(Q<#sWhFEE4)+(}3a);Xz>A_PB2nDV~-5C*jFv1c97htcDHx8&`<sBa#bxTDgM34lK5XVL3T^j&;wQNj2FfV^CO>S=Ygxru(MO0C!QmJZXL~$i#-FCE940JTfiExZ6t_azKL|ZMv1<FwzYY_^D&IzUzOPO-BS%`~BF<4#<=sb$*RIhU_38A|+c`)5rX`H+RkZNl1okAsKIzJyf4In6pLJ>&>rz&T=?BQ%6k%?Fj-"
    "wBl|A~FNv4etSOV#<v5rx)2}Fx3or97Ur82%}0!hhtmKQUDx|0790t%b!q&0;7;TkDr4F*Y*^1hQ1;;P5xnCgzO9+ZZ6E@cw1;Jo;YiOk-%txXFzUI*a<bHpr9;($^|O`ltY>s2@YtW+`<|D5e5(ow;QQqwC@%3#m7T8yI^SnGurMOqj5q(WClIC*+Xd#r*OzRw$6cIW=2Uy3m~us4$x(XyaE#IWiSQ+7q-<<z8Dn=6p{%=Q?TxZ9|rkX<?yim<NT@X@n>V4H+qblZ^Ku=zXrCx%G)Q1?Fm_=*Np8j=PN5$4vtVdVv-PV2c&B5WiJ^1Wxpm*9Mf*KG3J}ub(eVY0-20QZQKsK_v5WeO^c3R98UZ{sU&!=*BOr$&n?Rw-dmJAq8$+K@7Y{1Yo>ZZmi3*q(_FkajqzM}SQ9#9mXPUibUc_9Z?tL{NLcJe<aZu_9#<>EBH?l@+u|_n>`NN;8FJ@QXz9liEyws-43Z)JVoB!E?M!)st)gbwj;M)#u2nir!`rPnYqfW|*p~y6EO*yE`3s&Gfi+sS?z6qkZ;o}+@S0Z%?3jT(#6cZ&X76h{?GaB4G8PY&$a$j~qay~2GSO5t0M1_^wzy7Io8N?Qc#OiV@X^b(-0#UPOG~e5gN}Gneyg_V(vMdfR#M8CN-9*7-fB*f)|1q|1x7(6*W2Q_veY6xJRQX}{VTSv3yv4%c=J`u?AKKtyjQq>;TrL-obRUF@tkQ5dhc-"
    "iqU#kg=kYN}EQ%9cP*U8A@{zQ_?43%;JuV?lu6P<utr$j7M+3wx;s>X5XdovA<&!g{)@U@yvL%8aHbWv%oG`~M4AmQyO!r!w)|r_#SKYFXm>{uHV+IjH5=IADXVB$|!wE)?gxm~h5-y5?L@6L6laa=o&4ik;nuRzbvg&}b2&x7Al|7bnZConaTquC-+`2s%DVcU8kb%JjK8l@4%*IJOI34}NGq^*F(<@%ayx9p6_>O=^v=>KXT7m5h2JnMt8D&5=Fm;72P>?FBHr%B`D^!RMGvDMl(k4ilJd6gUOeUk-8j&fyNo%+a!9oNPY~#5w<7%Rq69irjB1j|z5*8aI6pI#1)_}=pw%}!o9s@K)L{&v7g2}2JIaGJd3GoLvRbK~(y_vHUo2YeFTjl-J_z28-%}3)<4>r&R{~;PC%3~aZNu&i617F##l&H^cszw=|xI4!(F8jim@{RU9bZ<N)x&9mNY?gBXK_X*)-h^G$Lj3pG<C;A4oX+*jY9xkOcm7W&adDu$V@WU|uK#gID!TIyE*8_ocAvc?<5t%BZI{-en^Q2)56<PK1Fu(;lsU>d5@}D_6POfW<cO-O?Ru%s9_h%{YIBbgXLR<3y@e8!ar8V_=H^K7OtX3b(4s2~*v7XPeQA%G=A}lCk2jmVa7H4cL4xK(zzQKY>vrd+6yMm|!JX;LCMNt)wtTGPRs}36r>cNzG5``J6b%!Borfh!<dtF=WsoYkYMcQ&K><VDXp~W{VWVs1%F*0qj;LE`o^?G#YB^V13Rrc`OXSw#)WvfGGQGa0o<a_po721Fbm9^TN@gdxs3NKAY2}A>LBVtgIvg4qig5{qku*}q_S}oNZtsL1(hLs{=Ew-cR!K_9H?}BE`1pA2Ko&v@Kv0SVSPDf0=d}n5Kv)X`5lBS{j1&bVkzi~ynnK-oUFek-"
    "oyIm+B+q(M1u_}|H6U`k5E0Pp@}X*kIi)kOX(aA+Rfz^t6QFK{mot#`oPvs*&W#Y`9vrC_MHMzphW!zH*4o%jbC^YuKw<%wQjA!wP!$-Ah+<UafnJG%Lm2^4AyYEdYEqD^BC3-elDbin6dPLsj<rg~5sF(a)q<I7*~7DMtsZTIx=EXi;vETGJR(z|8X(Qwbm3^^ohI?U+EqA38iP@4DOA!NIC9>FC77;AaV{8!j|dY0B(>24Hnk?tu!W&Io?B}{m|Hop=*k$R=R~Y6;Yi}dk>rLo0ILDxbrwMk3BvF;x6R?KMACMalPx?MrxBx0S}|8eLnT=bhsO)=Zs2d0p@b$I&}SfZs?0lZ2qK7{CB7D5a1I7oFj1nUsMN#&#11fN9h_A-#2x5i6i7q>Yg`Ye*!d0u5EP6AeEx5edLR80qyyb;jO@lS4w#i-O1InJh@YpjZLs~p@2>e{*Rr6(2ofR$tsLNji1@U-9+(Ye2pK42XbX-pLDV&rR2z(Zb=w1(O6p>M1bSj^H{~YJ#a0$Yw@w|e;{sqD{60R7)%kbu1?YRdXmBt(c0|dMg}sv(XJ(!MlJsH_S-3KUg2aO-;Xq1jL^{%notbj2hU6K-NhyK~z&Y?W;+|>tXy_VrA<$SY!X2?nB`rLN39LSSod!ogNO22xWDiTTP*veR2L4<kgka*1UAY4#(-DDj_(5`A)#T}!P8U1pG{g8QPTORmX|WiXhIeFw)FjeMW2th7YUF7mDw_!|ieqgFLXl0-JaH3&HY^G5D<Z=tt{l<h9wzf$wF}l7H^aLMU~W|4+}zX+gOiF3MM^Tn2+0LxgQc2%Fy8FB5GAq-"
    "NfZg@0DEXH4UjhW`)HR8kwlRh2vDTpppdkIWQ3d~-budni!#Be)l(H?jHyN*9IjqI?`MD+d(EnWA!<{54aMKvZSg#Q#-9UyF5`t8v-rya^~D2(=<`gDR8_;)QRoZC4C+<%xum6Z277kahfg4-1(Dv@S!7*Z251>kE@aPQO?%qULB%;6_Mup`H=0*oJut(2IWkNWd`D|B9j2ylqKZ1=o|`+dCXQV6XE`rcA2J#pD+;Qt&~~~O(ve-a`&a72p5fuK#DVSg=?Uo9XC0uY+pQw$unTei3mqJ!=onS2WY?@g`bYtNG!EX$^;IX^_!K^}6_B=sL-an@yR?iGR=v4%odHfjg0Ih%kQgWnBC4vgEL4mVc2C0IXZG0puH0tAt>@eL(jMh?R`#;!{TBneP1@|y{Q8GU-Ph`)YZS^mtc|h?9E}K3-vibiXDb0@`Z-J@Sbul%f&bwO2p_mv3kuR8{?F#LA}{Mv{{QVv2c~jS9_&poZ~q}Af(N@4O9KLsAc=s7v;Y&`-|zdbC*E|_1NVJYrLWvXyUk;y`~_3)sMvUh6`uA48!7nBN;OJY%)ijX`hPFu?Y|8Ee&oi!pM7$l6BY350%rVxpm6ytHes;XMI%?$-a3b;dT!;S5dZOaBvXY61LtYIFa"
)
# Exact companion review file:
# ai_scientist/fewshot_examples/132_automated_relational.json. Decoding yields
# the original JSON document; _fewshot_prompt() inserts its "review" value.
_FEWSHOT_REVIEW_B85 = (
    "LRx4!F+o`-Q(3oDgH8YiZQp=I5Kw1+kgw+Kjz9nZ`cMU4&Doit#E>XumkdY~CXGyqiIjOyNP3Kb0BMl*2B)B!n3F`v00000X`ld+5Fmjm_@|>p)b#ZrGynmnK<1d3089W(00LkDCJd;>8W@@lG}Ax;(+DvPfDl1HRJkM&slX=1;eL?Zfg}kLI{}1*mDUa{0hpG0cu<rAqP5ZD?hd{r3T`t+06_P3rDnO`VEP>vpXKO|3nGOh*o8K~BsBsh*<#AW3L$&lalPnCee?;&xXdiDb{JXD#Y{*p@#Bo?TkPa-X{M{mhHNnrE)>REIx=Cj+S_*q^Eeicy54gw7|J_^6)jOk5jn$0WJN>T*)rsaNc&t(6(We(LgQ<xOdS~9D$8Pz1R@!Pu%6nf79xQM_K6G$B#u`R>EF&caagQIYc4rQMLSn-d1KtA&FBV~TX6@w`DhGuX!GWk6658IlX^P+o5Dbh@jw9@1`i1!n`|3HW8TsrVB1H3($@{;;M#7Gr6T_CS{U0KN+CuApP-"
    "ky><{h+nHCILj7?kdC4f0dlk^hbaWzK`04a6m>JKK7VME@MuOSYOP&~|M_X{*!*PSx@B6DRzbdQEL0&nNyvj3Q6cuXj22bRi_A!d6~>GzW=b`pq9Hoq=uZ(&Puh_@P5yiq{JR)eJ+j>DCx;f*}pGL)fvg%qaPO0u~P@WWQD#WFSX=4-?bTdyXNLuH~H`4wQ7vh;}#d$(4#61x^qY05g?8{s$F)5C-7QIP)1V59Y0@D(Bj>^EJ5&L+910kK^7<kT9&Y}jLUPgON)ouQU4EV~uq!GV!r9{sl(H#bS2m{`FLC1{%op&Jt`78V*`2-5W-9EWm-6=<GZXhrCQ3L#aMkVAq@)z<C41G$E<6(&yQ$mJ006>U!bmJ#u28yM1TLlr`{hsneVDlUC9*!_J@4sBaYAQrIJZ09Mq1O56CTW^nr;N^g5EcIIZ2eVUheS=>Y#$Z4^krYlMjrTQot6VfOY*#AtOp6(^chZT%_9+MpCaT|EhicD~g`p%*i!GUF@#3mqlQkEu_-"
    "hVntJ>zT+hk7HrPM3K+HhQ+HKC(h+R3!$upJs!qsmXz5}|sxW@RoviGvYp44@8IW-8G|6z3==D5xDfYEsBW?Ap^nvbwmYv0^Sn#M)(=`#ciC*2e|Lkz9dFpI2KK1q_3$xeVlrV1~aLk#|MOC0da!!Qf&{Jb=za&B4%(!?LQylCET~UCCzrN$4qEnF!@Jsw)j8-(&ogNmRx#bclxwgD1H0J`A5S=y2OG8oHT6Id=J0r4O<dbN^>q)<`5!Ks?Jrgn2M3DQ*1D_qwSYdCQJu^zG>DTvb`8W7TPw6l5mDlQ<40>^4nncnQy%*E7)^23#?13q+*`E;189fsCC1V{u`5Se-d8b}gE(GmkYu7vKwCl9l%f9((Uo0j30lF0|N|JKlEOwl<}McG&?))y6lICkpV0xZE0Y6@Z|!PLx0w(cfdzY`2>n#6%$#D%`_>>AE60XBlc35bHc@Cz3+vjXFSyXns_!Qlt|S-RCAqQ9#x>?&753mP$%&y#&-QpeVPC-e*657ji{7P>{D%gH8"
)

_SCORE_RANGES = {
    "Originality": (1, 4),
    "Quality": (1, 4),
    "Clarity": (1, 4),
    "Significance": (1, 4),
    "Soundness": (1, 4),
    "Presentation": (1, 4),
    "Contribution": (1, 4),
    "Overall": (1, 10),
    "Confidence": (1, 5),
}
_TEXT_FIELDS = ("Summary",)
_LIST_FIELDS = ("Strengths", "Weaknesses", "Questions", "Limitations")
_REQUIRED_FIELDS = (
    "Summary",
    "Strengths",
    "Weaknesses",
    "Originality",
    "Quality",
    "Clarity",
    "Significance",
    "Questions",
    "Limitations",
    "Ethical Concerns",
    "Soundness",
    "Presentation",
    "Contribution",
    "Overall",
    "Confidence",
    "Decision",
)


class InvalidReviewOutputError(ValueError):
    """A generated reviewer response did not satisfy the review contract."""


def _decode_b85(value: str) -> str:
    return bz2.decompress(base64.b85decode(value)).decode("utf-8")


def _fewshot_prompt() -> str:
    paper_text = _decode_b85(_FEWSHOT_PAPER_B85)
    review_payload = json.loads(_decode_b85(_FEWSHOT_REVIEW_B85))
    return f"""
Below are some sample reviews, copied from previous machine learning conferences.
Note that while each review is formatted differently according to each reviewer's style, the reviews are well-structured and therefore easy to navigate.

Paper:

```
{paper_text}
```

Review:

```
{review_payload["review"]}
```
"""


def extract_json_between_markers(llm_output: str) -> dict[str, Any] | None:
    """Extract the first strict JSON object from an LLM response.

    Responses may contain thoughts, Markdown fences, or other surrounding text.
    ``JSONDecoder.raw_decode`` handles nested structures and braces inside quoted
    strings without attempting to repair or execute non-JSON input.
    """
    text = llm_output.lstrip("\ufeff").strip()
    decoder = json.JSONDecoder()

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, RecursionError):
        parsed = None
    else:
        return parsed if isinstance(parsed, dict) else None

    for start, character in enumerate(text):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text, start)
        except (json.JSONDecodeError, RecursionError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _validate_review(review: dict[str, Any] | None, *, context: str) -> dict[str, Any]:
    if review is None:
        raise InvalidReviewOutputError(f"{context} did not contain valid JSON")
    missing = [field for field in _REQUIRED_FIELDS if field not in review]
    if missing:
        raise InvalidReviewOutputError(
            f"{context} is missing required fields: {', '.join(missing)}"
        )
    for field in _TEXT_FIELDS:
        if not isinstance(review[field], str):
            raise InvalidReviewOutputError(f"{context} field {field!r} must be text")
    for field in _LIST_FIELDS:
        value = review[field]
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise InvalidReviewOutputError(
                f"{context} field {field!r} must be a list of strings"
            )
    if not isinstance(review["Ethical Concerns"], bool):
        raise InvalidReviewOutputError(
            f"{context} field 'Ethical Concerns' must be boolean"
        )
    if not isinstance(review["Decision"], str) or review["Decision"] not in {
        "Accept",
        "Reject",
    }:
        raise InvalidReviewOutputError(
            f"{context} field 'Decision' must be Accept or Reject"
        )
    for field, (minimum, maximum) in _SCORE_RANGES.items():
        value = review[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvalidReviewOutputError(f"{context} field {field!r} must be numeric")
        if not minimum <= value <= maximum:
            raise InvalidReviewOutputError(
                f"{context} field {field!r} is outside {minimum}..{maximum}"
            )
    return review


def _record_review_cost(
    response: object | None,
    *,
    stage: str,
    failed: bool = False,
) -> None:
    try:
        repo_root = str(Path(__file__).resolve().parents[1])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from src.investigation.ai_review_cost import record_review_api_cost

        record_review_api_cost(response, stage=stage, model=MODEL, failed=failed)
    except Exception as exc:
        print(f"[cost] warning: accounting unavailable ({type(exc).__name__})", file=sys.stderr)


@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.APITimeoutError, openai.InternalServerError),
    max_tries=6,
)
def _get_response(
    prompt: str,
    *,
    system_message: str,
    client: Any,
    cost_stage: str = "review",
    seed: int = 0,
) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            temperature=1,
            n=1,
            seed=seed,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )
    except Exception:
        _record_review_cost(None, stage=cost_stage, failed=True)
        raise
    _record_review_cost(response, stage=cost_stage)
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise InvalidReviewOutputError("OpenAI returned an empty response")
    return content


def _review_prompt(text: str) -> str:
    return (
        NEURIPS_FORM
        + _fewshot_prompt()
        + f"""
Here is the paper you are asked to review:
```
{text}
```"""
    )


def _meta_review_prompt(reviews: list[dict[str, Any]]) -> str:
    review_text = ""
    for index, review in enumerate(reviews):
        review_text += f"""
Review {index + 1}/{len(reviews)}:
```
{json.dumps(review)}
```
"""
    return NEURIPS_FORM + review_text


def _request_valid_review(
    prompt: str,
    *,
    system_message: str,
    client: Any,
    cost_stage: str,
    context: str,
) -> dict[str, Any]:
    for attempt in range(REVIEW_ATTEMPTS):
        try:
            response = _get_response(
                prompt if attempt == 0 else prompt + REVIEW_RETRY_CORRECTION,
                system_message=system_message,
                client=client,
                cost_stage=cost_stage,
                seed=attempt,
            )
            return _validate_review(extract_json_between_markers(response), context=context)
        except InvalidReviewOutputError:
            if attempt + 1 == REVIEW_ATTEMPTS:
                raise
            print(
                f"aireview: invalid {context}; "
                f"retrying {context} (attempt {attempt + 2}/{REVIEW_ATTEMPTS})",
                file=sys.stderr,
            )
    raise AssertionError("review attempt loop did not return or raise")


def _review_text_once(text: str, *, client: Any) -> dict[str, Any]:
    """Run one complete five-reviewer and meta-review ensemble."""
    prompt = _review_prompt(text)
    reviews = []
    for index in range(REVIEWER_COUNT):
        reviewer_number = index + 1
        reviews.append(
            _request_valid_review(
                prompt,
                system_message=REVIEWER_SYSTEM_PROMPT,
                client=client,
                cost_stage=f"reviewer_{reviewer_number}",
                context=f"review {reviewer_number}",
            )
        )

    meta_review = _request_valid_review(
        _meta_review_prompt(reviews),
        system_message=META_REVIEWER_SYSTEM_PROMPT.format(reviewer_count=len(reviews)),
        client=client,
        cost_stage="meta_review",
        context="meta-review",
    )
    for score in _SCORE_RANGES:
        meta_review[score] = int(round(np.mean([review[score] for review in reviews])))
    return meta_review


def review_text(text: str, *, client: Any) -> dict[str, Any]:
    """Review manuscript text, retrying only malformed reviewer outputs."""
    if len(text) < MIN_PAPER_CHARS:
        raise ValueError(f"paper text must contain at least {MIN_PAPER_CHARS} characters")
    return _review_text_once(text, client=client)


def _validated_pdf_path(pdf_path: str | Path) -> Path:
    path = Path(pdf_path).expanduser().resolve()
    if path.suffix.lower() != ".pdf":
        raise ValueError("input must have a .pdf extension")
    if not path.is_file():
        raise FileNotFoundError(f"PDF does not exist: {path}")
    with path.open("rb") as stream:
        if stream.read(5) != b"%PDF-":
            raise ValueError("input does not have a valid PDF header")
    return path


def _extract_with_pymupdf(path: Path) -> str:
    with pymupdf.open(path) as document:
        return "".join(page.get_text() for page in document)


def load_paper(pdf_path: str | Path) -> str:
    """Extract PDF text using the same ordered fallback chain as the source reviewer."""
    path = _validated_pdf_path(pdf_path)
    extractors = (
        lambda: pymupdf4llm.to_markdown(str(path)),
        lambda: _extract_with_pymupdf(path),
        lambda: "".join(page.extract_text() or "" for page in PdfReader(path).pages),
    )
    for extract in extractors:
        try:
            text = extract()
        except Exception:
            text = ""
        if len(text) >= MIN_PAPER_CHARS:
            return text
    raise ValueError(f"could not extract at least {MIN_PAPER_CHARS} characters from the PDF")


def review_pdf(pdf_path: str | Path, *, client: Any | None = None) -> dict[str, Any]:
    """Review a PDF with the paper's final Automated Reviewer configuration."""
    paper_text = load_paper(pdf_path)
    if client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not configured")
        client = OpenAI(api_key=api_key)
    return review_text(paper_text, client=client)


def _write_json_atomic(path: Path, payload: str) -> None:
    target = path.expanduser().resolve()
    if not target.parent.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {target.parent}")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            stream.write(payload)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary_path = Path(stream.name)
        os.replace(temporary_path, target)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aireview",
        description="Review a PDF with the paper-faithful AI Scientist Automated Reviewer.",
    )
    parser.add_argument("pdf_path", type=Path, help="Path to the manuscript PDF")
    parser.add_argument("--output", type=Path, help="Also write the final review JSON here")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the standalone reviewer CLI."""
    args = _build_parser().parse_args(argv)
    try:
        review = review_pdf(args.pdf_path)
        payload = json.dumps(review, indent=2)
        if args.output is not None:
            _write_json_atomic(args.output, payload)
        print(payload)
        return 0
    except openai.OpenAIError:
        print("aireview: OpenAI request failed", file=sys.stderr)
    except (OSError, ValueError) as exc:
        print(f"aireview: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
