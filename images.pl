# LaTeX2HTML 2018 (Released Feb 1, 2018)
# Associate images original text with physical files.


$key = q/(r,theta,varphi);MSF=1.6;AAT/;
$cached_env_img{$key} = q|<IMG
 WIDTH="55" HEIGHT="31" ALIGN="MIDDLE" BORDER="0"
 SRC="|."$dir".q|img4.png"
 ALT="$(r,\theta,\varphi)$">|; 

$key = q/A;MSF=1.6;AAT/;
$cached_env_img{$key} = q|<IMG
 WIDTH="16" HEIGHT="15" ALIGN="BOTTOM" BORDER="0"
 SRC="|."$dir".q|img2.png"
 ALT="$A$">|; 

$key = q/A=[-pi,pi]times[-1,1];MSF=1.6;AAT/;
$cached_env_img{$key} = q|<IMG
 WIDTH="148" HEIGHT="31" ALIGN="MIDDLE" BORDER="0"
 SRC="|."$dir".q|img1.png"
 ALT="$A = [-\pi,\pi] \times [-1,1]$">|; 

$key = q/r;MSF=1.6;AAT/;
$cached_env_img{$key} = q|<IMG
 WIDTH="11" HEIGHT="14" ALIGN="BOTTOM" BORDER="0"
 SRC="|."$dir".q|img5.png"
 ALT="$r$">|; 

$key = q/{displaymath}(x,y)mapsto(r,x,sin(y))nonumber.{displaymath};MSF=1.6;AAT/;
$cached_env_img{$key} = q|<IMG
 WIDTH="147" HEIGHT="38" BORDER="0"
 SRC="|."$dir".q|img3.png"
 ALT="\begin{displaymath}
(x,y) \mapsto (r,x,\sin(y)) \nonumber.
\end{displaymath}">|; 

1;

